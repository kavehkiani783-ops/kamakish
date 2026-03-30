import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, D)
    attention_mask: (B, T) with 1 for real tokens, 0 for padding
    """
    mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
    summed = (x * mask).sum(dim=1)               # (B, D)
    denom = mask.sum(dim=1).clamp_min(1.0)       # (B, 1)
    return summed / denom


def sparse_topk_softmax(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """
    scores: (B, T, S)
    topk = 0 means dense softmax
    """
    if topk <= 0 or topk >= scores.size(-1):
        return F.softmax(scores, dim=-1)

    top_vals, top_idx = torch.topk(scores, k=topk, dim=-1)
    masked = torch.full_like(scores, float("-inf"))
    masked.scatter_(-1, top_idx, top_vals)
    return F.softmax(masked, dim=-1)


class HUBNETBlockV1(nn.Module):
    """
    HUBNET-v1 baseline model.

    This version keeps the original EMA-style memory update:
        M_new = decay * M + (1 - decay) * write

    Optional gate:
        write = sigmoid(W_g(write)) * write

    Expected input:
      input_ids: (B, T)
      attention_mask: (B, T)

    Output:
      logits: (B, C)
    """

    def __init__(self, d_model: int, num_classes: int, config):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.mem_slots = config.mem_slots
        self.steps = config.steps
        self.decay = config.decay
        self.use_gate = config.gate
        self.topk = config.topk
        self.score_clip = config.score_clip

        self.token_emb = nn.Embedding(config.vocab_size, d_model)
        self.pos_emb = nn.Embedding(config.max_len, d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.memory_init = nn.Parameter(torch.randn(1, self.mem_slots, d_model) * 0.02)

        # token -> memory
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.write_proj = nn.Linear(d_model, d_model)

        if self.use_gate:
            self.gate_proj = nn.Linear(d_model, d_model)

        # memory -> token readout
        self.read_q_proj = nn.Linear(d_model, d_model)
        self.read_k_proj = nn.Linear(d_model, d_model)
        self.read_v_proj = nn.Linear(d_model, d_model)
        self.fuse_proj = nn.Linear(d_model * 2, d_model)

        self.token_ln = nn.LayerNorm(d_model)
        self.memory_ln = nn.LayerNorm(d_model)

        self.out_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def _expand_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.memory_init.expand(batch_size, -1, -1).to(device)

    def _token_memory_scores(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(tokens)   # (B, T, D)
        k = self.k_proj(memory)   # (B, S, D)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_model)
        if self.score_clip is not None and self.score_clip > 0:
            scores = scores.clamp(min=-self.score_clip, max=self.score_clip)
        return scores

    def _masked_binding(self, scores: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = sparse_topk_softmax(scores, self.topk)
        token_mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        weights = weights * token_mask
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = weights / denom
        return weights

    def _memory_write(self, tokens: torch.Tensor, binding: torch.Tensor) -> torch.Tensor:
        v = self.v_proj(tokens)  # (B, T, D)
        write = torch.matmul(binding.transpose(1, 2), v)  # (B, S, D)

        slot_usage = binding.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)  # (B, S, 1)
        write = write / slot_usage

        write = self.write_proj(write)

        if self.use_gate:
            gate = torch.sigmoid(self.gate_proj(write))
            write = gate * write

        return write

    def _update_memory(self, memory: torch.Tensor, write: torch.Tensor) -> torch.Tensor:
        new_memory = self.decay * memory + (1.0 - self.decay) * write
        new_memory = self.memory_ln(new_memory)
        return new_memory

    def _read_from_memory(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        q = self.read_q_proj(tokens)   # (B, T, D)
        k = self.read_k_proj(memory)   # (B, S, D)
        v = self.read_v_proj(memory)   # (B, S, D)

        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_model)
        if self.score_clip is not None and self.score_clip > 0:
            scores = scores.clamp(min=-self.score_clip, max=self.score_clip)

        read_weights = sparse_topk_softmax(scores, self.topk)
        token_mask = attention_mask.unsqueeze(-1).float()
        read_weights = read_weights * token_mask
        denom = read_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        read_weights = read_weights / denom

        context = torch.matmul(read_weights, v)  # (B, T, D)
        fused = torch.cat([tokens, context], dim=-1)
        fused = self.fuse_proj(fused)
        return fused

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

        tokens = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        tokens = self.embed_dropout(tokens)
        tokens = self.token_ln(tokens)

        memory = self._expand_memory(bsz, device)

        for _ in range(self.steps):
            scores = self._token_memory_scores(tokens, memory)      # (B, T, S)
            binding = self._masked_binding(scores, attention_mask)  # (B, T, S)
            write = self._memory_write(tokens, binding)             # (B, S, D)
            memory = self._update_memory(memory, write)             # (B, S, D)

        tokens_ctx = self._read_from_memory(tokens, memory, attention_mask)
        pooled = masked_mean(tokens_ctx, attention_mask)
        pooled = self.out_dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class HUBNETBlockV2(nn.Module):
    """
    HUBNET-v2:
    - residual memory updates
    - learned write gate
    - optional memory normalisation
    - keeps the same high-level HUBNET idea but avoids excessive memory mixing

    Expected input:
      input_ids: (B, T)
      attention_mask: (B, T) with 1=real token, 0=pad

    Output:
      logits: (B, C)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        d_model = config.d_model
        self.d_model = d_model
        self.mem_slots = config.mem_slots
        self.steps = config.steps
        self.topk = config.topk
        self.score_clip = config.score_clip

        # embeddings
        self.token_emb = nn.Embedding(config.vocab_size, d_model)
        self.pos_emb = nn.Embedding(config.max_len, d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # learned initial memory template
        self.memory_init = nn.Parameter(
            torch.randn(1, config.mem_slots, d_model) * config.init_scale
        )

        # token -> memory projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # write path
        self.write_proj = nn.Linear(d_model, d_model)
        self.write_gate_proj = nn.Linear(d_model * 2, d_model)

        # read path
        self.read_q_proj = nn.Linear(d_model, d_model)
        self.read_k_proj = nn.Linear(d_model, d_model)
        self.read_v_proj = nn.Linear(d_model, d_model)
        self.fuse_proj = nn.Linear(d_model * 2, d_model)

        # optional normalisation
        self.token_ln = nn.LayerNorm(d_model)
        self.memory_ln = nn.LayerNorm(d_model)
        self.slot_ln = nn.LayerNorm(d_model)

        # classifier head
        self.out_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(d_model, config.num_classes)

    def _expand_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.memory_init.expand(batch_size, -1, -1).to(device)

    def _token_memory_scores(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        q = self.q_proj(tokens)   # (B, T, D)
        k = self.k_proj(memory)   # (B, S, D)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_model)
        if self.score_clip is not None and self.score_clip > 0:
            scores = scores.clamp(min=-self.score_clip, max=self.score_clip)
        return scores

    def _masked_binding(
        self,
        scores: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        weights = sparse_topk_softmax(scores, self.topk)

        token_mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        weights = weights * token_mask

        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = weights / denom
        return weights

    def _memory_write(
        self,
        tokens: torch.Tensor,
        binding: torch.Tensor
    ) -> torch.Tensor:
        v = self.v_proj(tokens)  # (B, T, D)

        write = torch.matmul(binding.transpose(1, 2), v)  # (B, S, D)

        slot_usage = binding.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)  # (B, S, 1)
        write = write / slot_usage

        write = self.write_proj(write)
        return write

    def _update_memory(
        self,
        memory: torch.Tensor,
        write: torch.Tensor
    ) -> torch.Tensor:
        if self.config.use_write_gate:
            gate_inp = torch.cat([memory, write], dim=-1)  # (B, S, 2D)
            gate = torch.sigmoid(self.write_gate_proj(gate_inp))  # (B, S, D)
        else:
            gate = 1.0

        delta = gate * write

        if self.config.use_residual_update:
            new_memory = memory + delta
        else:
            new_memory = self.config.decay * memory + (1.0 - self.config.decay) * delta

        if self.config.use_slot_layernorm:
            new_memory = self.slot_ln(new_memory)

        if self.config.use_memory_norm:
            new_memory = self.memory_ln(new_memory)

        return new_memory

    def _read_from_memory(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        q = self.read_q_proj(tokens)   # (B, T, D)
        k = self.read_k_proj(memory)   # (B, S, D)
        v = self.read_v_proj(memory)   # (B, S, D)

        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_model)
        if self.score_clip is not None and self.score_clip > 0:
            scores = scores.clamp(min=-self.score_clip, max=self.score_clip)

        read_weights = sparse_topk_softmax(scores, self.topk)

        token_mask = attention_mask.unsqueeze(-1).float()
        read_weights = read_weights * token_mask
        denom = read_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        read_weights = read_weights / denom

        context = torch.matmul(read_weights, v)  # (B, T, D)

        fused = torch.cat([tokens, context], dim=-1)
        fused = self.fuse_proj(fused)
        return fused

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

        tokens = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        tokens = self.embed_dropout(tokens)
        tokens = self.token_ln(tokens)

        memory = self._expand_memory(bsz, device)

        for _ in range(self.steps):
            scores = self._token_memory_scores(tokens, memory)
            binding = self._masked_binding(scores, attention_mask)
            write = self._memory_write(tokens, binding)
            memory = self._update_memory(memory, write)

        tokens_ctx = self._read_from_memory(tokens, memory, attention_mask)
        pooled = masked_mean(tokens_ctx, attention_mask)
        pooled = self.out_dropout(pooled)
        logits = self.classifier(pooled)
        return logits
