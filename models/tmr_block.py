import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TMRModel(nn.Module):
    """
    Token–Memory Resonance (TMR)

    Constructor signature is intentionally:
        TMRModel(d_model, num_classes, config)

    Expected config fields:
        vocab_size: int
        max_len: int = 512
        mem_slots: int = 64
        steps: int = 4
        decay: float = 0.9
        gate: bool = False
        topk: int = 0              # 0 => softmax binding
        dropout: float = 0.1
        score_clip: float = 20.0
    """

    def __init__(self, d_model: int, num_classes: int, config):
        super().__init__()

        self.cfg = config
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)

        self.vocab_size = int(getattr(config, "vocab_size", 30522))
        self.max_len = int(getattr(config, "max_len", 512))

        self.mem_slots = int(getattr(config, "mem_slots", 64))
        self.steps = int(getattr(config, "steps", 4))
        self.dropout_p = float(getattr(config, "dropout", 0.1))

        self.decay = float(getattr(config, "decay", 0.9))
        self.gate = bool(getattr(config, "gate", False))
        self.topk = int(getattr(config, "topk", 0))
        self.score_clip = float(getattr(config, "score_clip", 20.0))

        self.eps = 1e-9

        # Embeddings
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        self.dropout = nn.Dropout(self.dropout_p)

        # Token -> memory projections
        self.Wq = nn.Linear(self.d_model, self.d_model, bias=False)
        self.Wk = nn.Linear(self.d_model, self.d_model, bias=False)
        self.Wv = nn.Linear(self.d_model, self.d_model, bias=False)

        # Optional gate
        if self.gate:
            self.gate_proj = nn.Linear(self.d_model, self.d_model)

        # Memory init
        self.mem_init = nn.Parameter(torch.randn(self.mem_slots, self.d_model) * 0.02)

        # Output head
        self.norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, T)
        returns: (B, T, d)
        """
        B, T = input_ids.shape

        if T > self.max_len:
            input_ids = input_ids[:, :self.max_len]
            T = input_ids.shape[1]

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        return self.dropout(x)

    def _safe_masked_softmax(self, scores: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        scores: (B, T, S)
        attention_mask: (B, T), 1 for valid token, 0 for padding
        returns: (B, T, S)
        """
        if attention_mask is None:
            w = F.softmax(scores, dim=-1)
            w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            denom = w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            return w / denom

        m = (attention_mask > 0).to(scores.dtype)  # (B, T)
        valid_counts = m.sum(dim=1, keepdim=True)  # (B, 1)

        mask3 = m.unsqueeze(-1)  # (B, T, 1)
        masked_scores = scores.masked_fill(mask3 == 0, float("-inf"))

        all_pad = (valid_counts.squeeze(1) == 0)  # (B,)
        if all_pad.any():
            masked_scores[all_pad] = scores[all_pad]

        w = F.softmax(masked_scores, dim=-1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        denom = w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return w / denom

    def _topk_binding(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """
        scores: (B, T, S)
        returns sparse softmax weights over top-k slots
        """
        k = max(1, min(k, scores.size(-1)))
        vals, idx = torch.topk(scores, k=k, dim=-1)
        sparse = torch.full_like(scores, float("-inf"))
        sparse.scatter_(-1, idx, vals)
        w = F.softmax(sparse, dim=-1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        denom = w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return w / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_deltas: bool = False
    ):
        """
        input_ids: (B, T)
        attention_mask: (B, T)
        """
        x = self._embed(input_ids)  # (B, T, d)
        B, T, d = x.shape

        # Initial memory per batch
        M = self.mem_init.unsqueeze(0).expand(B, self.mem_slots, d).contiguous()  # (B, S, d)

        # Token projections computed once
        Qt = self.Wq(x)  # (B, T, d)
        Vt = self.Wv(x)  # (B, T, d)

        delta_norms = []

        # Settling loop
        for _ in range(self.steps):
            Km = self.Wk(M)  # (B, S, d)

            # Token -> memory scores
            scores = torch.einsum("btd,bsd->bts", Qt, Km) / math.sqrt(d)
            scores = scores.clamp(-self.score_clip, self.score_clip)

            if self.topk > 0:
                W = self._topk_binding(scores, self.topk)
            else:
                W = self._safe_masked_softmax(scores, attention_mask)

            # Write token values into memory slots
            write = torch.einsum("bts,btd->bsd", W, Vt)  # (B, S, d)

            if self.gate:
                g = torch.sigmoid(self.gate_proj(M))
                write = g * write

            M_new = self.decay * M + (1.0 - self.decay) * write
            M_new = torch.clamp(M_new, -50.0, 50.0)

            delta = (M_new - M).norm(dim=-1).mean()
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            delta_norms.append(delta)

            M = M_new

        # Final read from memory back to token context
        Km = self.Wk(M)
        scores = torch.einsum("btd,bsd->bts", Qt, Km) / math.sqrt(d)
        scores = scores.clamp(-self.score_clip, self.score_clip)

        if self.topk > 0:
            W = self._topk_binding(scores, self.topk)
        else:
            W = self._safe_masked_softmax(scores, attention_mask)

        token_context = torch.einsum("bts,bsd->btd", W, M)  # (B, T, d)

        # Masked mean pooling
        if attention_mask is None:
            pooled = token_context.mean(dim=1)
        else:
            m = (attention_mask > 0).to(token_context.dtype).unsqueeze(-1)  # (B, T, 1)
            pooled = (token_context * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        pooled = self.norm(pooled)
        logits = self.classifier(pooled)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        if return_deltas:
            if len(delta_norms) == 0:
                return logits, torch.zeros(1, device=logits.device)
            return logits, torch.stack(delta_norms).detach()

        return logits
