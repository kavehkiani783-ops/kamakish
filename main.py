import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TMRModel(nn.Module):
    """
    Numerically-safe Token–Memory Resonance (TMR)

    Input:
      - input_ids: (B,T) int64
      - attention_mask: (B,T) with 1=real token, 0=pad

    Output:
      - logits: (B,C)
      - optionally (logits, delta_norms) where delta_norms is (K,) tensor
    """

    def __init__(self, d_model: int, num_classes: int, config):
        super().__init__()
        self.cfg = config
        self.d_model = d_model
        self.num_classes = num_classes

        self.vocab_size = int(getattr(config, "vocab_size", 30522))
        self.max_len = int(getattr(config, "max_len", 512))

        self.mem_slots = int(getattr(config, "mem_slots", 64))
        self.steps = int(getattr(config, "steps", 4))
        self.dropout_p = float(getattr(config, "dropout", 0.1))

        self.decay = float(getattr(config, "decay", 0.9))          # λ in (0,1)
        self.gate = bool(getattr(config, "gate", True))
        self.topk = int(getattr(config, "topk", 0))                # 0 => softmax
        self.score_clip = float(getattr(config, "score_clip", 20.0))  # helps stability

        self.tok_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.max_len, d_model)
        self.dropout = nn.Dropout(self.dropout_p)

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        if self.gate:
            self.gate_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self.mem_init = nn.Parameter(torch.randn(self.mem_slots, d_model) * 0.02)

        self.eps = 1e-9

    def _embed(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        if T > self.max_len:
            input_ids = input_ids[:, : self.max_len]
            T = input_ids.shape[1]
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        return self.dropout(x)

    def _safe_masked_softmax(self, scores: torch.Tensor, attention_mask: torch.Tensor):
        """
        scores: (B,T,S)
        attention_mask: (B,T) with 1=token, 0=pad

        Returns weights (B,T,S) with guaranteed finite values.
        """
        if attention_mask is None:
            return F.softmax(scores, dim=-1)

        # ensure mask is {0,1} float
        m = (attention_mask > 0).to(scores.dtype)  # (B,T)
        # if any row has all pads, fix it later
        valid_counts = m.sum(dim=1, keepdim=True)  # (B,1)

        # mask pads by -inf, but NEVER allow all -inf for a row
        mask3 = m.unsqueeze(-1)  # (B,T,1)
        masked_scores = scores.masked_fill(mask3 == 0, float("-inf"))

        # Detect "all-pad" sequences (rare but fatal)
        all_pad = (valid_counts.squeeze(1) == 0)  # (B,)
        if all_pad.any():
            # For those sequences, treat all positions as valid tokens (uniform over T)
            # so softmax won't become NaN.
            masked_scores[all_pad] = scores[all_pad]

        w = F.softmax(masked_scores, dim=-1)

        # Softmax can still produce NaNs if input was all -inf somewhere; kill them
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        # Renormalise (important if we zeroed NaNs)
        denom = w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        w = w / denom
        return w

    def _topk_binding(self, scores: torch.Tensor, k: int):
        """
        scores: (B,T,S)
        Keep top-k per token, softmax on sparse scores.
        """
        vals, idx = torch.topk(scores, k=k, dim=-1)
        sparse = torch.full_like(scores, float("-inf"))
        sparse.scatter_(-1, idx, vals)
        w = F.softmax(sparse, dim=-1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        denom = w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return w / denom

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, return_deltas: bool = False):
        x = self._embed(input_ids)  # (B,T,d)
        B, T, d = x.shape

        # Prepare memory
        M = self.mem_init.unsqueeze(0).expand(B, self.mem_slots, d).contiguous()  # (B,S,d)

        Qt = self.Wq(x)
        Vt = self.Wv(x)

        delta_norms = []

        for _ in range(self.steps):
            Km = self.Wk(M)

            # scores (B,T,S)
            scores = torch.einsum("btd,bsd->bts", Qt, Km) / math.sqrt(d)
            scores = scores.clamp(-self.score_clip, self.score_clip)

            # binding weights (B,T,S)
            if self.topk and self.topk > 0:
                W = self._topk_binding(scores, k=self.topk)
            else:
                W = self._safe_masked_softmax(scores, attention_mask)

            # write to memory (B,S,d)
            write = torch.einsum("bts,btd->bsd", W, Vt)

            if self.gate:
                g = torch.sigmoid(self.gate_proj(M))
                write = g * write

            # update
            M_new = self.decay * M + (1.0 - self.decay) * write

            # stabilise: clamp to prevent blow-up
            M_new = torch.clamp(M_new, -50.0, 50.0)

            delta = (M_new - M).norm(dim=-1).mean()
            delta_norms.append(torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0))

            M = M_new

        # final readout
        Km = self.Wk(M)
        scores = torch.einsum("btd,bsd->bts", Qt, Km) / math.sqrt(d)
        scores = scores.clamp(-self.score_clip, self.score_clip)

        if self.topk and self.topk > 0:
            W = self._topk_binding(scores, k=self.topk)
        else:
            W = self._safe_masked_softmax(scores, attention_mask)

        token_context = torch.einsum("bts,bsd->btd", W, M)

        # masked mean pool
        if attention_mask is None:
            pooled = token_context.mean(dim=1)
        else:
            m = (attention_mask > 0).to(token_context.dtype).unsqueeze(-1)
            pooled = (token_context * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        pooled = self.norm(pooled)
        logits = self.classifier(pooled)

        # final safeguard: no NaNs
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        if return_deltas:
            return logits, torch.stack(delta_norms).detach()

        return logits
