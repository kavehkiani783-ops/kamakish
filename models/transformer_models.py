import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseTransformerClassifier(nn.Module):
    """
    Shared transformer encoder + masked pooling + classifier.
    Accepts input_ids (B,T) and attention_mask (B,T) where 1=token, 0=pad.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        nhead: int,
        num_layers: int,
        dim_ff: int,
        dropout: float = 0.1,
        vocab_size: int = 30522,
        max_len: int = 512,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,   # IMPORTANT: (B,T,d)
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        input_ids: (B,T) int64
        attention_mask: (B,T) with 1 for valid tokens, 0 for pads
        """
        B, T = input_ids.shape
        if T > self.max_len:
            # hard safety
            input_ids = input_ids[:, : self.max_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_len]
            T = input_ids.shape[1]

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)  # (B,T,d)
        x = self.drop(x)

        # src_key_padding_mask expects: (B,T) bool, True for PAD positions
        if attention_mask is None:
            key_padding_mask = None
        else:
            # Ensure strict boolean. pad => True
            key_padding_mask = (attention_mask == 0)

        out = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B,T,d)

        # masked mean pool
        if attention_mask is None:
            pooled = out.mean(dim=1)
        else:
            m = attention_mask.to(out.dtype).unsqueeze(-1)  # (B,T,1)
            pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        pooled = self.norm(pooled)
        logits = self.classifier(pooled)
        return logits


class TinyTransformer(_BaseTransformerClassifier):
    """
    Lightweight attention baseline.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float = 0.1,
        vocab_size: int = 30522,
        max_len: int = 512,
        pad_id: int = 0,
    ):
        # small encoder: 2 layers, 4 heads
        super().__init__(
            d_model=d_model,
            num_classes=num_classes,
            nhead=4,
            num_layers=2,
            dim_ff=d_model * 4,
            dropout=dropout,
            vocab_size=vocab_size,
            max_len=max_len,
        )


class TransformerBase(_BaseTransformerClassifier):
    """
    Stronger attention baseline.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        dropout: float = 0.1,
        vocab_size: int = 30522,
        max_len: int = 512,
        pad_id: int = 0,
    ):
        # stronger: 4 layers, 8 heads (if d_model supports it)
        nhead = 8 if d_model % 8 == 0 else 4
        super().__init__(
            d_model=d_model,
            num_classes=num_classes,
            nhead=nhead,
            num_layers=4,
            dim_ff=d_model * 4,
            dropout=dropout,
            vocab_size=vocab_size,
            max_len=max_len,
        )
