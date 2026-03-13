import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPool(nn.Module):
    """
    Simple baseline:
      input_ids (B,T) -> token embeddings (B,T,d) -> masked mean pool -> classifier
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        vocab_size: int = 30522,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        input_ids: (B,T)
        attention_mask: (B,T) with 1 for valid tokens, 0 for padding
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)  # (B,T,d)
        x = self.dropout(x)

        if attention_mask is None:
            pooled = x.mean(dim=1)  # (B,d)
        else:
            m = attention_mask.to(x.dtype).unsqueeze(-1)  # (B,T,1)
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # (B,d)

        pooled = self.norm(pooled)
        logits = self.classifier(pooled)
        return logits


class BiLSTM(nn.Module):
    """
    BiLSTM baseline:
      input_ids (B,T) -> embeddings -> packed BiLSTM -> pooled hidden -> classifier
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        vocab_size: int = 30522,
        max_len: int = 512,
        hidden_size: int = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        if hidden_size is None:
            hidden_size = d_model // 2  # so biLSTM outputs ~d_model

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        input_ids: (B,T)
        attention_mask: (B,T) with 1 for valid tokens, 0 for padding
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)  # (B,T,d)
        x = self.dropout(x)

        if attention_mask is None:
            # Assume all tokens valid
            out, _ = self.lstm(x)  # (B,T,2h)
            pooled = out.mean(dim=1)
        else:
            # lengths: (B,)
            lengths = attention_mask.sum(dim=1).clamp_min(1).to(torch.int64).cpu()

            # pack -> lstm -> unpack
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )  # (B,T,2h)

            m = attention_mask.to(out.dtype).unsqueeze(-1)  # (B,T,1)
            pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        pooled = self.norm(pooled)
        logits = self.classifier(pooled)
        return logits
