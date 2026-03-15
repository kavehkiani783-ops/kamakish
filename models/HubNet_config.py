from dataclasses import dataclass


@dataclass
class HUBNETConfig:
    vocab_size: int
    num_classes: int

    max_len: int = 512
    d_model: int = 128
    mem_slots: int = 32
    steps: int = 1
    dropout: float = 0.1
    score_clip: float = 20.0
    topk: int = 0

    # existing v1-style option
    decay: float = 0.9
    gate: bool = False

    # v2 options
    use_residual_update: bool = True
    use_write_gate: bool = True
    use_memory_norm: bool = True
    use_slot_layernorm: bool = True
    init_scale: float = 0.02
