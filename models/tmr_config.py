from dataclasses import dataclass


@dataclass
class TMRConfig:
    # core
    vocab_size: int
    max_len: int = 512

    # memory / dynamics
    mem_slots: int = 64
    steps: int = 4
    decay: float = 0.9
    gate: bool = False

    # binding
    topk: int = 0   # 0 means softmax binding

    # regularisation / stability
    dropout: float = 0.1
    score_clip: float = 20.0
