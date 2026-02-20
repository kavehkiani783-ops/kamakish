from dataclasses import dataclass

@dataclass
class TMRConfig:
    # -------------------------
    # Core
    # -------------------------
    d_model: int
    vocab_size: int
    num_classes: int
    pad_id: int

    # -------------------------
    # Memory + dynamics
    # -------------------------
    num_slots: int = 64          # S
    num_steps: int = 4           # K (settling steps)
    decay: float = 0.9           # λ
    use_gate: bool = False       # gating on/off

    # -------------------------
    # Binding / interaction
    # -------------------------
    topk: int = 0                # 0 => softmax binding, k>0 => top-k binding

    # -------------------------
    # Init / stability
    # -------------------------
    memory_init_scale: float = 0.02
