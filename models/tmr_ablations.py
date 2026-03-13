# tmr_ablations.py

import copy
from typing import List, Dict, Any

from tmr_config import TMRConfig


def base_config(vocab_size: int, max_len: int = 512) -> TMRConfig:
    return TMRConfig(
        vocab_size=vocab_size,
        max_len=max_len,
        mem_slots=64,
        steps=4,
        decay=0.9,
        gate=False,
        topk=0,
        dropout=0.1,
        score_clip=20.0,
    )


ABLATION_AXES: Dict[str, List[Any]] = {
    "mem_slots": [32, 64, 128, 256],
    "steps": [0, 1, 2, 4, 8],
    "decay": [0.7, 0.85, 0.95],
    "gate": [True, False],
    "topk": [0, 1, 2, 4],   # 0 = softmax, >0 = top-k
}


def set_attr(cfg: TMRConfig, key: str, value: Any):
    setattr(cfg, key, value)


def sweep_axis(axis_name: str, vocab_size: int, max_len: int = 512) -> List[TMRConfig]:
    if axis_name not in ABLATION_AXES:
        raise ValueError(f"Unknown ablation axis: {axis_name}")

    configs = []
    for value in ABLATION_AXES[axis_name]:
        cfg = copy.deepcopy(base_config(vocab_size=vocab_size, max_len=max_len))
        set_attr(cfg, axis_name, value)
        configs.append(cfg)

    return configs


def ablation_set_steps(vocab_size: int, max_len: int = 512) -> List[TMRConfig]:
    return sweep_axis("steps", vocab_size=vocab_size, max_len=max_len)


def ablation_set_slots(vocab_size: int, max_len: int = 512) -> List[TMRConfig]:
    return sweep_axis("mem_slots", vocab_size=vocab_size, max_len=max_len)


def ablation_set_decay(vocab_size: int, max_len: int = 512) -> List[TMRConfig]:
    return sweep_axis("decay", vocab_size=vocab_size, max_len=max_len)


def ablation_set_gate(vocab_size: int, max_len: int = 512) -> List[TMRConfig]:
    return sweep_axis("gate", vocab_size=vocab_size, max_len=max_len)


def ablation_set_binding(vocab_size: int, max_len: int = 512) -> List[TMRConfig]:
    return sweep_axis("topk", vocab_size=vocab_size, max_len=max_len)
