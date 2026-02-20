```python
# tmr_ablations.py

import itertools
import copy
from typing import List, Dict, Any

from tmr_config import TMRConfig


# ------------------------------------------------------------------
# Base Config
# ------------------------------------------------------------------

def base_config() -> TMRConfig:
    cfg = TMRConfig()
    cfg.validate()
    return cfg


# ------------------------------------------------------------------
# Ablation Axes
# ------------------------------------------------------------------

ABLATION_AXES: Dict[str, List[Any]] = {
    "slots": [32, 64, 128, 256],
    "steps": [0, 1, 2, 4, 8],
    "decay": [0.7, 0.85, 0.95],
    "gate.enabled": [True, False],
    "binding.type": ["softmax", "topk"],
    "binding.topk": [1, 2, 4],
    "readout.type": ["query", "mean_pool"],
    "sharing.mem_update": ["shared", "per_step"],
}


# ------------------------------------------------------------------
# Utility: set nested attribute
# ------------------------------------------------------------------

def set_nested_attr(cfg: TMRConfig, key: str, value: Any):
    parts = key.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


# ------------------------------------------------------------------
# Generate single-axis sweep
# ------------------------------------------------------------------

def sweep_axis(axis_name: str) -> List[TMRConfig]:
    if axis_name not in ABLATION_AXES:
        raise ValueError(f"Unknown ablation axis: {axis_name}")

    configs = []
    for value in ABLATION_AXES[axis_name]:
        cfg = copy.deepcopy(base_config())
        set_nested_attr(cfg, axis_name, value)

        if axis_name != "binding.type" and cfg.binding.type == "softmax":
            cfg.binding.topk = 0

        cfg.validate()
        configs.append(cfg)

    return configs


# ------------------------------------------------------------------
# Generate grid from selected axes
# ------------------------------------------------------------------

def sweep_grid(selected_axes: List[str]) -> List[TMRConfig]:
    for axis in selected_axes:
        if axis not in ABLATION_AXES:
            raise ValueError(f"Unknown ablation axis: {axis}")

    values_product = list(
        itertools.product(*[ABLATION_AXES[a] for a in selected_axes])
    )

    configs = []

    for combo in values_product:
        cfg = copy.deepcopy(base_config())

        for axis_name, value in zip(selected_axes, combo):
            set_nested_attr(cfg, axis_name, value)

        if cfg.binding.type == "softmax":
            cfg.binding.topk = 0

        cfg.validate()
        configs.append(cfg)

    return configs


# ------------------------------------------------------------------
# Preset Ablation Groups
# ------------------------------------------------------------------

def ablation_set_core() -> List[TMRConfig]:
    axes = ["steps", "slots", "decay"]
    return sweep_grid(axes)


def ablation_set_binding() -> List[TMRConfig]:
    axes = ["binding.type", "binding.topk"]
    return sweep_grid(axes)


def ablation_set_gate() -> List[TMRConfig]:
    return sweep_axis("gate.enabled")


def ablation_set_readout() -> List[TMRConfig]:
    return sweep_axis("readout.type")


def ablation_set_sharing() -> List[TMRConfig]:
    return sweep_axis("sharing.mem_update")
```
