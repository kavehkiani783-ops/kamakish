# analysis/plots.py

import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_many(paths: List[str]) -> List[Dict[str, Any]]:
    return [load_json(p) for p in paths]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------
# Table printing (console-friendly)
# ------------------------------------------------------------

def print_main_table(results: List[Dict[str, Any]]):
    header = (
        "Model          | TestAcc | MacroF1 | BalAcc | ECE | NLL | Brier | "
        "Ent | Conf | Params(M) | ms/b | PeakMB | Time(min)"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for r in results:
        name = r["model_name"]
        m = r["metrics"]
        eff = r["efficiency"]

        print(
            f"{name:14s} | "
            f"{m.get('accuracy', 0):.4f} | "
            f"{m.get('macro_f1', 0):.4f} | "
            f"{m.get('balanced_accuracy', 0):.4f} | "
            f"{m.get('ece', 0):.4f} | "
            f"{m.get('nll', 0):.4f} | "
            f"{m.get('brier', 0):.4f} | "
            f"{m.get('entropy', 0):.4f} | "
            f"{m.get('confidence', 0):.4f} | "
            f"{eff.get('params_million', 0):8.2f} | "
            f"{eff.get('ms_per_batch', 0):6.2f} | "
            f"{eff.get('peak_memory_mb', 0):7.1f} | "
            f"{eff.get('total_time_min', 0):8.2f}"
        )

    print("=" * len(header))


# ------------------------------------------------------------
# Figure 1: Scaling with sequence length
# ------------------------------------------------------------

def plot_length_scaling(
    results: List[Dict[str, Any]],
    metric_key: str = "accuracy",
    save_path: Optional[str] = None
):
    """
    results must contain:
        - model_name
        - sequence_length
        - metrics[metric_key]
        - efficiency[ms_per_batch]
        - efficiency[peak_memory_mb]
    """

    models = sorted(set(r["model_name"] for r in results))

    plt.figure(figsize=(14, 4))

    # Accuracy subplot
    plt.subplot(1, 3, 1)
    for model in models:
        xs = []
        ys = []
        for r in results:
            if r["model_name"] == model:
                xs.append(r["sequence_length"])
                ys.append(r["metrics"][metric_key])
        order = np.argsort(xs)
        plt.plot(np.array(xs)[order], np.array(ys)[order], marker="o", label=model)
    plt.xlabel("Sequence length")
    plt.ylabel(metric_key)
    plt.title("Performance scaling")
    plt.legend()

    # Latency subplot
    plt.subplot(1, 3, 2)
    for model in models:
        xs = []
        ys = []
        for r in results:
            if r["model_name"] == model:
                xs.append(r["sequence_length"])
                ys.append(r["efficiency"]["ms_per_batch"])
        order = np.argsort(xs)
        plt.plot(np.array(xs)[order], np.array(ys)[order], marker="o")
    plt.xlabel("Sequence length")
    plt.ylabel("ms / batch")
    plt.title("Latency scaling")

    # Memory subplot
    plt.subplot(1, 3, 3)
    for model in models:
        xs = []
        ys = []
        for r in results:
            if r["model_name"] == model:
                xs.append(r["sequence_length"])
                ys.append(r["efficiency"]["peak_memory_mb"])
        order = np.argsort(xs)
        plt.plot(np.array(xs)[order], np.array(ys)[order], marker="o")
    plt.xlabel("Sequence length")
    plt.ylabel("Peak memory (MB)")
    plt.title("Memory scaling")

    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
    plt.show()


# ------------------------------------------------------------
# Figure 2: Efficiency frontier (Pareto)
# ------------------------------------------------------------

def plot_pareto_frontier(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None
):
    """
    x-axis: compute (ms/batch)
    y-axis: accuracy
    """

    plt.figure(figsize=(6, 5))

    for r in results:
        x = r["efficiency"]["ms_per_batch"]
        y = r["metrics"]["accuracy"]
        label = r["model_name"]
        plt.scatter(x, y)
        plt.text(x, y, label, fontsize=8)

    plt.xlabel("ms / batch")
    plt.ylabel("Accuracy")
    plt.title("Efficiency Frontier")
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
    plt.show()


# ------------------------------------------------------------
# Figure 3: Calibration comparison
# ------------------------------------------------------------

def plot_calibration_scatter(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None
):
    """
    x-axis: ECE
    y-axis: Accuracy
    """

    plt.figure(figsize=(6, 5))

    for r in results:
        ece = r["metrics"].get("ece", 0)
        acc = r["metrics"].get("accuracy", 0)
        label = r["model_name"]
        plt.scatter(ece, acc)
        plt.text(ece, acc, label, fontsize=8)

    plt.xlabel("ECE")
    plt.ylabel("Accuracy")
    plt.title("Calibration vs Performance")
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
    plt.show()


# ------------------------------------------------------------
# Figure 4: Resonance dynamics
# ------------------------------------------------------------

def plot_resonance_dynamics(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None
):
    """
    Requires:
        - model_name == "TMR" (or contains resonance)
        - dynamics["delta_norms"]
    """

    plt.figure(figsize=(6, 5))

    for r in results:
        if "delta_norms" in r.get("dynamics", {}):
            deltas = r["dynamics"]["delta_norms"]
            steps = list(range(1, len(deltas) + 1))
            plt.plot(steps, deltas, marker="o", label=r["model_name"])

    plt.xlabel("Resonance step")
    plt.ylabel("Delta norm")
    plt.title("Resonance Convergence Dynamics")
    plt.legend()
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
    plt.show()


# ------------------------------------------------------------
# Reliability Diagram
# ------------------------------------------------------------

def plot_reliability_diagram(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
    save_path: Optional[str] = None
):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins) - 1

    bin_acc = []
    bin_conf = []

    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() > 0:
            bin_acc.append(correctness[mask].mean())
            bin_conf.append(confidences[mask].mean())
        else:
            bin_acc.append(0)
            bin_conf.append(0)

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.bar(bin_conf, bin_acc, width=1/n_bins, alpha=0.6)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
    plt.show()
