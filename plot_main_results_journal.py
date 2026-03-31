from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import re

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV = Path("results") / "summary_final_models.csv"
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("figures_journal")

MODEL_ORDER: List[str] = [
    "meanpool",
    "bilstm",
    "tiny_transformer",
    "transformer_base",
    "hubnet_v1",
    "hubnet_v2",
]

MODEL_LABELS: Dict[str, str] = {
    "meanpool": "MeanPool",
    "bilstm": "BiLSTM",
    "tiny_transformer": "TinyTransformer",
    "transformer_base": "Transformer-Base",
    "hubnet_v1": "HubNet-v1",
    "hubnet_v2": "HubNet-v2",
}

DATASET_LABELS: Dict[str, str] = {
    "imdb": "IMDB",
    "listops_synth": "ListOps",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_journal_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.8,
        "savefig.dpi": 300,
    })


def load_summary_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"dataset", "model", "seed", "test_accuracy", "epoch_time_min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary CSV: {sorted(missing)}")

    df = df[df["model"].isin(MODEL_ORDER)].copy()

    agg = (
        df.groupby(["dataset", "model"], as_index=False)
        .agg(
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            epoch_time_mean=("epoch_time_min", "mean"),
            epoch_time_std=("epoch_time_min", "std"),
            n_runs=("seed", "count"),
        )
    )

    agg["test_accuracy_std"] = agg["test_accuracy_std"].fillna(0.0)
    agg["epoch_time_std"] = agg["epoch_time_std"].fillna(0.0)
    agg["model_label"] = agg["model"].map(MODEL_LABELS)
    agg["dataset_label"] = agg["dataset"].map(DATASET_LABELS)
    agg["model_order"] = agg["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    agg = agg.sort_values(["dataset", "model_order"]).reset_index(drop=True)

    return agg


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_model_performance_across_datasets(agg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    for ax, dataset in zip(axes, ["imdb", "listops_synth"]):
        sub = agg[agg["dataset"] == dataset].sort_values("model_order")
        x = list(range(len(sub)))

        ax.bar(
            x,
            sub["test_accuracy_mean"],
            yerr=sub["test_accuracy_std"],
            capsize=3,
            width=0.72,
            edgecolor="black",
            linewidth=0.7,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(sub["model_label"], rotation=28, ha="right")
        ax.set_ylabel("Test Accuracy")
        ax.set_xlabel(DATASET_LABELS[dataset])
        _style_axis(ax)

        ymin = max(0.0, sub["test_accuracy_mean"].min() - 0.05)
        ymax = min(1.0, sub["test_accuracy_mean"].max() + 0.03)
        ax.set_ylim(ymin, ymax)

        for i, v in enumerate(sub["test_accuracy_mean"]):
            ax.text(i, v + 0.0025, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    out_base = OUTPUT_DIR / "Model Performance Across Datasets"
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_model_efficiency_across_datasets(agg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    for ax, dataset in zip(axes, ["imdb", "listops_synth"]):
        sub = agg[agg["dataset"] == dataset].sort_values("model_order")
        x = list(range(len(sub)))

        ax.bar(
            x,
            sub["epoch_time_mean"],
            yerr=sub["epoch_time_std"],
            capsize=3,
            width=0.72,
            edgecolor="black",
            linewidth=0.7,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(sub["model_label"], rotation=28, ha="right")
        ax.set_ylabel("Training Time per Epoch (s)")
        ax.set_xlabel(DATASET_LABELS[dataset])
        _style_axis(ax)

        ymax = sub["epoch_time_mean"].max() * 1.15
        ax.set_ylim(0, ymax)

        for i, v in enumerate(sub["epoch_time_mean"]):
            ax.text(i, v + ymax * 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    out_base = OUTPUT_DIR / "Model Efficiency Across Datasets"
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_efficiency_tradeoff(agg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    for ax, dataset in zip(axes, ["imdb", "listops_synth"]):
        sub = agg[agg["dataset"] == dataset].sort_values("model_order")

        ax.scatter(
            sub["epoch_time_mean"],
            sub["test_accuracy_mean"],
            s=85,
            edgecolors="black",
            linewidths=0.7,
        )

        for _, row in sub.iterrows():
            ax.annotate(
                row["model_label"],
                (row["epoch_time_mean"], row["test_accuracy_mean"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
            )

        ax.set_xlabel("Training Time per Epoch (s)")
        ax.set_ylabel("Test Accuracy")
        _style_axis(ax)

        ax.text(
            0.5, 1.02, DATASET_LABELS[dataset],
            transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11
        )

        xmax = sub["epoch_time_mean"].max() * 1.15
        ymin = max(0.0, sub["test_accuracy_mean"].min() - 0.05)
        ymax = min(1.0, sub["test_accuracy_mean"].max() + 0.03)
        ax.set_xlim(0, xmax)
        ax.set_ylim(ymin, ymax)

    out_base = OUTPUT_DIR / "Accuracy Efficiency Trade-off"
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_hubnet_v2_vs_transformer_base(agg: pd.DataFrame) -> None:
    focus = ["transformer_base", "hubnet_v2"]
    datasets = ["imdb", "listops_synth"]
    width = 0.34
    x = list(range(len(datasets)))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    acc_t, acc_h, acc_t_std, acc_h_std = [], [], [], []
    time_t, time_h, time_t_std, time_h_std = [], [], [], []

    for ds in datasets:
        sub = agg[(agg["dataset"] == ds) & (agg["model"].isin(focus))].set_index("model")

        acc_t.append(sub.loc["transformer_base", "test_accuracy_mean"])
        acc_h.append(sub.loc["hubnet_v2", "test_accuracy_mean"])
        acc_t_std.append(sub.loc["transformer_base", "test_accuracy_std"])
        acc_h_std.append(sub.loc["hubnet_v2", "test_accuracy_std"])

        time_t.append(sub.loc["transformer_base", "epoch_time_mean"])
        time_h.append(sub.loc["hubnet_v2", "epoch_time_mean"])
        time_t_std.append(sub.loc["transformer_base", "epoch_time_std"])
        time_h_std.append(sub.loc["hubnet_v2", "epoch_time_std"])

    ax = axes[0]
    ax.bar(
        [i - width / 2 for i in x],
        acc_t,
        width=width,
        yerr=acc_t_std,
        capsize=3,
        edgecolor="black",
        linewidth=0.7,
        label="Transformer-Base",
    )
    ax.bar(
        [i + width / 2 for i in x],
        acc_h,
        width=width,
        yerr=acc_h_std,
        capsize=3,
        edgecolor="black",
        linewidth=0.7,
        label="HubNet-v2",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel("Test Accuracy")
    _style_axis(ax)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(
        [i - width / 2 for i in x],
        time_t,
        width=width,
        yerr=time_t_std,
        capsize=3,
        edgecolor="black",
        linewidth=0.7,
        label="Transformer-Base",
    )
    ax.bar(
        [i + width / 2 for i in x],
        time_h,
        width=width,
        yerr=time_h_std,
        capsize=3,
        edgecolor="black",
        linewidth=0.7,
        label="HubNet-v2",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel("Training Time per Epoch (s)")
    _style_axis(ax)

    out_base = OUTPUT_DIR / "HubNet-v2 vs Transformer-Base"
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _extract_seed_from_filename(name: str) -> Optional[int]:
    m = re.search(r"_seed(\d+)", name)
    return int(m.group(1)) if m else None


def _find_first_list_of_dicts_with_epoch_key(obj: Any) -> Optional[List[dict]]:
    """
    Try to find a history structure like:
    [{"epoch": 1, "val_accuracy": ...}, ...]
    """
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            key_union = set()
            for item in obj:
                key_union.update(item.keys())
            if "epoch" in key_union and (
                "val_accuracy" in key_union or
                "val_acc" in key_union or
                "accuracy" in key_union
            ):
                return obj
        for item in obj:
            found = _find_first_list_of_dicts_with_epoch_key(item)
            if found is not None:
                return found

    if isinstance(obj, dict):
        for value in obj.values():
            found = _find_first_list_of_dicts_with_epoch_key(value)
            if found is not None:
                return found

    return None


def _extract_epoch_accuracy_history(json_obj: dict) -> Optional[List[dict]]:
    """
    Returns a list like:
    [{"epoch": 1, "accuracy": 0.61}, ...]
    if found, otherwise None.
    """
    history = _find_first_list_of_dicts_with_epoch_key(json_obj)
    if history is None:
        return None

    cleaned = []
    for row in history:
        epoch = row.get("epoch")
        acc = None
        for k in ["val_accuracy", "val_acc", "accuracy"]:
            if k in row and isinstance(row[k], (int, float)):
                acc = float(row[k])
                break
        if epoch is not None and acc is not None:
            cleaned.append({"epoch": int(epoch), "accuracy": acc})

    return cleaned if cleaned else None


def build_epoch_accuracy_dataframe() -> pd.DataFrame:
    """
    Reads raw JSON result files and tries to build a per-epoch validation-accuracy table.
    """
    rows = []

    if not RESULTS_DIR.exists():
        return pd.DataFrame()

    for path in RESULTS_DIR.glob("*.json"):
        if path.name == "summary_final_models.csv":
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        history = _extract_epoch_accuracy_history(data)
        if history is None:
            continue

        parts = path.stem
        model = None
        dataset = None

        for candidate in MODEL_ORDER:
            prefix = candidate + "_"
            if parts.startswith(prefix):
                model = candidate
                rest = parts[len(prefix):]
                m = re.match(r"(.+?)_seed(\d+)", rest)
                if m:
                    dataset = m.group(1)
                break

        seed = _extract_seed_from_filename(path.name)

        if model is None or dataset is None or seed is None:
            continue

        for item in history:
            rows.append({
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "epoch": item["epoch"],
                "accuracy": item["accuracy"],
            })

    return pd.DataFrame(rows)


def plot_epoch_accuracy_stability_boxplot() -> None:
    epoch_df = build_epoch_accuracy_dataframe()

    if epoch_df.empty:
        print("No per-epoch accuracy history found in raw JSON files. Skipping boxplot.")
        return

    epoch_df = epoch_df[epoch_df["model"].isin(MODEL_ORDER)].copy()

    datasets = sorted(epoch_df["dataset"].unique())
    if not datasets:
        print("No datasets found for epoch boxplot. Skipping.")
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5), constrained_layout=True)

    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub = epoch_df[epoch_df["dataset"] == dataset].copy()
        if sub.empty:
            continue

        epochs = sorted(sub["epoch"].unique())
        models = [m for m in MODEL_ORDER if m in sub["model"].unique()]

        positions = []
        box_data = []
        labels = []
        pos = 1

        for epoch in epochs:
            for model in models:
                vals = sub[(sub["epoch"] == epoch) & (sub["model"] == model)]["accuracy"].tolist()
                if vals:
                    positions.append(pos)
                    box_data.append(vals)
                    labels.append(MODEL_LABELS[model])
                pos += 1
            pos += 1

        ax.boxplot(
            box_data,
            positions=positions,
            widths=0.7,
            patch_artist=False,
            showfliers=True,
        )

        tick_positions = []
        tick_labels = []
        pos = 1
        for epoch in epochs:
            start = pos
            end = pos + len(models) - 1
            tick_positions.append((start + end) / 2)
            tick_labels.append(f"Epoch {epoch}")
            pos = end + 2

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Validation Accuracy Across Seeds")
        ax.set_xlabel(DATASET_LABELS.get(dataset, dataset))
        _style_axis(ax)

        ax.text(
            0.5, 1.02, DATASET_LABELS.get(dataset, dataset),
            transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11
        )

    out_base = OUTPUT_DIR / "Epoch-wise Accuracy Stability Across Seeds"
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_output_dir()
    set_journal_style()
    agg = load_summary_data(RESULTS_CSV)

    plot_model_performance_across_datasets(agg)
    plot_model_efficiency_across_datasets(agg)
    plot_accuracy_efficiency_tradeoff(agg)
    plot_hubnet_v2_vs_transformer_base(agg)
    plot_epoch_accuracy_stability_boxplot()

    print("\nSaved all journal-style figures to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
