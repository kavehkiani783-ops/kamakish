from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV = Path("results") / "summary_final_models.csv"
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
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.8,
        "savefig.dpi": 300,
    })


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"dataset", "model", "seed", "test_accuracy", "epoch_time_min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

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


def plot_accuracy_bar(agg: pd.DataFrame) -> None:
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
        ax.set_title(DATASET_LABELS[dataset])
        _style_axis(ax)

        ymin = max(0.0, sub["test_accuracy_mean"].min() - 0.05)
        ymax = min(1.0, sub["test_accuracy_mean"].max() + 0.03)
        ax.set_ylim(ymin, ymax)

        for i, v in enumerate(sub["test_accuracy_mean"]):
            ax.text(i, v + 0.0025, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Performance Across Datasets")
    fig.savefig(OUTPUT_DIR / "journal_accuracy_bar.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "journal_accuracy_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_time_bar(agg: pd.DataFrame) -> None:
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
        ax.set_title(DATASET_LABELS[dataset])
        _style_axis(ax)

        ymax = sub["epoch_time_mean"].max() * 1.15
        ax.set_ylim(0, ymax)

        for i, v in enumerate(sub["epoch_time_mean"]):
            ax.text(i, v + ymax * 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Efficiency Across Datasets")
    fig.savefig(OUTPUT_DIR / "journal_time_bar.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "journal_time_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_scatter(agg: pd.DataFrame) -> None:
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
        ax.set_title(DATASET_LABELS[dataset])
        _style_axis(ax)

        xmax = sub["epoch_time_mean"].max() * 1.15
        ymin = max(0.0, sub["test_accuracy_mean"].min() - 0.05)
        ymax = min(1.0, sub["test_accuracy_mean"].max() + 0.03)
        ax.set_xlim(0, xmax)
        ax.set_ylim(ymin, ymax)

    fig.suptitle("Accuracy–Efficiency Trade-off")
    fig.savefig(OUTPUT_DIR / "journal_tradeoff_scatter.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "journal_tradeoff_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_hubnet_vs_transformer(agg: pd.DataFrame) -> None:
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
    ax.set_title("Accuracy")
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
    ax.set_title("Efficiency")
    _style_axis(ax)

    fig.suptitle("HubNet-v2 vs Transformer-Base")
    fig.savefig(OUTPUT_DIR / "journal_hubnet_vs_transformer.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "journal_hubnet_vs_transformer.pdf", bbox_inches="tight")
    plt.close(fig)


def save_paper_table_csv(agg: pd.DataFrame) -> None:
    rows = []
    for dataset in ["imdb", "listops_synth"]:
        sub = agg[agg["dataset"] == dataset].sort_values("model_order").copy()

        transformer_time = float(
            sub.loc[sub["model"] == "transformer_base", "epoch_time_mean"].iloc[0]
        )

        for _, row in sub.iterrows():
            speedup = transformer_time / row["epoch_time_mean"]
            rows.append({
                "dataset": DATASET_LABELS[dataset],
                "model": row["model_label"],
                "accuracy_mean": round(row["test_accuracy_mean"], 4),
                "accuracy_std": round(row["test_accuracy_std"], 4),
                "epoch_time_mean_s": round(row["epoch_time_mean"], 4),
                "epoch_time_std_s": round(row["epoch_time_std"], 4),
                "relative_speed_vs_transformer": round(speedup, 2),
            })

    out_path = OUTPUT_DIR / "paper_main_results_table.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main() -> None:
    ensure_output_dir()
    set_journal_style()
    agg = load_data(RESULTS_CSV)

    save_paper_table_csv(agg)
    plot_accuracy_bar(agg)
    plot_time_bar(agg)
    plot_tradeoff_scatter(agg)
    plot_hubnet_vs_transformer(agg)

    print("\nSaved all journal-style figures to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
