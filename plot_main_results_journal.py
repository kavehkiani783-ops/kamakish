from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV = Path("results") / "summary_final_models.csv"
RESULTS_JSON_DIR = Path("results")
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


def load_summary_data(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"dataset", "model", "seed", "test_accuracy", "epoch_time_min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary file: {sorted(missing)}")

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

    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS)
    df["model_order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df = df.sort_values(["dataset", "model_order", "seed"]).reset_index(drop=True)

    return df, agg


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: plt.Figure, filename_stem: str) -> None:
    png = OUTPUT_DIR / f"{filename_stem}.png"
    pdf = OUTPUT_DIR / f"{filename_stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


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
        _style_axis(ax)

        ymin = max(0.0, sub["test_accuracy_mean"].min() - 0.05)
        ymax = min(1.0, sub["test_accuracy_mean"].max() + 0.03)
        ax.set_ylim(ymin, ymax)

        for i, v in enumerate(sub["test_accuracy_mean"]):
            ax.text(i, v + 0.0025, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    _save(fig, "Model_Performance_Across_Datasets")


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
        _style_axis(ax)

        ymax = sub["epoch_time_mean"].max() * 1.15
        ax.set_ylim(0, ymax)

        for i, v in enumerate(sub["epoch_time_mean"]):
            ax.text(i, v + ymax * 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    _save(fig, "Model_Efficiency_Across_Datasets")


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

        xmax = sub["epoch_time_mean"].max() * 1.15
        ymin = max(0.0, sub["test_accuracy_mean"].min() - 0.05)
        ymax = min(1.0, sub["test_accuracy_mean"].max() + 0.03)
        ax.set_xlim(0, xmax)
        ax.set_ylim(ymin, ymax)

    _save(fig, "Accuracy_Efficiency_Tradeoff")


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

    _save(fig, "HubNet_v2_vs_Transformer_Base")


def plot_final_accuracy_stability_boxplot(summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    for ax, dataset in zip(axes, ["imdb", "listops_synth"]):
        sub = summary_df[summary_df["dataset"] == dataset].sort_values("model_order")

        data = []
        labels = []
        for model in MODEL_ORDER:
            vals = sub[sub["model"] == model]["test_accuracy"].tolist()
            if vals:
                data.append(vals)
                labels.append(MODEL_LABELS[model])

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.6,
            showfliers=True,
            medianprops={"linewidth": 1.5, "color": "black"},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )

        for patch in bp["boxes"]:
            patch.set_alpha(0.7)

        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel("Test Accuracy Across Seeds")
        _style_axis(ax)

    _save(fig, "Final_Accuracy_Stability_Boxplot")


def parse_filename_for_meta(name: str) -> Optional[dict]:
    stem = name[:-5] if name.endswith(".json") else name
    model = None

    valid_models = [
        "transformer_base",
        "tiny_transformer",
        "meanpool",
        "bilstm",
        "hubnet_v1",
        "hubnet_v2",
    ]

    for candidate in valid_models:
        prefix = candidate + "_"
        if stem.startswith(prefix):
            model = candidate
            rest = stem[len(prefix):]
            break
    else:
        return None

    import re
    m = re.match(r"(.+?)_seed(\d+)(.*)", rest)
    if not m:
        return None

    dataset = m.group(1)
    seed = int(m.group(2))
    return {"dataset": dataset, "model": model, "seed": seed, "file": name}


def _find_epoch_records(obj: Any) -> List[dict]:
    found: List[dict] = []

    if isinstance(obj, dict):
        # common case: list under "history", "epochs", etc.
        for key in ["history", "epoch_history", "epochs", "per_epoch", "train_history"]:
            if key in obj and isinstance(obj[key], list):
                for item in obj[key]:
                    if isinstance(item, dict):
                        found.append(item)

        for v in obj.values():
            found.extend(_find_epoch_records(v))

    elif isinstance(obj, list):
        # direct list of dicts
        if obj and all(isinstance(x, dict) for x in obj):
            for item in obj:
                found.append(item)
        for item in obj:
            found.extend(_find_epoch_records(item))

    return found


def _extract_metric_from_epoch_record(record: dict) -> Optional[float]:
    candidate_keys = [
        "val_accuracy",
        "val_acc",
        "best_val_acc",
        "best_val_accuracy",
        "accuracy",
    ]
    for key in candidate_keys:
        val = record.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _extract_epoch_index(record: dict, default_idx: int) -> int:
    for key in ["epoch", "epoch_idx", "epoch_index"]:
        val = record.get(key)
        if isinstance(val, int):
            return int(val)
    return default_idx


def build_epoch_accuracy_dataframe(results_dir: Path) -> pd.DataFrame:
    rows = []

    for path in sorted(results_dir.glob("*.json")):
        meta = parse_filename_for_meta(path.name)
        if meta is None:
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = _find_epoch_records(data)
        if not records:
            continue

        seen = set()
        for i, record in enumerate(records, start=1):
            epoch_idx = _extract_epoch_index(record, i)
            value = _extract_metric_from_epoch_record(record)
            if value is None:
                continue

            key = (meta["dataset"], meta["model"], meta["seed"], epoch_idx)
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "dataset": meta["dataset"],
                "model": meta["model"],
                "seed": meta["seed"],
                "epoch": epoch_idx,
                "val_accuracy": value,
            })

    if not rows:
        return pd.DataFrame(columns=["dataset", "model", "seed", "epoch", "val_accuracy"])

    df = pd.DataFrame(rows)
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    return df.sort_values(["dataset", "model", "seed", "epoch"]).reset_index(drop=True)


def plot_epoch_accuracy_stability_boxplot(epoch_df: pd.DataFrame) -> None:
    if epoch_df.empty:
        print("No epoch-level accuracy records found in results JSON files. Skipping epoch stability boxplot.")
        return

    # one figure per dataset to avoid clutter
    for dataset in ["imdb", "listops_synth"]:
        sub_ds = epoch_df[epoch_df["dataset"] == dataset].copy()
        if sub_ds.empty:
            continue

        # label format: Model E1, Model E2, ...
        labels = []
        data = []

        for model in MODEL_ORDER:
            sub_model = sub_ds[sub_ds["model"] == model]
            if sub_model.empty:
                continue

            epochs = sorted(sub_model["epoch"].unique())
            for ep in epochs:
                vals = sub_model[sub_model["epoch"] == ep]["val_accuracy"].tolist()
                if vals:
                    labels.append(f"{MODEL_LABELS[model]}\nE{ep}")
                    data.append(vals)

        if not data:
            continue

        fig, ax = plt.subplots(figsize=(max(13, len(labels) * 0.6), 5.2), constrained_layout=True)

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.55,
            showfliers=True,
            medianprops={"linewidth": 1.5, "color": "black"},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )

        for patch in bp["boxes"]:
            patch.set_alpha(0.7)

        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Validation Accuracy Across Seeds")
        _style_axis(ax)

        filename = f"Epoch_Accuracy_Stability_Boxplot_{DATASET_LABELS[dataset]}"
        _save(fig, filename.replace(" ", "_"))


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

    out_path = OUTPUT_DIR / "Paper_Main_Results_Table.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main() -> None:
    ensure_output_dir()
    set_journal_style()

    summary_df, agg = load_summary_data(RESULTS_CSV)
    save_paper_table_csv(agg)

    plot_model_performance_across_datasets(agg)
    plot_model_efficiency_across_datasets(agg)
    plot_accuracy_efficiency_tradeoff(agg)
    plot_hubnet_v2_vs_transformer_base(agg)
    plot_final_accuracy_stability_boxplot(summary_df)

    epoch_df = build_epoch_accuracy_dataframe(RESULTS_JSON_DIR)
    plot_epoch_accuracy_stability_boxplot(epoch_df)

    print(f"\nSaved all figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
