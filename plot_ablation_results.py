import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def build_parser():
    parser = argparse.ArgumentParser(description="Create tables and figures for HubNet ablations.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to one ablation run directory")
    return parser


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)

    all_runs_csv = run_dir / "all_runs.csv"
    summary_csv = run_dir / "summary_table.csv"

    if not all_runs_csv.exists():
        raise FileNotFoundError(f"Missing file: {all_runs_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing file: {summary_csv}")

    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    df_all = pd.read_csv(all_runs_csv)
    df_summary = pd.read_csv(summary_csv)

    # Try to standardise column names from summarise_runs output
    lower_map = {c: c.lower() for c in df_summary.columns}
    df_summary = df_summary.rename(columns=lower_map)

    # Expected semantic columns
    # We try a few common names to be robust.
    def pick_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"Could not find any of these columns: {candidates}")

    dataset_col = pick_col(df_summary, ["dataset"])
    ablation_name_col = pick_col(df_summary, ["ablation_name"])
    ablation_value_col = pick_col(df_summary, ["ablation_value"])
    mean_acc_col = pick_col(df_summary, ["mean_test_accuracy", "avg_test_accuracy", "test_accuracy_mean", "mean_accuracy"])
    mean_time_col = pick_col(df_summary, ["mean_epoch_time_min", "avg_epoch_time_min", "epoch_time_min_mean", "mean_epoch_time"])

    # Save cleaner paper table
    paper_table = df_summary[[dataset_col, ablation_name_col, ablation_value_col, mean_acc_col, mean_time_col]].copy()
    paper_table.columns = ["dataset", "ablation", "value", "mean_test_accuracy", "mean_epoch_time"]
    paper_table.to_csv(tables_dir / "paper_ablation_table.csv", index=False)

    # Create one pair of plots per dataset
    datasets = sorted(paper_table["dataset"].unique())
    ablation_name = str(paper_table["ablation"].iloc[0])

    for dataset in datasets:
        d = paper_table[paper_table["dataset"] == dataset].sort_values("value")

        # Accuracy plot
        plt.figure(figsize=(7, 5))
        plt.plot(d["value"], d["mean_test_accuracy"], marker="o")
        plt.xlabel(ablation_name)
        plt.ylabel("Mean test accuracy")
        plt.title(f"{dataset}: accuracy vs {ablation_name}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{dataset}_{ablation_name}_accuracy.png", dpi=300)
        plt.close()

        # Time plot
        plt.figure(figsize=(7, 5))
        plt.plot(d["value"], d["mean_epoch_time"], marker="o")
        plt.xlabel(ablation_name)
        plt.ylabel("Mean epoch time")
        plt.title(f"{dataset}: epoch time vs {ablation_name}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{dataset}_{ablation_name}_time.png", dpi=300)
        plt.close()

        # Trade-off scatter
        plt.figure(figsize=(7, 5))
        plt.scatter(d["mean_epoch_time"], d["mean_test_accuracy"])
        for _, row in d.iterrows():
            plt.annotate(str(row["value"]), (row["mean_epoch_time"], row["mean_test_accuracy"]))
        plt.xlabel("Mean epoch time")
        plt.ylabel("Mean test accuracy")
        plt.title(f"{dataset}: accuracy-efficiency trade-off ({ablation_name})")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{dataset}_{ablation_name}_tradeoff.png", dpi=300)
        plt.close()

    # Combined comparison plot across datasets
    plt.figure(figsize=(8, 5))
    for dataset in datasets:
        d = paper_table[paper_table["dataset"] == dataset].sort_values("value")
        plt.plot(d["value"], d["mean_test_accuracy"], marker="o", label=dataset)
    plt.xlabel(ablation_name)
    plt.ylabel("Mean test accuracy")
    plt.title(f"Accuracy across datasets: {ablation_name} ablation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"combined_{ablation_name}_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    for dataset in datasets:
        d = paper_table[paper_table["dataset"] == dataset].sort_values("value")
        plt.plot(d["value"], d["mean_epoch_time"], marker="o", label=dataset)
    plt.xlabel(ablation_name)
    plt.ylabel("Mean epoch time")
    plt.title(f"Efficiency across datasets: {ablation_name} ablation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"combined_{ablation_name}_time.png", dpi=300)
    plt.close()

    print(f"Saved figures to: {figures_dir}")
    print(f"Saved tables to: {tables_dir}")


if __name__ == "__main__":
    main()
