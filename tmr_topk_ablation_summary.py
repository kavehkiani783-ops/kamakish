import os
import json
import pandas as pd

BASE_OUTPUT_DIR = "ablation_runs/topk_ablation"

rows = []

# -------------------------------------------------------
# Parse folder names safely
# Examples:
#   imdb_topk4_seed42
#   listops_synth_topk16_seed999
# -------------------------------------------------------

def parse_run_dir(run_dir):
    parts = run_dir.split("_")

    dataset_parts = []
    topk = None
    seed = None

    for p in parts:
        if p.startswith("topk"):
            topk = int(p.replace("topk", ""))
        elif p.startswith("seed"):
            seed = int(p.replace("seed", ""))
        else:
            dataset_parts.append(p)

    dataset = "_".join(dataset_parts)
    return dataset, topk, seed

# -------------------------------------------------------
# Scan run folders and read metrics
# -------------------------------------------------------

for run_dir in os.listdir(BASE_OUTPUT_DIR):

    run_path = os.path.join(BASE_OUTPUT_DIR, run_dir)

    if not os.path.isdir(run_path):
        continue

    json_files = [f for f in os.listdir(run_path) if f.endswith(".json")]
    if len(json_files) == 0:
        continue

    json_file = os.path.join(run_path, json_files[0])

    with open(json_file, "r") as f:
        results = json.load(f)

    dataset, topk, seed = parse_run_dir(run_dir)

    test_metrics = results["best_metrics"]["test"]

    row = {
        "dataset": dataset,
        "topk": topk,
        "seed": seed,
        "test_acc": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "auroc": test_metrics["auroc"],
        "auprc": test_metrics["auprc"],
        "nll": test_metrics["nll"],
        "brier": test_metrics["brier"],
        "ece": test_metrics["ece"],
        "wall_time": results.get("total_minutes")
    }

    rows.append(row)

# -------------------------------------------------------
# Build full comparison table
# -------------------------------------------------------

df = pd.DataFrame(rows)

if df.empty:
    raise ValueError("No completed JSON runs were found in ablation_runs/topk_ablation")

df = df.sort_values(["dataset", "topk", "seed"]).reset_index(drop=True)

all_runs_path = os.path.join(BASE_OUTPUT_DIR, "topk_all_runs_comparison.csv")
df.to_csv(all_runs_path, index=False)

print("\nSaved full comparison table:")
print(all_runs_path)

print("\nDetected runs:")
print(df.to_string(index=False))

# -------------------------------------------------------
# Build grouped summary table
# -------------------------------------------------------

summary_df = (
    df
    .groupby(["dataset", "topk"], as_index=False)
    .agg(
        n=("test_acc", "count"),
        mean_acc=("test_acc", "mean"),
        std_acc=("test_acc", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_weighted_f1=("weighted_f1", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_auroc=("auroc", "mean"),
        mean_auprc=("auprc", "mean"),
        mean_nll=("nll", "mean"),
        mean_brier=("brier", "mean"),
        mean_ece=("ece", "mean"),
        mean_wall_time=("wall_time", "mean")
    )
    .sort_values(["dataset", "topk"])
    .reset_index(drop=True)
)

summary_path = os.path.join(BASE_OUTPUT_DIR, "topk_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\nSaved summary table:")
print(summary_path)

# -------------------------------------------------------
# Print summary nicely
# -------------------------------------------------------

for dataset in summary_df["dataset"].unique():
    print(f"\nDataset: {dataset}")
    print("-" * 120)
    print(summary_df[summary_df["dataset"] == dataset].to_string(index=False))
