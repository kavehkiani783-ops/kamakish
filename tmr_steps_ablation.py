import itertools
import json
import os
import subprocess
import pandas as pd

# -------------------------------------------------------
# Experiment configuration
# -------------------------------------------------------

DATASETS = ["imdb", "listops_synth"]
SEEDS = [42, 123, 999]
TMR_STEPS_LIST = [0, 1, 2, 4, 8]

TMR_SLOTS = 64
TMR_DECAY = 0.9
TMR_TOPK = 0
TMR_GATE = False

BASE_OUTPUT_DIR = "ablation_runs/steps_ablation"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Run experiments
# -------------------------------------------------------

print("\nStarting TMR steps ablation\n")

all_rows = []

for dataset, steps, seed in itertools.product(DATASETS, TMR_STEPS_LIST, SEEDS):

    print("------------------------------------------------------------")
    print(f"Running dataset={dataset} | tmr_steps={steps} | seed={seed}")
    print("------------------------------------------------------------")

    run_output_dir = os.path.join(
        BASE_OUTPUT_DIR,
        f"{dataset}_steps{steps}_seed{seed}"
    )
    os.makedirs(run_output_dir, exist_ok=True)

    # -------------------------------------------------------
    # Build command
    # -------------------------------------------------------

    if dataset == "imdb":
        cmd = [
            "python",
            "main.py",
            "--dataset", "imdb",
            "--model", "tmr",
            "--epochs", "3",
            "--val_ratio", "0.1",
            "--seed", str(seed),
            "--tmr_steps", str(steps),
            "--tmr_slots", str(TMR_SLOTS),
            "--tmr_decay", str(TMR_DECAY),
            "--tmr_topk", str(TMR_TOPK),
            "--output_dir", run_output_dir,
        ]
    else:
        cmd = [
            "python",
            "main.py",
            "--dataset", "listops_synth",
            "--model", "tmr",
            "--epochs", "3",
            "--batch_size", "64",
            "--max_len", "512",
            "--seed", str(seed),
            "--tmr_steps", str(steps),
            "--tmr_slots", str(TMR_SLOTS),
            "--tmr_decay", str(TMR_DECAY),
            "--tmr_topk", str(TMR_TOPK),
            "--output_dir", run_output_dir,
        ]

    if TMR_GATE:
        cmd.append("--tmr_gate")

    # -------------------------------------------------------
    # Run experiment
    # -------------------------------------------------------

    subprocess.run(cmd, check=True)

    # -------------------------------------------------------
    # Locate JSON result file automatically
    # -------------------------------------------------------

    json_files = [
        f for f in os.listdir(run_output_dir)
        if f.endswith(".json")
    ]

    if len(json_files) == 0:
        print(f"WARNING: No JSON result found in {run_output_dir}")
        continue

    json_file = os.path.join(run_output_dir, json_files[0])

    print(f"Loading results from: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    row = {
        "dataset": dataset,
        "tmr_steps": steps,
        "tmr_slots": TMR_SLOTS,
        "tmr_decay": TMR_DECAY,
        "tmr_topk": TMR_TOPK,
        "tmr_gate": TMR_GATE,
        "seed": seed,
        "test_acc": results.get("test_acc"),
        "macro_f1": results.get("macro_f1"),
        "weighted_f1": results.get("weighted_f1"),
        "balanced_accuracy": results.get("balanced_accuracy"),
        "auroc": results.get("auroc"),
        "auprc": results.get("auprc"),
        "nll": results.get("nll"),
        "brier": results.get("brier"),
        "ece": results.get("ece"),
        "mean_epoch_time": results.get("mean_epoch_time"),
        "wall_time": results.get("wall_time"),
    }

    all_rows.append(row)

# -------------------------------------------------------
# Create full raw results table
# -------------------------------------------------------

if len(all_rows) == 0:
    print("\nNo runs were collected. Something is wrong with result files.")
    raise SystemExit(1)

all_runs_df = pd.DataFrame(all_rows)

all_runs_path = os.path.join(BASE_OUTPUT_DIR, "steps_all_runs_comparison.csv")
all_runs_df.to_csv(all_runs_path, index=False)

print("\nSaved full run table:")
print(all_runs_path)

# -------------------------------------------------------
# Create summary table
# -------------------------------------------------------

summary_df = (
    all_runs_df
    .groupby(["dataset", "tmr_steps"], as_index=False)
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
        mean_epoch_time=("mean_epoch_time", "mean"),
        mean_wall_time=("wall_time", "mean"),
    )
    .sort_values(["dataset", "tmr_steps"])
)

summary_path = os.path.join(BASE_OUTPUT_DIR, "steps_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\nSaved summary table:")
print(summary_path)

# -------------------------------------------------------
# Print summary nicely
# -------------------------------------------------------

for dataset in summary_df["dataset"].unique():

    print(f"\nDataset: {dataset}")
    print("-" * 120)

    print(
        summary_df[summary_df["dataset"] == dataset]
        .to_string(index=False)
    )

print("\nTMR steps ablation finished successfully.")
