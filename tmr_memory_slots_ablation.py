import subprocess
import itertools
import os
import json
import pandas as pd

# -------------------------------------------------------
# Experiment configuration
# -------------------------------------------------------

DATASETS = ["imdb", "listops_synth"]
SEEDS = [42, 123, 999]
MEM_SLOTS = [16, 32, 64, 128, 256]

TMR_STEPS = 4

BASE_OUTPUT_DIR = "ablation_runs/memory_slots_ablation"

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Run experiments
# -------------------------------------------------------

print("\nStarting TMR memory slots ablation\n")

all_rows = []

for dataset, slots, seed in itertools.product(DATASETS, MEM_SLOTS, SEEDS):

    print("------------------------------------------------------------")
    print(f"Running dataset={dataset} | mem_slots={slots} | seed={seed}")
    print("------------------------------------------------------------")

    # unique folder per run so files never overwrite
    run_output_dir = os.path.join(
        BASE_OUTPUT_DIR,
        f"{dataset}_slots{slots}_seed{seed}"
    )

    os.makedirs(run_output_dir, exist_ok=True)

    if dataset == "imdb":
        cmd = [
            "python",
            "main.py",
            "--dataset", "imdb",
            "--model", "tmr",
            "--epochs", "3",
            "--val_ratio", "0.1",
            "--seed", str(seed),
            "--tmr_slots", str(slots),
            "--tmr_steps", str(TMR_STEPS),
            "--output_dir", run_output_dir
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
            "--tmr_slots", str(slots),
            "--tmr_steps", str(TMR_STEPS),
            "--output_dir", run_output_dir
        ]

    subprocess.run(cmd, check=True)

    # -------------------------------------------------------
    # Read JSON result produced by main.py
    # -------------------------------------------------------

    json_file = os.path.join(run_output_dir, f"tmr_{dataset}_{seed}.json")

    if not os.path.exists(json_file):
        print(f"WARNING: Missing result file {json_file}")
        continue

    with open(json_file, "r") as f:
        results = json.load(f)

    row = {
        "dataset": dataset,
        "tmr_slots": slots,
        "tmr_steps": TMR_STEPS,
        "seed": seed,
        "test_acc": results.get("test_acc"),
        "mean_epoch_time": results.get("mean_epoch_time"),
        "wall_time": results.get("wall_time")
    }

    all_rows.append(row)

# -------------------------------------------------------
# Create full raw results table
# -------------------------------------------------------

all_runs_df = pd.DataFrame(all_rows)

all_runs_path = os.path.join(BASE_OUTPUT_DIR, "memory_slots_all_runs.csv")

all_runs_df.to_csv(all_runs_path, index=False)

print("\nSaved full run table:")
print(all_runs_path)

# -------------------------------------------------------
# Create summary table (mean/std across seeds)
# -------------------------------------------------------

summary_df = (
    all_runs_df
    .groupby(["dataset", "tmr_slots"], as_index=False)
    .agg(
        n=("test_acc", "count"),
        mean_acc=("test_acc", "mean"),
        std_acc=("test_acc", "std"),
        mean_epoch_time=("mean_epoch_time", "mean"),
        mean_wall_time=("wall_time", "mean")
    )
    .sort_values(["dataset", "tmr_slots"])
)

summary_path = os.path.join(BASE_OUTPUT_DIR, "memory_slots_summary.csv")

summary_df.to_csv(summary_path, index=False)

print("\nSaved summary table:")
print(summary_path)

# -------------------------------------------------------
# Print summary table nicely
# -------------------------------------------------------

for dataset in summary_df["dataset"].unique():

    print(f"\nDataset: {dataset}")
    print("-" * 100)

    print(
        summary_df[summary_df["dataset"] == dataset]
        .to_string(index=False)
    )

print("\nAblation finished successfully.")
