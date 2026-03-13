import subprocess
import itertools
import os
import json
import pandas as pd

datasets = ["imdb", "listops_synth"]
seeds = [42, 123, 999]
mem_slots = [16, 32, 64, 128, 256]
steps = 4

base_output_dir = "ablation_runs/memory_slots_ablation"
os.makedirs(base_output_dir, exist_ok=True)

all_rows = []

for dataset, slots, seed in itertools.product(datasets, mem_slots, seeds):
    print("------------------------------------------------------------")
    print(f"Running dataset={dataset} | mem_slots={slots} | seed={seed}")
    print("------------------------------------------------------------")

    # unique folder per run
    run_output_dir = os.path.join(
        base_output_dir,
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
            "--tmr_steps", str(steps),
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
            "--tmr_steps", str(steps),
            "--output_dir", run_output_dir
        ]

    subprocess.run(cmd, check=True)

    json_path = os.path.join(run_output_dir, f"tmr_{dataset}_{seed}.json")
    if not os.path.exists(json_path):
        print(f"Warning: missing result file {json_path}")
        continue

    with open(json_path, "r") as f:
        result = json.load(f)

    all_rows.append({
        "dataset": dataset,
        "tmr_slots": slots,
        "seed": seed,
        "tmr_steps": steps,
        "test_acc": result.get("test_acc"),
        "mean_epoch_time": result.get("mean_epoch_time"),
        "wall_time": result.get("wall_time"),
    })

# save one raw table with all runs
all_runs_df = pd.DataFrame(all_rows)
all_runs_path = os.path.join(base_output_dir, "memory_slots_all_runs.csv")
all_runs_df.to_csv(all_runs_path, index=False)

# save one summary table
summary_df = (
    all_runs_df
    .groupby(["dataset", "tmr_slots"], as_index=False)
    .agg(
        n=("test_acc", "count"),
        mean_acc=("test_acc", "mean"),
        std_acc=("test_acc", "std"),
        mean_epoch_time=("mean_epoch_time", "mean"),
        mean_wall_time=("wall_time", "mean"),
    )
    .sort_values(["dataset", "tmr_slots"])
)

summary_path = os.path.join(base_output_dir, "memory_slots_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\nSaved all runs to:", all_runs_path)
print("Saved summary table to:", summary_path)

for dataset in summary_df["dataset"].unique():
    print(f"\nDataset: {dataset}")
    print("-" * 100)
    print(summary_df[summary_df["dataset"] == dataset].to_string(index=False))
