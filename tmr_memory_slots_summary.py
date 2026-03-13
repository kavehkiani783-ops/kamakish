import os
import json
import pandas as pd

BASE_OUTPUT_DIR = "ablation_runs/memory_slots_ablation"

all_rows = []

# -------------------------------------------------------
# Scan run folders
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

    # -------------------------------------------------------
    # Parse folder name safely
    # Example:
    # listops_synth_slots256_seed999
    # imdb_slots64_seed42
    # -------------------------------------------------------

    parts = run_dir.split("_")

    dataset_parts = []
    slots = None
    seed = None

    for p in parts:
        if p.startswith("slots"):
            slots = int(p.replace("slots", ""))
        elif p.startswith("seed"):
            seed = int(p.replace("seed", ""))
        else:
            dataset_parts.append(p)

    dataset = "_".join(dataset_parts)

    row = {
        "dataset": dataset,
        "tmr_slots": slots,
        "seed": seed,
        "test_acc": results.get("test_acc"),
        "mean_epoch_time": results.get("mean_epoch_time"),
        "wall_time": results.get("wall_time")
    }

    all_rows.append(row)

# -------------------------------------------------------
# Build dataframe
# -------------------------------------------------------

df = pd.DataFrame(all_rows)

all_runs_path = os.path.join(BASE_OUTPUT_DIR, "memory_slots_all_runs.csv")
df.to_csv(all_runs_path, index=False)

print("\nSaved full run table:")
print(all_runs_path)

# -------------------------------------------------------
# Summary table
# -------------------------------------------------------

summary_df = (
    df
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
# Print results
# -------------------------------------------------------

for dataset in summary_df["dataset"].unique():

    print(f"\nDataset: {dataset}")
    print("-" * 90)

    print(
        summary_df[summary_df["dataset"] == dataset]
        .to_string(index=False)
    )
