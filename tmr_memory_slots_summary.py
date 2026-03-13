import os
import json
import pandas as pd

BASE_OUTPUT_DIR = "ablation_runs/memory_slots_ablation"

rows = []

# ----------------------------------------------------
# Parse folder names safely
# ----------------------------------------------------

def parse_run_dir(name):
    parts = name.split("_")

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

    return dataset, slots, seed


# ----------------------------------------------------
# Scan all run folders
# ----------------------------------------------------

for run_dir in os.listdir(BASE_OUTPUT_DIR):

    run_path = os.path.join(BASE_OUTPUT_DIR, run_dir)

    if not os.path.isdir(run_path):
        continue

    json_files = [f for f in os.listdir(run_path) if f.endswith(".json")]

    if len(json_files) == 0:
        continue

    json_file = os.path.join(run_path, json_files[0])

    with open(json_file) as f:
        results = json.load(f)

    dataset, slots, seed = parse_run_dir(run_dir)

    test_metrics = results["best_metrics"]["test"]

    acc = test_metrics["accuracy"]

    wall_time = results.get("total_minutes")

    row = {
        "dataset": dataset,
        "tmr_slots": slots,
        "seed": seed,
        "test_acc": acc,
        "wall_time": wall_time
    }

    rows.append(row)

# ----------------------------------------------------
# Build dataframe
# ----------------------------------------------------

df = pd.DataFrame(rows)

all_runs_path = os.path.join(BASE_OUTPUT_DIR, "memory_slots_all_runs.csv")
df.to_csv(all_runs_path, index=False)

print("\nSaved full run table:")
print(all_runs_path)

print("\nDetected runs:")
print(df.to_string(index=False))


# ----------------------------------------------------
# Summary table
# ----------------------------------------------------

summary_df = (
    df
    .groupby(["dataset", "tmr_slots"], as_index=False)
    .agg(
        n=("test_acc", "count"),
        mean_acc=("test_acc", "mean"),
        std_acc=("test_acc", "std"),
        mean_wall_time=("wall_time", "mean")
    )
    .sort_values(["dataset", "tmr_slots"])
)

summary_path = os.path.join(BASE_OUTPUT_DIR, "memory_slots_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\nSaved summary table:")
print(summary_path)

# ----------------------------------------------------
# Print summary nicely
# ----------------------------------------------------

for dataset in summary_df["dataset"].unique():

    print(f"\nDataset: {dataset}")
    print("-" * 90)

    print(
        summary_df[summary_df["dataset"] == dataset]
        .to_string(index=False)
    )
