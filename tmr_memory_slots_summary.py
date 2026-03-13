import os
import json
import pandas as pd

BASE_OUTPUT_DIR = "ablation_runs/memory_slots_ablation"

all_rows = []

# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def parse_run_dir(run_dir: str):
    """
    Parses folder names like:
      imdb_slots64_seed42
      listops_synth_slots256_seed999
    """
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
    return dataset, slots, seed


def extract_mean_epoch_time(history):
    """
    Try to compute mean epoch time from history.
    Handles common possibilities safely.
    """
    if not isinstance(history, list) or len(history) == 0:
        return None

    epoch_times = []

    for item in history:
        if not isinstance(item, dict):
            continue

        # Try common key names
        for k in ["epoch_time", "epoch_minutes", "train_minutes", "minutes"]:
            if k in item and item[k] is not None:
                epoch_times.append(item[k])
                break

    if len(epoch_times) == 0:
        return None

    return sum(epoch_times) / len(epoch_times)


def extract_test_acc(best_metrics):
    """
    Try common accuracy keys inside best_metrics.
    """
    if not isinstance(best_metrics, dict):
        return None

    for k in ["test_acc", "accuracy", "acc", "val_acc", "best_val_accuracy"]:
        if k in best_metrics and best_metrics[k] is not None:
            return best_metrics[k]

    return None


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

    dataset, slots, seed = parse_run_dir(run_dir)

    best_metrics = results.get("best_metrics", {})
    history = results.get("history", [])
    total_minutes = results.get("total_minutes", None)

    test_acc = extract_test_acc(best_metrics)
    mean_epoch_time = extract_mean_epoch_time(history)

    row = {
        "dataset": dataset,
        "tmr_slots": slots,
        "seed": seed,
        "test_acc": test_acc,
        "mean_epoch_time": mean_epoch_time,
        "wall_time": total_minutes,
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

print("\nDetected rows:")
print(df.to_string(index=False))

# -------------------------------------------------------
# Summary table
# -------------------------------------------------------

summary_df = (
    df
    .groupby(["dataset", "tmr_slots"], as_index=False)
    .agg(
        n=("test_acc", lambda x: x.notna().sum()),
        mean_acc=("test_acc", "mean"),
        std_acc=("test_acc", "std"),
        mean_epoch_time=("mean_epoch_time", "mean"),
        mean_wall_time=("wall_time", "mean"),
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
    print(summary_df[summary_df["dataset"] == dataset].to_string(index=False))
