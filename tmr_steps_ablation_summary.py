import json
import math
from pathlib import Path

import pandas as pd

# -------------------------------------------------------
# Fixed paths
# -------------------------------------------------------

BASE_OUTPUT_DIR = Path("/home/ubuntu/taha/ablation_runs/steps_ablation")
ALL_RUNS_CSV = BASE_OUTPUT_DIR / "steps_all_runs_comparison.csv"
SUMMARY_CSV = BASE_OUTPUT_DIR / "steps_summary.csv"

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def safe_float(x):
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_json_file(run_dir):
    json_files = sorted([p for p in run_dir.iterdir() if p.suffix == ".json"])
    if not json_files:
        return None

    preferred = [p for p in json_files if p.name.startswith("tmr_")]
    if preferred:
        return preferred[0]

    return json_files[0]


def parse_run_dir_name(run_dir_name):
    """
    Expected folder pattern:
    imdb_steps0_seed42
    listops_synth_steps4_seed123
    """
    parts = run_dir_name.split("_")

    seed_idx = None
    steps_idx = None

    for i, p in enumerate(parts):
        if p.startswith("steps"):
            steps_idx = i
        if p.startswith("seed"):
            seed_idx = i

    if steps_idx is None or seed_idx is None:
        return None

    dataset = "_".join(parts[:steps_idx])

    try:
        steps = int(parts[steps_idx].replace("steps", ""))
        seed = int(parts[seed_idx].replace("seed", ""))
    except Exception:
        return None

    return {
        "dataset": dataset,
        "tmr_steps": steps,
        "seed": seed,
    }


def recursive_find_key(obj, target_keys):
    """
    Search recursively through nested dict/list structures
    and return the first non-None value matching any target key.
    """
    if isinstance(target_keys, str):
        target_keys = [target_keys]

    if isinstance(obj, dict):
        for k in target_keys:
            if k in obj and obj[k] is not None:
                return obj[k]

        for v in obj.values():
            found = recursive_find_key(v, target_keys)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = recursive_find_key(item, target_keys)
            if found is not None:
                return found

    return None


def extract_metrics(results):
    return {
        "test_acc": safe_float(
            recursive_find_key(results, ["test_acc", "accuracy", "acc", "test_accuracy"])
        ),
        "macro_f1": safe_float(
            recursive_find_key(results, ["macro_f1"])
        ),
        "weighted_f1": safe_float(
            recursive_find_key(results, ["weighted_f1"])
        ),
        "balanced_accuracy": safe_float(
            recursive_find_key(results, ["balanced_accuracy"])
        ),
        "auroc": safe_float(
            recursive_find_key(results, ["auroc", "roc_auc"])
        ),
        "auprc": safe_float(
            recursive_find_key(results, ["auprc", "pr_auc"])
        ),
        "nll": safe_float(
            recursive_find_key(results, ["nll", "log_loss"])
        ),
        "brier": safe_float(
            recursive_find_key(results, ["brier", "brier_score"])
        ),
        "ece": safe_float(
            recursive_find_key(results, ["ece"])
        ),
        "mean_epoch_time": safe_float(
            recursive_find_key(results, ["mean_epoch_time", "epoch_time_min", "epoch_time", "train_time_min"])
        ),
        "wall_time": safe_float(
            recursive_find_key(results, ["wall_time", "wall_time_min", "elapsed_time", "time"])
        ),
    }


# -------------------------------------------------------
# Collect existing runs
# -------------------------------------------------------

print("\nReading existing TMR steps ablation runs\n")

if not BASE_OUTPUT_DIR.exists():
    raise FileNotFoundError(f"Directory not found: {BASE_OUTPUT_DIR}")

all_rows = []

for item in sorted(BASE_OUTPUT_DIR.iterdir()):
    if not item.is_dir():
        continue

    parsed = parse_run_dir_name(item.name)
    if parsed is None:
        continue

    json_file = find_json_file(item)

    if json_file is None:
        print(f"WARNING: No JSON file found in {item}")
        row = {
            "dataset": parsed["dataset"],
            "tmr_steps": parsed["tmr_steps"],
            "seed": parsed["seed"],
            "test_acc": None,
            "macro_f1": None,
            "weighted_f1": None,
            "balanced_accuracy": None,
            "auroc": None,
            "auprc": None,
            "nll": None,
            "brier": None,
            "ece": None,
            "mean_epoch_time": None,
            "wall_time": None,
        }
        all_rows.append(row)
        continue

    print(f"Loading results from: {json_file}")

    try:
        results = load_json(json_file)
    except Exception as e:
        print(f"WARNING: Failed to read {json_file}: {e}")
        results = {}

    metrics = extract_metrics(results)

    row = {
        "dataset": parsed["dataset"],
        "tmr_steps": parsed["tmr_steps"],
        "seed": parsed["seed"],
        "test_acc": metrics["test_acc"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "nll": metrics["nll"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        "mean_epoch_time": metrics["mean_epoch_time"],
        "wall_time": metrics["wall_time"],
    }

    all_rows.append(row)

if len(all_rows) == 0:
    raise RuntimeError(
        f"No valid steps-ablation run folders were found inside: {BASE_OUTPUT_DIR}"
    )

# -------------------------------------------------------
# Create full raw results table
# -------------------------------------------------------

all_runs_df = pd.DataFrame(all_rows)
all_runs_df = all_runs_df.sort_values(["dataset", "tmr_steps", "seed"]).reset_index(drop=True)
all_runs_df.to_csv(ALL_RUNS_CSV, index=False)

print("\nSaved full run table:")
print(ALL_RUNS_CSV)

# -------------------------------------------------------
# Create summary table
# -------------------------------------------------------

summary_df = (
    all_runs_df
    .groupby(["dataset", "tmr_steps"], as_index=False)
    .agg(
        n=("seed", "count"),
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

summary_df.to_csv(SUMMARY_CSV, index=False)

print("\nSaved summary table:")
print(SUMMARY_CSV)

# -------------------------------------------------------
# Print detected runs nicely
# -------------------------------------------------------

print("\nDetected runs:")
print("-" * 120)
print(all_runs_df.to_string(index=False))

# -------------------------------------------------------
# Print summary nicely
# -------------------------------------------------------

for dataset in summary_df["dataset"].unique():
    print(f"\nDataset: {dataset}")
    print("-" * 120)
    print(summary_df[summary_df["dataset"] == dataset].to_string(index=False))

print("\nTMR steps ablation summary finished successfully.")
