import glob
import json
import os
import re
import csv
from collections import defaultdict


RESULTS_DIR = "results"
PATTERN = os.path.join(RESULTS_DIR, "tmr_v2_*_seed*_steps*_slots*_topk*_gate*.json")


def extract_from_filename(path):
    name = os.path.basename(path)

    dataset_match = re.search(r"tmr_v2_(.+?)_seed\d+_steps", name)
    seed_match = re.search(r"seed(\d+)", name)
    steps_match = re.search(r"steps(\d+)", name)
    slots_match = re.search(r"slots(\d+)", name)
    topk_match = re.search(r"topk(\d+)", name)
    gate_match = re.search(r"gate(\d+)", name)

    return {
        "dataset": dataset_match.group(1) if dataset_match else None,
        "seed": int(seed_match.group(1)) if seed_match else None,
        "steps": int(steps_match.group(1)) if steps_match else None,
        "slots": int(slots_match.group(1)) if slots_match else None,
        "topk": int(topk_match.group(1)) if topk_match else None,
        "gate": int(gate_match.group(1)) if gate_match else None,
    }


def safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "NA"


def main():
    files = sorted(glob.glob(PATTERN))
    rows = []

    if not files:
        print("No matching result files found.")
        return

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = extract_from_filename(path)

        row = {
            "file": os.path.basename(path),
            "dataset": meta["dataset"],
            "seed": meta["seed"],
            "steps": meta["steps"],
            "slots": meta["slots"],
            "topk": meta["topk"],
            "gate": meta["gate"],
            "val_accuracy": (
                safe_get(data, "val", "accuracy")
                or safe_get(data, "val_metrics", "accuracy")
                or safe_get(data, "best_val", "accuracy")
                or safe_get(data, "val_accuracy")
            ),
            "test_accuracy": (
                safe_get(data, "test", "accuracy")
                or safe_get(data, "test_metrics", "accuracy")
                or safe_get(data, "accuracy")
                or safe_get(data, "test_accuracy")
            ),
            "epoch_time_min": (
                safe_get(data, "epoch_time_min")
                or safe_get(data, "avg_epoch_time_min")
                or safe_get(data, "epoch_time")
            ),
            "num_parameters": (
                safe_get(data, "num_parameters")
                or safe_get(data, "params")
            ),
        }
        rows.append(row)

    out_csv = os.path.join(RESULTS_DIR, "summary_tmr_v2_runs.csv")
    fieldnames = [
        "file",
        "dataset",
        "seed",
        "steps",
        "slots",
        "topk",
        "gate",
        "val_accuracy",
        "test_accuracy",
        "epoch_time_min",
        "num_parameters",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary to: {out_csv}")

    print("\nSummary:")
    for row in sorted(rows, key=lambda r: (str(r["dataset"]), r["steps"], r["seed"])):
        print(
            f"dataset={row['dataset']} | seed={row['seed']} | steps={row['steps']} | "
            f"val_acc={fmt(row['val_accuracy'])} | test_acc={fmt(row['test_accuracy'])} | "
            f"epoch_time={fmt(row['epoch_time_min'])}"
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["steps"])].append(row)

    print("\nAverages by dataset and steps:")
    for (dataset, steps) in sorted(grouped.keys(), key=lambda x: (str(x[0]), x[1])):
        avg_val = mean([r["val_accuracy"] for r in grouped[(dataset, steps)]])
        avg_test = mean([r["test_accuracy"] for r in grouped[(dataset, steps)]])
        avg_time = mean([r["epoch_time_min"] for r in grouped[(dataset, steps)]])
        print(
            f"dataset={dataset} | steps={steps} | "
            f"avg_val_acc={fmt(avg_val)} | avg_test_acc={fmt(avg_test)} | "
            f"avg_epoch_time={fmt(avg_time)}"
        )


if __name__ == "__main__":
    main()
