import glob
import json
import os
import re
import csv
from collections import defaultdict


RESULTS_DIR = "results"
PATTERN = os.path.join(RESULTS_DIR, "HubNet_v2_*_seed*_steps*_slots*_topk*_gate*.json")


def extract_from_filename(path):
    name = os.path.basename(path)

    dataset_match = re.search(r"HubNet_v2_(.+?)_seed\d+_steps", name)
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


def find_first_key(obj, target_keys):
    """
    Recursively search nested dict/list structures for the first matching key.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in target_keys and isinstance(v, (int, float)):
                return v
        for v in obj.values():
            found = find_first_key(v, target_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_first_key(item, target_keys)
            if found is not None:
                return found
    return None


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

        val_acc = find_first_key(
            data,
            {
                "val_accuracy",
                "best_val_acc",
                "best_val_accuracy",
                "val_acc",
            },
        )

        test_acc = find_first_key(
            data,
            {
                "test_accuracy",
                "test_acc",
                "accuracy",
            },
        )

        epoch_time = find_first_key(
            data,
            {
                "epoch_time_min",
                "avg_epoch_time_min",
                "epoch_time",
                "avg_epoch_time",
            },
        )

        num_parameters = find_first_key(
            data,
            {
                "num_parameters",
                "params",
                "parameter_count",
            },
        )

        row = {
            "file": os.path.basename(path),
            "dataset": meta["dataset"],
            "seed": meta["seed"],
            "steps": meta["steps"],
            "slots": meta["slots"],
            "topk": meta["topk"],
            "gate": meta["gate"],
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "epoch_time_min": epoch_time,
            "num_parameters": num_parameters,
        }
        rows.append(row)

    out_csv = os.path.join(RESULTS_DIR, "summary_HubNet_v2_runs.csv")
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
