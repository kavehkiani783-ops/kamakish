import glob
import json
import os
import re
import csv


RESULTS_DIR = "results"
PATTERN = os.path.join(RESULTS_DIR, "tmr_v2_listops_synth_seed*_steps*_slots*_topk*_gate1.json")


def extract_from_filename(path):
    name = os.path.basename(path)

    seed_match = re.search(r"seed(\d+)", name)
    steps_match = re.search(r"steps(\d+)", name)
    slots_match = re.search(r"slots(\d+)", name)
    topk_match = re.search(r"topk(\d+)", name)
    gate_match = re.search(r"gate(\d+)", name)

    return {
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
            "seed": meta["seed"],
            "steps": meta["steps"],
            "slots": meta["slots"],
            "topk": meta["topk"],
            "gate": meta["gate"],
            "val_accuracy": (
                safe_get(data, "val", "accuracy")
                or safe_get(data, "val_metrics", "accuracy")
                or safe_get(data, "best_val", "accuracy")
            ),
            "test_accuracy": (
                safe_get(data, "test", "accuracy")
                or safe_get(data, "test_metrics", "accuracy")
                or safe_get(data, "accuracy")
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

    out_csv = os.path.join(RESULTS_DIR, "summary_tmr_v2_steps_seeds.csv")
    fieldnames = [
        "file",
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

    print("\nQuick view:")
    for row in rows:
        print(
            f"seed={row['seed']} | steps={row['steps']} | "
            f"test_acc={row['test_accuracy']} | epoch_time={row['epoch_time_min']}"
        )


if __name__ == "__main__":
    main()
