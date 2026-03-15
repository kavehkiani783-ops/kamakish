import json
import glob
import os
import csv
import re


RESULTS_DIR = "results"
PATTERN = os.path.join(RESULTS_DIR, "tmr_v2_listops_synth_seed*_steps*_slots*_topk*_gate1.json")


def parse_filename(fname):
    seed = int(re.search(r"seed(\d+)", fname).group(1))
    steps = int(re.search(r"steps(\d+)", fname).group(1))
    slots = int(re.search(r"slots(\d+)", fname).group(1))
    return seed, steps, slots


def extract_metrics(data):
    """
    Try several possible result structures.
    """
    if "test" in data and "accuracy" in data["test"]:
        test_acc = data["test"]["accuracy"]
    elif "test_metrics" in data and "accuracy" in data["test_metrics"]:
        test_acc = data["test_metrics"]["accuracy"]
    elif "test_acc" in data:
        test_acc = data["test_acc"]
    else:
        test_acc = None

    if "epoch_time" in data:
        epoch_time = data["epoch_time"]
    elif "avg_epoch_time" in data:
        epoch_time = data["avg_epoch_time"]
    else:
        epoch_time = None

    return test_acc, epoch_time


def main():
    files = sorted(glob.glob(PATTERN))

    rows = []

    for f in files:
        with open(f) as fp:
            data = json.load(fp)

        seed, steps, slots = parse_filename(f)
        test_acc, epoch_time = extract_metrics(data)

        rows.append({
            "seed": seed,
            "steps": steps,
            "slots": slots,
            "test_accuracy": test_acc,
            "epoch_time": epoch_time,
        })

    rows.sort(key=lambda x: (x["steps"], x["seed"]))

    out_csv = os.path.join(RESULTS_DIR, "summary_tmr_v2_steps_seeds.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved summary to:", out_csv)

    print("\nSummary:")
    for r in rows:
        print(
            f"seed={r['seed']} | steps={r['steps']} | "
            f"test_acc={r['test_accuracy']}"
        )


if __name__ == "__main__":
    main()
