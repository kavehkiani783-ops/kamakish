import json
import glob
import os
import csv
import re
from statistics import mean

RESULTS_DIR = "results"
PATTERN = os.path.join(
    RESULTS_DIR,
    "tmr_v2_listops_synth_seed*_steps*_slots*_topk*_gate1.json"
)


def parse_filename(fname):
    seed = int(re.search(r"seed(\d+)", fname).group(1))
    steps = int(re.search(r"steps(\d+)", fname).group(1))
    slots = int(re.search(r"slots(\d+)", fname).group(1))
    topk = int(re.search(r"topk(\d+)", fname).group(1))
    gate = int(re.search(r"gate(\d+)", fname).group(1))
    return seed, steps, slots, topk, gate


def main():
    files = sorted(glob.glob(PATTERN))

    if not files:
        print("No matching result files found.")
        return

    rows = []

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        seed, steps, slots, topk, gate = parse_filename(os.path.basename(fpath))

        best_val_acc = data.get("best_val_accuracy", None)

        best_metrics = data.get("best_metrics", {})
        val_metrics = best_metrics.get("val", {})
        test_metrics = best_metrics.get("test", {})

        val_acc = val_metrics.get("accuracy", None)
        test_acc = test_metrics.get("accuracy", None)

        total_minutes = data.get("total_minutes", None)

        history = data.get("history", {})
        epochs = history.get("epochs", [])
        if epochs:
            last_epoch_time = epochs[-1].get("epoch_time_min", None)
        else:
            last_epoch_time = None

        rows.append({
            "file": os.path.basename(fpath),
            "seed": seed,
            "steps": steps,
            "slots": slots,
            "topk": topk,
            "gate": gate,
            "best_val_accuracy": best_val_acc,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "last_epoch_time_min": last_epoch_time,
            "total_minutes": total_minutes,
        })

    rows.sort(key=lambda r: (r["steps"], r["seed"]))

    out_csv = os.path.join(RESULTS_DIR, "summary_tmr_v2_steps_seeds.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved summary to:", out_csv)
    print("\nSummary:")
    for r in rows:
        print(
            f"seed={r['seed']} | steps={r['steps']} | "
            f"val_acc={r['val_accuracy']:.4f} | "
            f"test_acc={r['test_accuracy']:.4f} | "
            f"epoch_time={r['last_epoch_time_min']:.4f}"
        )

    by_steps = {}
    for r in rows:
        by_steps.setdefault(r["steps"], []).append(r)

    print("\nAverages by steps:")
    for steps in sorted(by_steps):
        group = by_steps[steps]
        avg_val = mean(r["val_accuracy"] for r in group if r["val_accuracy"] is not None)
        avg_test = mean(r["test_accuracy"] for r in group if r["test_accuracy"] is not None)
        avg_epoch = mean(r["last_epoch_time_min"] for r in group if r["last_epoch_time_min"] is not None)
        print(
            f"steps={steps} | avg_val_acc={avg_val:.4f} | "
            f"avg_test_acc={avg_test:.4f} | avg_epoch_time={avg_epoch:.4f}"
        )


if __name__ == "__main__":
    main()
