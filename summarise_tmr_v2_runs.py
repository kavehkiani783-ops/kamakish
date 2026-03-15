import json
import glob
import os
import csv
import re

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


def find_first_key(obj, candidate_keys):
    """
    Recursively search dict/list structures for the first matching key.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in candidate_keys:
                return v
        for v in obj.values():
            found = find_first_key(v, candidate_keys)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = find_first_key(item, candidate_keys)
            if found is not None:
                return found

    return None


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

        test_acc = find_first_key(
            data,
            {
                "test_acc",
                "test_accuracy",
                "accuracy_test",
            }
        )

        # Also try nested "accuracy" only under test-like containers
        if test_acc is None and isinstance(data, dict):
            for k, v in data.items():
                if "test" in k.lower() and isinstance(v, dict):
                    if "accuracy" in v:
                        test_acc = v["accuracy"]
                        break

        val_acc = find_first_key(
            data,
            {
                "val_acc",
                "val_accuracy",
                "accuracy_val",
            }
        )

        if val_acc is None and isinstance(data, dict):
            for k, v in data.items():
                if "val" in k.lower() and isinstance(v, dict):
                    if "accuracy" in v:
                        val_acc = v["accuracy"]
                        break

        epoch_time = find_first_key(
            data,
            {
                "epoch_time",
                "epoch_time_min",
                "avg_epoch_time",
                "avg_epoch_time_min",
            }
        )

        num_parameters = find_first_key(
            data,
            {
                "num_parameters",
                "params",
                "parameter_count",
            }
        )

        rows.append({
            "file": os.path.basename(fpath),
            "seed": seed,
            "steps": steps,
            "slots": slots,
            "topk": topk,
            "gate": gate,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "epoch_time": epoch_time,
            "num_parameters": num_parameters,
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
            f"test_acc={r['test_accuracy']} | val_acc={r['val_accuracy']} | "
            f"epoch_time={r['epoch_time']}"
        )


if __name__ == "__main__":
    main()
