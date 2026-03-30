import glob
import json
import os
import re
import csv
from collections import defaultdict

RESULTS_DIR = "results"
PATTERN = os.path.join(RESULTS_DIR, "*.json")

FINAL_SEEDS = {42, 43, 44}
BASELINE_MODELS = {"meanpool", "bilstm", "tiny_transformer", "transformer_base"}

def parse_filename(name):
    stem = name[:-5] if name.endswith(".json") else name

    model = None
    valid_models = [
        "transformer_base",
        "tiny_transformer",
        "meanpool",
        "bilstm",
        "HubNet_v1",
        "HubNet_v2",
    ]

    for candidate in valid_models:
        prefix = candidate + "_"
        if stem.startswith(prefix):
            model = candidate
            rest = stem[len(prefix):]
            break
    else:
        return None

    m = re.match(r"(.+?)_seed(\d+)(.*)", rest)
    if not m:
        return None

    dataset = m.group(1)
    seed = int(m.group(2))
    suffix = m.group(3)

    def extract_int(pattern):
        mm = re.search(pattern, suffix)
        return int(mm.group(1)) if mm else None

    return {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "steps": extract_int(r"_steps(\d+)"),
        "slots": extract_int(r"_slots(\d+)"),
        "topk": extract_int(r"_topk(\d+)"),
        "gate": extract_int(r"_gate(\d+)"),
        "file": name,
    }

def find_first_key(obj, target_keys):
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

def keep_row(meta):
    if meta["seed"] not in FINAL_SEEDS:
        return False

    if meta["model"] in BASELINE_MODELS:
        return True

    if meta["model"] == "hubnet_v1":
        return (
            meta["steps"] == 1 and
            meta["slots"] == 32 and
            meta["topk"] == 0 and
            meta["gate"] == 0
        )

    if meta["model"] == "hubnet_v2":
        return (
            meta["steps"] == 4 and
            meta["slots"] == 32 and
            meta["topk"] == 0 and
            meta["gate"] == 1
        )

    return False

def main():
    rows = []

    for path in sorted(glob.glob(PATTERN)):
        name = os.path.basename(path)
        meta = parse_filename(name)
        if meta is None:
            continue

        if not keep_row(meta):
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        val_acc = find_first_key(data, {"val_accuracy", "best_val_acc", "best_val_accuracy", "val_acc"})
        test_acc = find_first_key(data, {"test_accuracy", "test_acc", "accuracy"})
        epoch_time = find_first_key(data, {"epoch_time_min", "avg_epoch_time_min", "epoch_time", "avg_epoch_time"})
        num_parameters = find_first_key(data, {"num_parameters", "params", "parameter_count"})

        rows.append({
            "file": name,
            "dataset": meta["dataset"],
            "model": meta["model"],
            "seed": meta["seed"],
            "steps": meta["steps"],
            "slots": meta["slots"],
            "topk": meta["topk"],
            "gate": meta["gate"],
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "epoch_time_min": epoch_time,
            "num_parameters": num_parameters,
        })

    out_csv = os.path.join(RESULTS_DIR, "summary_final_models.csv")
    fieldnames = [
        "file", "dataset", "model", "seed",
        "steps", "slots", "topk", "gate",
        "val_accuracy", "test_accuracy",
        "epoch_time_min", "num_parameters"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary to: {out_csv}")

    print("\nPer-run summary:")
    for row in sorted(rows, key=lambda r: (r["dataset"], r["model"], r["seed"])):
        extra = []
        if row["model"] in {"tmr", "tmr_v2"}:
            extra.append(f"steps={row['steps']}")
            extra.append(f"slots={row['slots']}")
            extra.append(f"topk={row['topk']}")
            extra.append(f"gate={row['gate']}")
        extra_str = " | " + " | ".join(extra) if extra else ""

        print(
            f"dataset={row['dataset']} | model={row['model']} | seed={row['seed']}"
            f"{extra_str} | val_acc={fmt(row['val_accuracy'])} | "
            f"test_acc={fmt(row['test_accuracy'])} | epoch_time={fmt(row['epoch_time_min'])}"
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row)

    print("\nAverages by dataset and model:")
    for (dataset, model) in sorted(grouped.keys()):
        grp = grouped[(dataset, model)]
        avg_val = mean([r["val_accuracy"] for r in grp])
        avg_test = mean([r["test_accuracy"] for r in grp])
        avg_time = mean([r["epoch_time_min"] for r in grp])

        config_bits = []
        if model in {"tmr", "tmr_v2"}:
            config_bits.append(f"steps={grp[0]['steps']}")
            config_bits.append(f"slots={grp[0]['slots']}")
            config_bits.append(f"topk={grp[0]['topk']}")
            config_bits.append(f"gate={grp[0]['gate']}")

        config_str = " | " + " | ".join(config_bits) if config_bits else ""

        print(
            f"dataset={dataset} | model={model}{config_str} | "
            f"avg_val_acc={fmt(avg_val)} | avg_test_acc={fmt(avg_test)} | "
            f"avg_epoch_time={fmt(avg_time)}"
        )

if __name__ == "__main__":
    main()
