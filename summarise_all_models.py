import glob
import json
import os
import re
import csv
from collections import defaultdict


RESULTS_DIR = "results"
PATTERN = os.path.join(RESULTS_DIR, "*.json")


VALID_MODELS = {
    "meanpool",
    "bilstm",
    "tiny_transformer",
    "transformer_base",
    "tmr",
    "tmr_v2",
}


def parse_filename(name):
    """
    Supports filenames like:
      meanpool_imdb_seed42.json
      tmr_imdb_seed42_steps1_slots32_topk0_gate0.json
      tmr_v2_listops_synth_seed44_steps4_slots32_topk0_gate1.json
    """
    stem = name[:-5] if name.endswith(".json") else name

    model = None
    dataset = None
    seed = None
    steps = None
    slots = None
    topk = None
    gate = None

    # model
    for candidate in sorted(VALID_MODELS, key=len, reverse=True):
        prefix = candidate + "_"
        if stem.startswith(prefix):
            model = candidate
            rest = stem[len(prefix):]
            break
    else:
        return None

    # dataset and rest
    m = re.match(r"(.+?)_seed(\d+)(.*)", rest)
    if not m:
        return None

    dataset = m.group(1)
    seed = int(m.group(2))
    suffix = m.group(3)

    m_steps = re.search(r"_steps(\d+)", suffix)
    m_slots = re.search(r"_slots(\d+)", suffix)
    m_topk = re.search(r"_topk(\d+)", suffix)
    m_gate = re.search(r"_gate(\d+)", suffix)

    steps = int(m_steps.group(1)) if m_steps else None
    slots = int(m_slots.group(1)) if m_slots else None
    topk = int(m_topk.group(1)) if m_topk else None
    gate = int(m_gate.group(1)) if m_gate else None

    return {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "slots": slots,
        "topk": topk,
        "gate": gate,
    }


def find_first_key(obj, target_keys):
    """
    Recursively search nested dict/list structures for the first matching scalar key.
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

    for path in files:
        name = os.path.basename(path)
        meta = parse_filename(name)
        if meta is None:
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        val_acc = find_first_key(
            data,
            {"val_accuracy", "best_val_acc", "best_val_accuracy", "val_acc"},
        )

        test_acc = find_first_key(
            data,
            {"test_accuracy", "test_acc", "accuracy"},
        )

        epoch_time = find_first_key(
            data,
            {"epoch_time_min", "avg_epoch_time_min", "epoch_time", "avg_epoch_time"},
        )

        num_parameters = find_first_key(
            data,
            {"num_parameters", "params", "parameter_count"},
        )

        rows.append(
            {
                "file": name,
                "model": meta["model"],
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
        )

    if not rows:
        print("No matching result files found.")
        return

    out_csv = os.path.join(RESULTS_DIR, "summary_all_models.csv")
    fieldnames = [
        "file",
        "model",
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
    for (dataset, model) in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        grp = grouped[(dataset, model)]
        avg_val = mean([r["val_accuracy"] for r in grp])
        avg_test = mean([r["test_accuracy"] for r in grp])
        avg_time = mean([r["epoch_time_min"] for r in grp])

        # show TMR config if same across grouped rows
        steps_vals = sorted({r["steps"] for r in grp if r["steps"] is not None})
        slots_vals = sorted({r["slots"] for r in grp if r["slots"] is not None})
        topk_vals = sorted({r["topk"] for r in grp if r["topk"] is not None})
        gate_vals = sorted({r["gate"] for r in grp if r["gate"] is not None})

        config_bits = []
        if steps_vals:
            config_bits.append(f"steps={steps_vals}")
        if slots_vals:
            config_bits.append(f"slots={slots_vals}")
        if topk_vals:
            config_bits.append(f"topk={topk_vals}")
        if gate_vals:
            config_bits.append(f"gate={gate_vals}")

        config_str = " | " + " | ".join(config_bits) if config_bits else ""

        print(
            f"dataset={dataset} | model={model}{config_str} | "
            f"avg_val_acc={fmt(avg_val)} | avg_test_acc={fmt(avg_test)} | "
            f"avg_epoch_time={fmt(avg_time)}"
        )


if __name__ == "__main__":
    main()
