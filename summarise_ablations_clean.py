import glob
import json
import os
import re
import csv
from collections import defaultdict
from statistics import mean, stdev

ABLATION_ROOT = "ablation_runs"
PATTERN = os.path.join(ABLATION_ROOT, "**", "*.json")

ABLATION_MODELS = {"hubnet_v1", "hubnet_v2"}


def parse_filename(name):
    stem = name[:-5] if name.endswith(".json") else name

    valid_models = [
        "hubnet_v2",
        "hubnet_v1",
        "transformer_base",
        "tiny_transformer",
        "meanpool",
        "bilstm",
    ]

    model = None
    rest = None
    for candidate in valid_models:
        prefix = candidate + "_"
        if stem.startswith(prefix):
            model = candidate
            rest = stem[len(prefix):]
            break

    if model is None:
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
        "file": name,
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "steps": extract_int(r"_steps(\d+)"),
        "slots": extract_int(r"_slots(\d+)"),
        "topk": extract_int(r"_topk(\d+)"),
        "gate": extract_int(r"_gate(\d+)"),
    }


def find_first_key(obj, target_keys):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in target_keys and isinstance(v, (int, float)):
                return v
        for v in obj.values():
            out = find_first_key(v, target_keys)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for item in obj:
            out = find_first_key(item, target_keys)
            if out is not None:
                return out
    return None


def is_real_result_json(data):
    if not isinstance(data, dict):
        return False

    if data.get("status") == "failed_missing_result_json":
        return False

    if find_first_key(data, {"val_accuracy", "best_val_acc", "best_val_accuracy", "val_acc"}) is not None:
        return True

    if find_first_key(data, {"test_accuracy", "test_acc", "accuracy"}) is not None:
        return True

    return False


def fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "NA"


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def safe_std(xs):
    xs = [x for x in xs if x is not None]
    return stdev(xs) if len(xs) > 1 else (0.0 if len(xs) == 1 else None)


def load_rows():
    rows = []

    for path in sorted(glob.glob(PATTERN, recursive=True)):
        name = os.path.basename(path)
        meta = parse_filename(name)
        if meta is None:
            continue

        if meta["model"] not in ABLATION_MODELS:
            continue

        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        if not is_real_result_json(data):
            continue

        val_acc = find_first_key(data, {"val_accuracy", "best_val_acc", "best_val_accuracy", "val_acc"})
        test_acc = find_first_key(data, {"test_accuracy", "test_acc", "accuracy"})
        epoch_time = find_first_key(data, {"epoch_time_min", "avg_epoch_time_min", "epoch_time", "avg_epoch_time"})
        num_parameters = find_first_key(data, {"num_parameters", "params", "parameter_count"})

        rows.append({
            **meta,
            "source_path": path,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "epoch_time": epoch_time,
            "num_parameters": num_parameters,
        })

    return rows


def save_all_runs(rows):
    out_csv = os.path.join(ABLATION_ROOT, "ablation_all_runs_clean.csv")
    fieldnames = [
        "file", "source_path", "dataset", "model", "seed",
        "steps", "slots", "topk", "gate",
        "val_accuracy", "test_accuracy", "epoch_time", "num_parameters"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved all clean runs to: {out_csv}")


def summarise_ablation(rows, ablation_name, fixed_filters):
    filtered = []
    for r in rows:
        ok = True
        for k, v in fixed_filters.items():
            if r.get(k) != v:
                ok = False
                break
        if ok and r.get(ablation_name) is not None:
            filtered.append(r)

    grouped = defaultdict(list)
    for r in filtered:
        grouped[(r["dataset"], r["model"], r[ablation_name])].append(r)

    out_rows = []
    for (dataset, model, ablation_value), grp in sorted(grouped.items()):
        out_rows.append({
            "dataset": dataset,
            "model": model,
            ablation_name: ablation_value,
            "n_runs": len(grp),
            "mean_val_accuracy": safe_mean([x["val_accuracy"] for x in grp]),
            "std_val_accuracy": safe_std([x["val_accuracy"] for x in grp]),
            "mean_test_accuracy": safe_mean([x["test_accuracy"] for x in grp]),
            "std_test_accuracy": safe_std([x["test_accuracy"] for x in grp]),
            "mean_epoch_time": safe_mean([x["epoch_time"] for x in grp]),
            "std_epoch_time": safe_std([x["epoch_time"] for x in grp]),
            "mean_num_parameters": safe_mean([x["num_parameters"] for x in grp]),
        })

    out_csv = os.path.join(ABLATION_ROOT, f"ablation_{ablation_name}_summary.csv")
    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "dataset", "model", ablation_name, "n_runs",
        "mean_val_accuracy", "std_val_accuracy",
        "mean_test_accuracy", "std_test_accuracy",
        "mean_epoch_time", "std_epoch_time",
        "mean_num_parameters"
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Saved {ablation_name} summary to: {out_csv}")

    print(f"\n{ablation_name.upper()} ABLATION")
    if not out_rows:
        print("No matching runs found.")
        return

    for row in out_rows:
        print(
            f"dataset={row['dataset']} | model={row['model']} | {ablation_name}={row[ablation_name]} | "
            f"test_acc={fmt(row['mean_test_accuracy'])} ± {fmt(row['std_test_accuracy'])} | "
            f"epoch_time={fmt(row['mean_epoch_time'])}"
        )


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} clean result JSON files.")

    if not rows:
        print("No usable result files found.")
        return

    save_all_runs(rows)

    # Steps ablation: fix slots=32, topk=0, gate=1
    summarise_ablation(
        rows,
        ablation_name="steps",
        fixed_filters={"slots": 32, "topk": 0, "gate": 1}
    )

    # Slots ablation: fix steps=4, topk=0, gate=1
    summarise_ablation(
        rows,
        ablation_name="slots",
        fixed_filters={"steps": 4, "topk": 0, "gate": 1}
    )

    # Top-k ablation: fix steps=4, slots=32, gate=1
    summarise_ablation(
        rows,
        ablation_name="topk",
        fixed_filters={"steps": 4, "slots": 32, "gate": 1}
    )


if __name__ == "__main__":
    main()
