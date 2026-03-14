import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

OUTPUT_DIR = Path("models_runs")
ALL_RUNS_CSV = OUTPUT_DIR / "all_runs.csv"
SUMMARY_CSV = OUTPUT_DIR / "summary_table.csv"

DATASET_ORDER = ["imdb", "listops_synth"]
MODEL_ORDER = ["meanpool", "bilstm", "tiny_transformer", "transformer_base", "tmr"]


METRIC_PATTERNS = {
    "test_acc": [
        r"test[_\s-]*acc(?:uracy)?\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"\baccuracy\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bacc\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "macro_f1": [
        r"macro[_\s-]*f1\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "weighted_f1": [
        r"weighted[_\s-]*f1\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "balanced_accuracy": [
        r"balanced[_\s-]*accuracy\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "auroc": [
        r"auroc\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "auprc": [
        r"auprc\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "nll": [
        r"nll\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "brier": [
        r"brier(?:[_\s-]*score)?\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "ece": [
        r"ece\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "epoch_time_min": [
        r"epoch[_\s-]*time.*?[:=]\s*([0-9]*\.?[0-9]+)",
        r"mean[_\s-]*epoch[_\s-]*time.*?[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "tokens_per_sec": [
        r"tokens[/_\s-]*sec\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"tokens\s*per\s*sec(?:ond)?\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "gpu_mem_gb": [
        r"gpu.*?mem.*?[:=]\s*([0-9]*\.?[0-9]+)",
        r"memory.*?[:=]\s*([0-9]*\.?[0-9]+)",
    ],
    "param_count": [
        r"param(?:eter)?s?\s*[:=]\s*([0-9]+)",
        r"num[_\s-]*params\s*[:=]\s*([0-9]+)",
    ],
}


def try_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def extract_last_json_object(text):
    if not isinstance(text, str) or not text.strip():
        return None

    candidates = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for chunk in reversed(candidates):
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def find_metric_in_text(text, patterns):
    if not isinstance(text, str):
        return None

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return try_float(match.group(1))
    return None


def parse_metrics(stdout_text):
    result = {
        "test_acc": None,
        "macro_f1": None,
        "weighted_f1": None,
        "balanced_accuracy": None,
        "auroc": None,
        "auprc": None,
        "nll": None,
        "brier": None,
        "ece": None,
        "epoch_time_min": None,
        "tokens_per_sec": None,
        "gpu_mem_gb": None,
        "param_count": None,
    }

    obj = extract_last_json_object(stdout_text)
    if obj is not None:
        alias_map = {
            "test_acc": ["test_acc", "test_accuracy", "acc", "accuracy"],
            "macro_f1": ["macro_f1"],
            "weighted_f1": ["weighted_f1"],
            "balanced_accuracy": ["balanced_accuracy", "bal_acc"],
            "auroc": ["auroc"],
            "auprc": ["auprc"],
            "nll": ["nll"],
            "brier": ["brier", "brier_score"],
            "ece": ["ece"],
            "epoch_time_min": ["epoch_time_min", "epoch_time", "mean_epoch_time"],
            "tokens_per_sec": ["tokens_per_sec", "tokens_sec"],
            "gpu_mem_gb": ["gpu_mem_gb", "gpu_memory_gb"],
            "param_count": ["param_count", "params", "num_params"],
        }

        for key, aliases in alias_map.items():
            for alias in aliases:
                if alias in obj and obj[alias] is not None:
                    result[key] = try_float(obj[alias])
                    if result[key] is not None:
                        break

    for key, patterns in METRIC_PATTERNS.items():
        if result[key] is None:
            result[key] = find_metric_in_text(stdout_text, patterns)

    return result


def summarise(values):
    clean = [
        v for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if len(clean) == 0:
        return None, None
    if len(clean) == 1:
        return clean[0], 0.0
    return mean(clean), stdev(clean)


def dataset_index(name):
    return DATASET_ORDER.index(name) if name in DATASET_ORDER else 999


def model_index(name):
    return MODEL_ORDER.index(name) if name in MODEL_ORDER else 999


def fmt(value, decimals=4):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def main():
    if not OUTPUT_DIR.exists():
        print(f"Directory not found: {OUTPUT_DIR.resolve()}")
        return

    json_files = sorted(OUTPUT_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {OUTPUT_DIR.resolve()}")
        return

    all_rows = []

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"Skipping unreadable file: {path} | error: {e}")
            continue

        metrics = parse_metrics(raw.get("stdout", ""))

        row = {
            "dataset": raw.get("dataset"),
            "model": raw.get("model"),
            "seed": raw.get("seed"),
            "command": raw.get("command"),
            "return_code": raw.get("return_code"),
            "wall_time_min": try_float(raw.get("wall_time_min")),
        }
        row.update(metrics)
        all_rows.append(row)

    if not all_rows:
        print("No valid run records found.")
        return

    all_runs_fields = [
        "dataset",
        "model",
        "seed",
        "command",
        "return_code",
        "test_acc",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "auroc",
        "auprc",
        "nll",
        "brier",
        "ece",
        "epoch_time_min",
        "tokens_per_sec",
        "gpu_mem_gb",
        "param_count",
        "wall_time_min",
    ]

    with open(ALL_RUNS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_runs_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["dataset"], row["model"])].append(row)

    summary_rows = []
    for (dataset, model), rows in grouped.items():
        mean_acc, std_acc = summarise([r["test_acc"] for r in rows])
        mean_macro_f1, std_macro_f1 = summarise([r["macro_f1"] for r in rows])
        mean_weighted_f1, std_weighted_f1 = summarise([r["weighted_f1"] for r in rows])
        mean_bal_acc, std_bal_acc = summarise([r["balanced_accuracy"] for r in rows])
        mean_auroc, std_auroc = summarise([r["auroc"] for r in rows])
        mean_auprc, std_auprc = summarise([r["auprc"] for r in rows])
        mean_nll, std_nll = summarise([r["nll"] for r in rows])
        mean_brier, std_brier = summarise([r["brier"] for r in rows])
        mean_ece, std_ece = summarise([r["ece"] for r in rows])
        mean_epoch_time, std_epoch_time = summarise([r["epoch_time_min"] for r in rows])
        mean_wall_time, std_wall_time = summarise([r["wall_time_min"] for r in rows])

        summary_rows.append({
            "dataset": dataset,
            "model": model,
            "n": len(rows),
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_macro_f1": mean_macro_f1,
            "std_macro_f1": std_macro_f1,
            "mean_weighted_f1": mean_weighted_f1,
            "std_weighted_f1": std_weighted_f1,
            "mean_balanced_accuracy": mean_bal_acc,
            "std_balanced_accuracy": std_bal_acc,
            "mean_auroc": mean_auroc,
            "std_auroc": std_auroc,
            "mean_auprc": mean_auprc,
            "std_auprc": std_auprc,
            "mean_nll": mean_nll,
            "std_nll": std_nll,
            "mean_brier": mean_brier,
            "std_brier": std_brier,
            "mean_ece": mean_ece,
            "std_ece": std_ece,
            "mean_epoch_time_min": mean_epoch_time,
            "std_epoch_time_min": std_epoch_time,
            "mean_wall_time_min": mean_wall_time,
            "std_wall_time_min": std_wall_time,
            "success_runs": sum(1 for r in rows if r["return_code"] == 0),
        })

    summary_rows.sort(key=lambda r: (dataset_index(r["dataset"]), model_index(r["model"])))

    summary_fields = [
        "dataset",
        "model",
        "n",
        "mean_acc",
        "std_acc",
        "mean_macro_f1",
        "std_macro_f1",
        "mean_weighted_f1",
        "std_weighted_f1",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
        "mean_auroc",
        "std_auroc",
        "mean_auprc",
        "std_auprc",
        "mean_nll",
        "std_nll",
        "mean_brier",
        "std_brier",
        "mean_ece",
        "std_ece",
        "mean_epoch_time_min",
        "std_epoch_time_min",
        "mean_wall_time_min",
        "std_wall_time_min",
        "success_runs",
    ]

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSummary table:\n")

    header = (
        f"{'dataset':<15}"
        f"{'model':<20}"
        f"{'n':<5}"
        f"{'mean_acc':<12}"
        f"{'std_acc':<12}"
        f"{'mean_epoch_time':<18}"
        f"{'mean_wall_time':<18}"
        f"{'success':<10}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(
            f"{fmt(row['dataset'], 0):<15}"
            f"{fmt(row['model'], 0):<20}"
            f"{fmt(row['n'], 0):<5}"
            f"{fmt(row['mean_acc']):<12}"
            f"{fmt(row['std_acc']):<12}"
            f"{fmt(row['mean_epoch_time_min']):<18}"
            f"{fmt(row['mean_wall_time_min']):<18}"
            f"{fmt(row['success_runs'], 0):<10}"
        )

    print(f"\nSaved: {ALL_RUNS_CSV.resolve()}")
    print(f"Saved: {SUMMARY_CSV.resolve()}")


if __name__ == "__main__":
    main()
