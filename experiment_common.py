import csv
import json
import math
import shutil
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def remove_dir_if_empty(path: Path) -> None:
    if path.exists() and path.is_dir():
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()


def clear_directory_contents(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)


def expected_main_result_path(model: str, dataset: str, seed: int, results_dir: Path) -> Path:
    return results_dir / f"{model}_{dataset}_seed{seed}.json"


def extract_metric(payload: Dict[str, Any], candidate_keys: Iterable[str]) -> Optional[float]:
    """
    Try several possible key names because different main.py versions often use
    slightly different field names.
    """
    for key in candidate_keys:
        if key in payload:
            value = safe_float(payload.get(key))
            if value is not None:
                return value

    # Try one level nested dictionaries
    for _, value in payload.items():
        if isinstance(value, dict):
            for key in candidate_keys:
                if key in value:
                    nested_value = safe_float(value.get(key))
                    if nested_value is not None:
                        return nested_value

    return None


def summarise_runs(run_json_paths: List[Path], output_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []

    for path in sorted(run_json_paths):
        payload = read_json(path)

        metrics = payload.get("metrics", {})
        row = {
            "run_file": path.name,
            "experiment_type": payload.get("experiment_type"),
            "run_id": payload.get("run_id"),
            "dataset": payload.get("dataset"),
            "model": payload.get("model"),
            "seed": payload.get("seed"),
            "ablation_name": payload.get("ablation_name"),
            "ablation_value": payload.get("ablation_value"),
            "command": payload.get("command"),
            "wall_time_min": safe_float(payload.get("wall_time_min")),
            "test_accuracy": extract_metric(metrics, ["test_accuracy", "accuracy", "acc", "test_acc"]),
            "macro_f1": extract_metric(metrics, ["macro_f1", "f1_macro"]),
            "weighted_f1": extract_metric(metrics, ["weighted_f1", "f1_weighted"]),
            "balanced_accuracy": extract_metric(metrics, ["balanced_accuracy", "bal_acc"]),
            "auroc": extract_metric(metrics, ["auroc", "roc_auc"]),
            "auprc": extract_metric(metrics, ["auprc", "pr_auc"]),
            "nll": extract_metric(metrics, ["nll", "neg_log_likelihood", "negative_log_likelihood"]),
            "brier": extract_metric(metrics, ["brier", "brier_score"]),
            "ece": extract_metric(metrics, ["ece"]),
            "epoch_time_min": extract_metric(metrics, ["epoch_time_min", "mean_epoch_time_min", "epoch_time"]),
            "tokens_per_sec": extract_metric(metrics, ["tokens_per_sec", "tok_per_sec"]),
            "gpu_memory_mb": extract_metric(metrics, ["gpu_memory_mb", "max_gpu_memory_mb", "gpu_mem_mb"]),
            "param_count": extract_metric(metrics, ["param_count", "params", "num_parameters"]),
        }
        rows.append(row)

    all_runs_csv = output_dir / "all_runs.csv"
    write_csv(all_runs_csv, rows)

    summary_rows = build_summary_rows(rows)
    summary_csv = output_dir / "summary_table.csv"
    write_csv(summary_csv, summary_rows)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate across seeds.
    Grouping:
      - models run: dataset + model
      - ablation run: dataset + model + ablation_name + ablation_value
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        key_parts = [
            str(row.get("dataset")),
            str(row.get("model")),
            str(row.get("ablation_name")),
            str(row.get("ablation_value")),
        ]
        key = "||".join(key_parts)
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []

    for _, group_rows in grouped.items():
        first = group_rows[0]

        accs = [safe_float(r.get("test_accuracy")) for r in group_rows]
        accs = [x for x in accs if x is not None]

        wall_times = [safe_float(r.get("wall_time_min")) for r in group_rows]
        wall_times = [x for x in wall_times if x is not None]

        epoch_times = [safe_float(r.get("epoch_time_min")) for r in group_rows]
        epoch_times = [x for x in epoch_times if x is not None]

        summary_rows.append(
            {
                "dataset": first.get("dataset"),
                "model": first.get("model"),
                "ablation_name": first.get("ablation_name"),
                "ablation_value": first.get("ablation_value"),
                "n_runs": len(group_rows),
                "mean_test_accuracy": mean_or_none(accs),
                "std_test_accuracy": std_or_none(accs),
                "min_test_accuracy": min(accs) if accs else None,
                "max_test_accuracy": max(accs) if accs else None,
                "mean_epoch_time_min": mean_or_none(epoch_times),
                "mean_wall_time_min": mean_or_none(wall_times),
            }
        )

    # Stable ordering
    summary_rows.sort(
        key=lambda r: (
            str(r.get("dataset")),
            str(r.get("model")),
            str(r.get("ablation_name")),
            float(r.get("ablation_value")) if is_number_like(r.get("ablation_value")) else str(r.get("ablation_value")),
        )
    )
    return summary_rows


def mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.mean(values)


def std_or_none(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    return statistics.stdev(values)


def is_number_like(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
