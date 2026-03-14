import csv
import json
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


def normalise_key(key: Any) -> str:
    if key is None:
        return ""
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def extract_metric(payload: Any, candidate_keys: Iterable[str]) -> Optional[float]:
    """
    Recursively search a nested dict/list structure for any matching metric key.
    Returns the first numeric value found.
    """
    normalised_candidates = {normalise_key(k) for k in candidate_keys}

    def _search(obj: Any) -> Optional[float]:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if normalise_key(key) in normalised_candidates:
                    v = safe_float(value)
                    if v is not None:
                        return v
                found = _search(value)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _search(item)
                if found is not None:
                    return found
        return None

    return _search(payload)


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
            "status": payload.get("status"),
            "returncode": payload.get("returncode"),
            "wall_time_min": safe_float(payload.get("wall_time_min")),
            "test_accuracy": extract_metric(
                metrics,
                [
                    "test_accuracy",
                    "accuracy",
                    "acc",
                    "test_acc",
                    "test/accuracy",
                    "test_accuracy_mean",
                ],
            ),
            "macro_f1": extract_metric(
                metrics,
                [
                    "macro_f1",
                    "f1_macro",
                    "macro_f1_score",
                    "test_macro_f1",
                ],
            ),
            "weighted_f1": extract_metric(
                metrics,
                [
                    "weighted_f1",
                    "f1_weighted",
                    "weighted_f1_score",
                    "test_weighted_f1",
                ],
            ),
            "balanced_accuracy": extract_metric(
                metrics,
                [
                    "balanced_accuracy",
                    "balanced_acc",
                    "bal_acc",
                    "test_balanced_accuracy",
                ],
            ),
            "auroc": extract_metric(
                metrics,
                [
                    "auroc",
                    "roc_auc",
                    "auc_roc",
                    "test_auroc",
                ],
            ),
            "auprc": extract_metric(
                metrics,
                [
                    "auprc",
                    "pr_auc",
                    "auc_pr",
                    "test_auprc",
                ],
            ),
            "nll": extract_metric(
                metrics,
                [
                    "nll",
                    "neg_log_likelihood",
                    "negative_log_likelihood",
                    "test_nll",
                ],
            ),
            "brier": extract_metric(
                metrics,
                [
                    "brier",
                    "brier_score",
                    "test_brier",
                ],
            ),
            "ece": extract_metric(
                metrics,
                [
                    "ece",
                    "expected_calibration_error",
                    "test_ece",
                ],
            ),
            "epoch_time_min": extract_metric(
                metrics,
                [
                    "epoch_time_min",
                    "mean_epoch_time_min",
                    "epoch_time",
                    "avg_epoch_time_min",
                ],
            ),
            "tokens_per_sec": extract_metric(
                metrics,
                [
                    "tokens_per_sec",
                    "tok_per_sec",
                    "tokens_sec",
                ],
            ),
            "gpu_memory_mb": extract_metric(
                metrics,
                [
                    "gpu_memory_mb",
                    "max_gpu_memory_mb",
                    "gpu_mem_mb",
                    "gpu_memory",
                ],
            ),
            "param_count": extract_metric(
                metrics,
                [
                    "param_count",
                    "params",
                    "num_parameters",
                    "parameter_count",
                ],
            ),
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

        macro_f1s = [safe_float(r.get("macro_f1")) for r in group_rows]
        macro_f1s = [x for x in macro_f1s if x is not None]

        weighted_f1s = [safe_float(r.get("weighted_f1")) for r in group_rows]
        weighted_f1s = [x for x in weighted_f1s if x is not None]

        balanced_accs = [safe_float(r.get("balanced_accuracy")) for r in group_rows]
        balanced_accs = [x for x in balanced_accs if x is not None]

        aurocs = [safe_float(r.get("auroc")) for r in group_rows]
        aurocs = [x for x in aurocs if x is not None]

        auprcs = [safe_float(r.get("auprc")) for r in group_rows]
        auprcs = [x for x in auprcs if x is not None]

        nlls = [safe_float(r.get("nll")) for r in group_rows]
        nlls = [x for x in nlls if x is not None]

        briers = [safe_float(r.get("brier")) for r in group_rows]
        briers = [x for x in briers if x is not None]

        eces = [safe_float(r.get("ece")) for r in group_rows]
        eces = [x for x in eces if x is not None]

        wall_times = [safe_float(r.get("wall_time_min")) for r in group_rows]
        wall_times = [x for x in wall_times if x is not None]

        epoch_times = [safe_float(r.get("epoch_time_min")) for r in group_rows]
        epoch_times = [x for x in epoch_times if x is not None]

        tokens_per_sec = [safe_float(r.get("tokens_per_sec")) for r in group_rows]
        tokens_per_sec = [x for x in tokens_per_sec if x is not None]

        gpu_memory_mb = [safe_float(r.get("gpu_memory_mb")) for r in group_rows]
        gpu_memory_mb = [x for x in gpu_memory_mb if x is not None]

        param_counts = [safe_float(r.get("param_count")) for r in group_rows]
        param_counts = [x for x in param_counts if x is not None]

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
                "mean_macro_f1": mean_or_none(macro_f1s),
                "mean_weighted_f1": mean_or_none(weighted_f1s),
                "mean_balanced_accuracy": mean_or_none(balanced_accs),
                "mean_auroc": mean_or_none(aurocs),
                "mean_auprc": mean_or_none(auprcs),
                "mean_nll": mean_or_none(nlls),
                "mean_brier": mean_or_none(briers),
                "mean_ece": mean_or_none(eces),
                "mean_epoch_time_min": mean_or_none(epoch_times),
                "mean_wall_time_min": mean_or_none(wall_times),
                "mean_tokens_per_sec": mean_or_none(tokens_per_sec),
                "mean_gpu_memory_mb": mean_or_none(gpu_memory_mb),
                "mean_param_count": mean_or_none(param_counts),
            }
        )

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
