import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def find_json_file(run_dir: Path) -> Optional[Path]:
    json_files = list(run_dir.glob("*.json"))
    if not json_files:
        return None
    # Prefer tmr_*.json if present
    preferred = [p for p in json_files if p.name.startswith("tmr_")]
    return preferred[0] if preferred else json_files[0]


def parse_run_dir_name(run_dir_name: str) -> Optional[Dict[str, Any]]:
    """
    Expected pattern from your script:
    {dataset}_seed{seed}_steps{steps}_slots{mem_slots}_decay{decay}_gate{0/1}_topk{topk}

    Example:
    imdb_seed42_steps4_slots64_decay0.9_gate0_topk0
    listops_synth_seed123_steps2_slots64_decay0.9_gate0_topk0
    """
    parts = run_dir_name.split("_")

    seed_idx = None
    for i, p in enumerate(parts):
        if p.startswith("seed"):
            seed_idx = i
            break

    if seed_idx is None:
        return None

    dataset = "_".join(parts[:seed_idx])

    try:
        seed = int(parts[seed_idx].replace("seed", ""))
        steps = int(parts[seed_idx + 1].replace("steps", ""))
        mem_slots = int(parts[seed_idx + 2].replace("slots", ""))
        decay = float(parts[seed_idx + 3].replace("decay", ""))
        gate = bool(int(parts[seed_idx + 4].replace("gate", "")))
        topk = int(parts[seed_idx + 5].replace("topk", ""))
    except Exception:
        return None

    return {
        "dataset": dataset,
        "seed": seed,
        "steps": steps,
        "mem_slots": mem_slots,
        "decay": decay,
        "gate": gate,
        "topk": topk,
    }


def load_metrics_from_json(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "test_acc": safe_float(
            data.get("test_acc", data.get("accuracy", data.get("acc", data.get("test_accuracy"))))
        ),
        "macro_f1": safe_float(data.get("macro_f1")),
        "weighted_f1": safe_float(data.get("weighted_f1")),
        "balanced_accuracy": safe_float(data.get("balanced_accuracy")),
        "auroc": safe_float(data.get("auroc")),
        "auprc": safe_float(data.get("auprc")),
        "nll": safe_float(data.get("nll")),
        "brier": safe_float(data.get("brier")),
        "ece": safe_float(data.get("ece")),
        "wall_time": safe_float(
            data.get("wall_time", data.get("wall_time_min", data.get("time", data.get("epoch_time_min"))))
        ),
    }


def collect_runs(model_outputs_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if not model_outputs_dir.exists():
        raise FileNotFoundError(f"Directory not found: {model_outputs_dir}")

    for run_dir in sorted(model_outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        meta = parse_run_dir_name(run_dir.name)
        if meta is None:
            continue

        json_path = find_json_file(run_dir)
        if json_path is None:
            row = dict(meta)
            row.update({
                "test_acc": None,
                "macro_f1": None,
                "weighted_f1": None,
                "balanced_accuracy": None,
                "auroc": None,
                "auprc": None,
                "nll": None,
                "brier": None,
                "ece": None,
                "wall_time": None,
                "json_file": None,
            })
            rows.append(row)
            continue

        metrics = load_metrics_from_json(json_path)

        row = dict(meta)
        row.update(metrics)
        row["json_file"] = str(json_path)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid run folders found in: {model_outputs_dir}")

    df = pd.DataFrame(rows)

    sort_cols = [c for c in ["dataset", "steps", "seed"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def summarise_by_steps(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "test_acc",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "auroc",
        "auprc",
        "nll",
        "brier",
        "ece",
        "wall_time",
    ]

    summary_rows: List[Dict[str, Any]] = []

    for (dataset, steps), g in df.groupby(["dataset", "steps"], dropna=False):
        row: Dict[str, Any] = {
            "dataset": dataset,
            "steps": steps,
            "n": int(len(g)),
        }

        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"mean_{col}"] = float(vals.mean()) if len(vals) else None
            row[f"std_{col}"] = float(vals.std(ddof=0)) if len(vals) else None

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["dataset", "steps"]).reset_index(drop=True)
    return summary_df


def print_pretty(df: pd.DataFrame, title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 120,
        "display.float_format", lambda x: f"{x:.6f}" if isinstance(x, float) else str(x),
    ):
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Create all-runs and summary CSVs for TMR steps ablation.")
    parser.add_argument(
        "--model_outputs_dir",
        type=str,
        default="ablation_runs/model_outputs",
        help="Directory containing steps ablation run folders.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="ablation_runs/steps_ablation",
        help="Output directory for summary CSV files.",
    )
    parser.add_argument(
        "--all_runs_csv",
        type=str,
        default="steps_all_runs_comparison.csv",
        help="Filename for all-runs CSV.",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="steps_summary.csv",
        help="Filename for grouped summary CSV.",
    )
    args = parser.parse_args()

    model_outputs_dir = Path(args.model_outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_runs(model_outputs_dir)

    # Keep only rows that belong to the steps ablation.
    # If your model_outputs folder contains other ablations too, this filter helps:
    # fixed baseline config for steps ablation is usually slots=64, decay=0.9, gate=False, topk=0
    df_steps = df[
        (df["mem_slots"] == 64) &
        (df["decay"] == 0.9) &
        (df["gate"] == False) &
        (df["topk"] == 0)
    ].copy()

    if df_steps.empty:
        raise RuntimeError(
            "No runs matched the expected steps-ablation filter "
            "(mem_slots=64, decay=0.9, gate=False, topk=0). "
            "Adjust the filter in the script if your base config was different."
        )

    all_runs_csv_path = out_dir / args.all_runs_csv
    summary_csv_path = out_dir / args.summary_csv

    df_steps = df_steps[
        [
            "dataset",
            "steps",
            "seed",
            "test_acc",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "auroc",
            "auprc",
            "nll",
            "brier",
            "ece",
            "wall_time",
        ]
    ].sort_values(["dataset", "steps", "seed"]).reset_index(drop=True)

    summary_df = summarise_by_steps(df_steps)

    df_steps.to_csv(all_runs_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    print_pretty(df_steps, "Detected steps-ablation runs")
    print_pretty(summary_df, "Steps ablation summary")
    print(f"\nSaved all-runs table:\n{all_runs_csv_path}")
    print(f"\nSaved summary table:\n{summary_csv_path}")


if __name__ == "__main__":
    main()
