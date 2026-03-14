import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# Fixed paths so the script can be run as:
# python steps_ablation_summary.py
BASE_DIR = Path("/home/ubuntu/taha")
ABLATION_ROOT = BASE_DIR / "ablation_runs"
MODEL_OUTPUTS_DIR = ABLATION_ROOT / "model_outputs"
STEPS_ABLATION_DIR = ABLATION_ROOT / "steps_ablation"

# Fixed baseline config for the steps ablation
BASE_MEM_SLOTS = 64
BASE_DECAY = 0.9
BASE_GATE = False
BASE_TOPK = 0


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


def safe_bool_from_int_string(x: str) -> bool:
    return bool(int(x))


def parse_run_dir_name(run_dir_name: str) -> Optional[Dict[str, Any]]:
    """
    Expected folder name pattern from tmr_steps_ablations.py:
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
        gate = safe_bool_from_int_string(parts[seed_idx + 4].replace("gate", ""))
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


def find_json_file(run_dir: Path) -> Optional[Path]:
    json_files = list(run_dir.glob("*.json"))
    if not json_files:
        return None

    preferred = [p for p in json_files if p.name.startswith("tmr_")]
    return preferred[0] if preferred else json_files[0]


def load_metrics_from_json(json_path: Path) -> Dict[str, Any]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {
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
        }

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


def collect_steps_runs_from_model_outputs() -> pd.DataFrame:
    if not MODEL_OUTPUTS_DIR.exists():
        raise FileNotFoundError(f"model_outputs directory not found: {MODEL_OUTPUTS_DIR}")

    rows: List[Dict[str, Any]] = []

    for run_dir in sorted(MODEL_OUTPUTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        meta = parse_run_dir_name(run_dir.name)
        if meta is None:
            continue

        # Keep only runs belonging to the steps ablation
        if not (
            meta["mem_slots"] == BASE_MEM_SLOTS
            and meta["decay"] == BASE_DECAY
            and meta["gate"] == BASE_GATE
            and meta["topk"] == BASE_TOPK
        ):
            continue

        json_path = find_json_file(run_dir)
        metrics = load_metrics_from_json(json_path) if json_path else {
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
        }

        row = dict(meta)
        row.update(metrics)
        row["source_json"] = str(json_path) if json_path else None
        row["source_run_dir"] = str(run_dir)
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No matching steps-ablation runs were found in model_outputs. "
            "That usually means the base filter does not match your actual steps-ablation runs."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "steps", "seed"]).reset_index(drop=True)
    return df


def copy_run_jsons_to_steps_folder(df: pd.DataFrame) -> None:
    STEPS_ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    copied_names = set()

    for _, row in df.iterrows():
        src_json = row.get("source_json")
        if not src_json:
            continue

        src_path = Path(src_json)
        if not src_path.exists():
            continue

        dataset = row["dataset"]
        seed = row["seed"]
        dst_name = f"tmr_{dataset}_{seed}.json"
        dst_path = STEPS_ABLATION_DIR / dst_name

        # Avoid overwriting repeatedly with the same source name if already copied once
        key = str(dst_path)
        if key in copied_names:
            continue

        shutil.copy2(src_path, dst_path)
        copied_names.add(key)


def summarise_steps(df: pd.DataFrame) -> pd.DataFrame:
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

    rows: List[Dict[str, Any]] = []

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

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["dataset", "steps"]).reset_index(drop=True)
    return out


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * 120)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 120,
        "display.float_format", lambda x: f"{x:.6f}",
    ):
        print(df.to_string(index=False))


def main() -> None:
    STEPS_ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    all_runs_csv = STEPS_ABLATION_DIR / "steps_all_runs_comparison.csv"
    summary_csv = STEPS_ABLATION_DIR / "steps_summary.csv"

    df = collect_steps_runs_from_model_outputs()

    # Copy JSON files into the steps_ablation folder for consistency with your other ablations
    copy_run_jsons_to_steps_folder(df)

    display_cols = [
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
    df_display = df[display_cols].copy()
    df_display.to_csv(all_runs_csv, index=False)

    summary_df = summarise_steps(df_display)
    summary_df.to_csv(summary_csv, index=False)

    print_table("Detected runs:", df_display)
    print(f"\nSaved all-runs table:\n{all_runs_csv}")

    for dataset in summary_df["dataset"].unique():
        ds_df = summary_df[summary_df["dataset"] == dataset].copy()
        print_table(f"Dataset: {dataset}", ds_df)

    print(f"\nSaved summary table:\n{summary_csv}")


if __name__ == "__main__":
    main()
