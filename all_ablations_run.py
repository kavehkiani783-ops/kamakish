import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

from experiment_common import (
    clear_directory_contents,
    ensure_dir,
    expected_main_result_path,
    read_json,
    remove_dir_if_empty,
    summarise_runs,
    timestamp_now,
    write_json,
)

# ============================================================
# CENTRAL ABLATION CONFIG
# Everything important is defined here in code.
# No manual dataset/value selection from terminal.
# ============================================================

ABLATION_DATASETS = ["imdb", "listops_synth"]
ABLATION_SEEDS = [42, 123, 999]

# Shared defaults per dataset
DATASET_CONFIGS = {
    "imdb": {
        "epochs": 3,
        "batch_size": 32,
        "max_len": 512,
        "val_ratio": 0.1,
        "d_model": 128,
    },
    "listops_synth": {
        "epochs": 3,
        "batch_size": 64,
        "max_len": 512,
        "val_ratio": 0.1,
        "d_model": 128,
    },
}

# Default TMR config used as the base before sweeping one parameter
TMR_BASE_CONFIG = {
    "tmr_steps": 2,
    "tmr_slots": 16,
    "tmr_decay": 0.9,
    "tmr_topk": 0,
    "tmr_dropout": 0.1,
    "tmr_score_clip": 20.0,
    "tmr_gate": False,
}

# Sweep values are defined here in code
ABLATION_SWEEPS = {
    "steps": [0, 1, 2, 4, 8],
    "slots": [16, 32, 64, 128, 256],
    "topk": [0, 2, 4, 8, 16],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TMR ablation sweeps across both datasets and store everything in one folder."
    )
    parser.add_argument("--python_exec", type=str, default=sys.executable, help="Python executable to use")
    parser.add_argument("--main_py", type=str, default="main.py", help="Path to main.py")
    parser.add_argument("--root_dir", type=str, default="ablation_runs", help="Root directory for ablation runs")
    parser.add_argument("--results_dir", type=str, default="results", help="Temporary results directory used by main.py")
    parser.add_argument("--clean_results_dir", action="store_true", help="Clear temporary results dir before starting")
    parser.add_argument("--tag", type=str, default="", help="Optional suffix for run folder name")

    # Only ablation type is selected manually
    parser.add_argument("--ablation", type=str, required=True, choices=["steps", "slots", "topk"])

    return parser


def replace_arg_value(cmd: List[str], arg_name: str, new_value: str) -> None:
    idx = cmd.index(arg_name)
    cmd[idx + 1] = new_value


def build_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    sweep_values = ABLATION_SWEEPS[args.ablation]
    jobs: List[Dict[str, Any]] = []

    for dataset in ABLATION_DATASETS:
        dataset_cfg = DATASET_CONFIGS[dataset]

        for ablation_value in sweep_values:
            for seed in ABLATION_SEEDS:
                cmd = [
                    "--dataset", dataset,
                    "--model", "tmr",
                    "--epochs", str(dataset_cfg["epochs"]),
                    "--seed", str(seed),
                    "--batch_size", str(dataset_cfg["batch_size"]),
                    "--max_len", str(dataset_cfg["max_len"]),
                    "--val_ratio", str(dataset_cfg["val_ratio"]),
                    "--d_model", str(dataset_cfg["d_model"]),
                    "--tmr_steps", str(TMR_BASE_CONFIG["tmr_steps"]),
                    "--tmr_slots", str(TMR_BASE_CONFIG["tmr_slots"]),
                    "--tmr_decay", str(TMR_BASE_CONFIG["tmr_decay"]),
                    "--tmr_topk", str(TMR_BASE_CONFIG["tmr_topk"]),
                    "--tmr_dropout", str(TMR_BASE_CONFIG["tmr_dropout"]),
                    "--tmr_score_clip", str(TMR_BASE_CONFIG["tmr_score_clip"]),
                ]

                if TMR_BASE_CONFIG["tmr_gate"]:
                    cmd.append("--tmr_gate")

                if args.ablation == "steps":
                    replace_arg_value(cmd, "--tmr_steps", str(ablation_value))
                elif args.ablation == "slots":
                    replace_arg_value(cmd, "--tmr_slots", str(ablation_value))
                elif args.ablation == "topk":
                    replace_arg_value(cmd, "--tmr_topk", str(ablation_value))
                else:
                    raise ValueError(f"Unsupported ablation type: {args.ablation}")

                jobs.append(
                    {
                        "dataset": dataset,
                        "model": "tmr",
                        "seed": seed,
                        "ablation_name": args.ablation,
                        "ablation_value": ablation_value,
                        "args": cmd,
                    }
                )

    return jobs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    results_dir = Path(args.results_dir)

    run_id = f"run_{timestamp_now()}_{args.ablation}"
    if args.tag.strip():
        safe_tag = args.tag.strip().replace(" ", "_")
        run_id = f"{run_id}_{safe_tag}"

    run_dir = root_dir / run_id
    runs_dir = run_dir / "runs"

    ensure_dir(run_dir)
    ensure_dir(runs_dir)
    ensure_dir(results_dir)

    if args.clean_results_dir:
        clear_directory_contents(results_dir)

    jobs = build_jobs(args)
    saved_run_paths: List[Path] = []

    print(f"Run directory: {run_dir}")
    print(f"Datasets (from code): {ABLATION_DATASETS}")
    print(f"Ablation type: {args.ablation}")
    print(f"Sweep values (from code): {ABLATION_SWEEPS[args.ablation]}")
    print(f"Seeds (from code): {ABLATION_SEEDS}")
    print(f"Total runs: {len(jobs)}")
    print("-" * 80)

    for idx, job in enumerate(jobs, start=1):
        dataset = job["dataset"]
        model = job["model"]
        seed = job["seed"]
        ablation_name = job["ablation_name"]
        ablation_value = job["ablation_value"]

        command = [args.python_exec, args.main_py] + job["args"]
        command_str = " ".join(shlex.quote(part) for part in command)

        expected_result = expected_main_result_path(model, dataset, seed, results_dir)
        if expected_result.exists():
            expected_result.unlink()

        print(f"[{idx}/{len(jobs)}] Running: {command_str}")

        start = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        wall_time_min = (time.perf_counter() - start) / 60.0

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        out_name = f"tmr_{dataset}_{ablation_name}{ablation_value}_seed{seed}.json"
        out_path = runs_dir / out_name

        if completed.returncode != 0:
            payload = {
                "experiment_type": "ablation",
                "run_id": run_id,
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "ablation_name": ablation_name,
                "ablation_value": ablation_value,
                "command": command_str,
                "wall_time_min": wall_time_min,
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "metrics": {},
                "status": "failed",
            }
            write_json(out_path, payload)
            saved_run_paths.append(out_path)

            print(f"  FAILED (return code {completed.returncode})")
            print("-" * 80)
            continue

        if not expected_result.exists():
            payload = {
                "experiment_type": "ablation",
                "run_id": run_id,
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "ablation_name": ablation_name,
                "ablation_value": ablation_value,
                "command": command_str,
                "wall_time_min": wall_time_min,
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "metrics": {},
                "status": "failed_missing_result_json",
            }
            write_json(out_path, payload)
            saved_run_paths.append(out_path)

            print("  FAILED (main.py finished but result JSON not found)")
            print("-" * 80)
            continue

        result_metrics = read_json(expected_result)

        payload = {
            "experiment_type": "ablation",
            "run_id": run_id,
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "ablation_name": ablation_name,
            "ablation_value": ablation_value,
            "command": command_str,
            "wall_time_min": wall_time_min,
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "metrics": result_metrics,
            "status": "ok",
        }
        write_json(out_path, payload)
        saved_run_paths.append(out_path)

        expected_result.unlink(missing_ok=True)

        print(f"  OK | wall_time={wall_time_min:.3f} min | saved={out_path.name}")
        print("-" * 80)

    summarise_runs(saved_run_paths, run_dir)

    remove_dir_if_empty(results_dir)

    print("Finished.")
    print(f"Run folder: {run_dir}")
    print(f"Runs JSON folder: {runs_dir}")
    print(f"CSV files: {run_dir / 'all_runs.csv'} , {run_dir / 'summary_table.csv'}")


if __name__ == "__main__":
    main()
