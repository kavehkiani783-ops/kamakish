import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all baseline models across datasets and seeds.")
    parser.add_argument("--python_exec", type=str, default=sys.executable, help="Python executable to use")
    parser.add_argument("--main_py", type=str, default="main.py", help="Path to main.py")
    parser.add_argument("--root_dir", type=str, default="models_runs", help="Root directory for model comparison runs")
    parser.add_argument("--results_dir", type=str, default="results", help="Temporary results directory used by main.py")
    parser.add_argument("--clean_results_dir", action="store_true", help="Clear temporary results dir before starting")
    parser.add_argument("--tag", type=str, default="", help="Optional suffix for run folder name")
    return parser


def model_grid() -> List[Dict]:
    seeds = [42, 123, 999]
    models = ["meanpool", "bilstm", "tiny_transformer", "transformer_base", "tmr"]

    runs: List[Dict] = []

    # IMDB
    for seed in seeds:
        for model in models:
            cmd = [
                "--dataset", "imdb",
                "--model", model,
                "--epochs", "3",
                "--val_ratio", "0.1",
                "--seed", str(seed),
            ]
            runs.append(
                {
                    "dataset": "imdb",
                    "model": model,
                    "seed": seed,
                    "args": cmd,
                }
            )

    # ListOps synthetic
    for seed in seeds:
        for model in models:
            cmd = [
                "--dataset", "listops_synth",
                "--model", model,
                "--epochs", "3",
                "--batch_size", "64",
                "--max_len", "512",
                "--seed", str(seed),
            ]
            runs.append(
                {
                    "dataset": "listops_synth",
                    "model": model,
                    "seed": seed,
                    "args": cmd,
                }
            )

    return runs


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    results_dir = Path(args.results_dir)

    run_id = f"run_{timestamp_now()}"
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

    all_jobs = model_grid()
    saved_run_paths: List[Path] = []

    print(f"Run directory: {run_dir}")
    print(f"Total runs: {len(all_jobs)}")
    print("-" * 80)

    for idx, job in enumerate(all_jobs, start=1):
        dataset = job["dataset"]
        model = job["model"]
        seed = job["seed"]

        command = [args.python_exec, args.main_py] + job["args"]
        command_str = " ".join(shlex.quote(part) for part in command)

        expected_result = expected_main_result_path(model, dataset, seed, results_dir)
        if expected_result.exists():
            expected_result.unlink()

        print(f"[{idx}/{len(all_jobs)}] Running: {command_str}")

        start = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        wall_time_min = (time.perf_counter() - start) / 60.0

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if completed.returncode != 0:
            payload = {
                "experiment_type": "models",
                "run_id": run_id,
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "command": command_str,
                "wall_time_min": wall_time_min,
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "metrics": {},
                "status": "failed",
            }
            out_path = runs_dir / f"{model}_{dataset}_seed{seed}.json"
            write_json(out_path, payload)
            saved_run_paths.append(out_path)

            print(f"  FAILED (return code {completed.returncode})")
            print("-" * 80)
            continue

        if not expected_result.exists():
            payload = {
                "experiment_type": "models",
                "run_id": run_id,
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "command": command_str,
                "wall_time_min": wall_time_min,
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "metrics": {},
                "status": "failed_missing_result_json",
            }
            out_path = runs_dir / f"{model}_{dataset}_seed{seed}.json"
            write_json(out_path, payload)
            saved_run_paths.append(out_path)

            print("  FAILED (main.py finished but result JSON not found)")
            print("-" * 80)
            continue

        result_metrics = read_json(expected_result)

        payload = {
            "experiment_type": "models",
            "run_id": run_id,
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "command": command_str,
            "wall_time_min": wall_time_min,
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "metrics": result_metrics,
            "status": "ok",
        }

        out_path = runs_dir / f"{model}_{dataset}_seed{seed}.json"
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
