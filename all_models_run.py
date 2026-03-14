import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON_BIN = sys.executable

OUTPUT_DIR = Path("models_runs")
OUTPUT_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 999]

DATASET_CONFIGS = {
    "imdb": {
        "args": [
            "--epochs", "3",
            "--batch_size", "32",
            "--max_len", "512",
            "--val_ratio", "0.1",
            "--d_model", "128",
            "--lr", "3e-4",
            "--output_dir", "results",
        ],
    },
    "listops_synth": {
        "args": [
            "--epochs", "3",
            "--batch_size", "64",
            "--max_len", "512",
            "--val_ratio", "0.1",
            "--d_model", "128",
            "--lr", "3e-4",
            "--output_dir", "results",
        ],
    },
}

MODEL_CONFIGS = {
    "meanpool": {
        "args": [],
    },
    "bilstm": {
        "args": [],
    },
    "tiny_transformer": {
        "args": [],
    },
    "transformer_base": {
        "args": [],
    },
    "tmr": {
        "args": [
            "--tmr_steps", "2",
            "--tmr_slots", "16",
            "--tmr_decay", "0.9",
            "--tmr_topk", "0",
            "--tmr_dropout", "0.1",
            "--tmr_score_clip", "20.0",
        ],
    },
}


def build_command(dataset, model, seed):
    cmd = [
        PYTHON_BIN,
        "main.py",
        "--dataset", dataset,
        "--model", model,
        "--seed", str(seed),
    ]
    cmd.extend(DATASET_CONFIGS[dataset]["args"])
    cmd.extend(MODEL_CONFIGS[model]["args"])
    return cmd


def run_one(run_idx, total_runs, dataset, model, seed):
    cmd = build_command(dataset, model, seed)
    cmd_str = " ".join(cmd)

    print("\n" + "=" * 90)
    print(f"RUN {run_idx}/{total_runs}")
    print(cmd_str)
    print("=" * 90)

    start = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    captured_lines = []

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        captured_lines.append(line)

    process.wait()

    wall_time_min = (time.time() - start) / 60.0

    record = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "command": cmd_str,
        "return_code": process.returncode,
        "wall_time_min": wall_time_min,
        "stdout": "".join(captured_lines),
    }

    out_path = OUTPUT_DIR / f"{dataset}__{model}__seed{seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print("-" * 90)
    print(f"Finished run {run_idx}/{total_runs} | return_code={process.returncode}")
    print(f"Saved raw log to: {out_path}")
    print("-" * 90)


def main():
    combos = [
        (dataset, model, seed)
        for dataset in DATASET_CONFIGS
        for model in MODEL_CONFIGS
        for seed in SEEDS
    ]

    total_runs = len(combos)
    print(f"Total runs: {total_runs}")

    for i, (dataset, model, seed) in enumerate(combos, start=1):
        run_one(i, total_runs, dataset, model, seed)

    print("\nAll runs completed.")
    print(f"Logs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
