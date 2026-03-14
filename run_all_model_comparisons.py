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
        "args": ["--epochs", "3", "--val_ratio", "0.1"],
    },
    "listops_synth": {
        "args": ["--epochs", "3", "--batch_size", "64", "--max_len", "512"],
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
        "args": [],
        # If you want fixed TMR settings for the comparison, use for example:
        # "args": ["--tmr_steps", "2", "--tmr_slots", "16"]
    },
}


def build_command(dataset_name, model_name, seed):
    cmd = [
        PYTHON_BIN,
        "main.py",
        "--dataset", dataset_name,
        "--model", model_name,
        "--seed", str(seed),
    ]
    cmd.extend(DATASET_CONFIGS[dataset_name]["args"])
    cmd.extend(MODEL_CONFIGS[model_name]["args"])
    return cmd


def run_one(dataset_name, model_name, seed):
    cmd = build_command(dataset_name, model_name, seed)
    cmd_str = " ".join(cmd)
    json_name = f"{dataset_name}__{model_name}__seed{seed}.json"
    json_path = OUTPUT_DIR / json_name

    print(f"\n[RUN] {cmd_str}", flush=True)

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    end = time.perf_counter()

    wall_time_min = (end - start) / 60.0

    record = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "command": cmd_str,
        "return_code": proc.returncode,
        "wall_time_min": wall_time_min,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    if proc.returncode == 0:
        print(f"[OK] saved -> {json_path}")
    else:
        print(f"[FAIL] return_code={proc.returncode} -> {json_path}")


def main():
    total = len(DATASET_CONFIGS) * len(MODEL_CONFIGS) * len(SEEDS)
    print(f"Total runs: {total}")

    for dataset_name in DATASET_CONFIGS:
        for model_name in MODEL_CONFIGS:
            for seed in SEEDS:
                run_one(dataset_name, model_name, seed)

    print("\nAll runs finished.")
    print(f"JSON outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
