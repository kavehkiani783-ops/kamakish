import subprocess
import os
import time
import json
import glob
from pathlib import Path

RESULTS_DIR = Path("results")


def clear_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for f in RESULTS_DIR.glob("*.json"):
        f.unlink(missing_ok=True)


def expected_result_path(dataset, seed, steps, slots, topk, gate=True):
    fname = (
        f"hubnet_v2_{dataset}_seed{seed}"
        f"_steps{steps}_slots{slots}_topk{topk}_gate{int(gate)}.json"
    )
    return RESULTS_DIR / fname


def has_metrics(payload):
    if not isinstance(payload, dict):
        return False

    candidate_keys = {
        "val_accuracy",
        "best_val_acc",
        "best_val_accuracy",
        "val_acc",
        "test_accuracy",
        "test_acc",
        "accuracy",
        "epoch_time",
        "epoch_time_min",
        "avg_epoch_time",
        "avg_epoch_time_min",
    }

    def search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in candidate_keys and isinstance(v, (int, float)):
                    return True
                if search(v):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if search(item):
                    return True
        return False

    return search(payload)


def run_one(label, cmd, expected_json):
    clear_results_dir()

    print(f"\n=== SMOKE TEST: {label} ===")
    print("Running:", " ".join(cmd))

    start = time.time()
    res = subprocess.run(cmd, text=True)
    elapsed = time.time() - start

    if res.returncode != 0:
        print("FAILED: main.py crashed")
        return False

    if not expected_json.exists():
        print(f"FAILED: expected JSON not found -> {expected_json}")
        available = sorted(glob.glob(str(RESULTS_DIR / "*.json")))
        if available:
            print("Available JSON files:")
            for f in available:
                print(" ", f)
        return False

    try:
        with expected_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"FAILED: could not read JSON -> {e}")
        return False

    if not has_metrics(payload):
        print(f"FAILED: JSON exists but no metrics found -> {expected_json}")
        return False

    print(f"OK: result captured -> {expected_json} | time={elapsed:.2f}s")
    return True


def main():
    python_exec = "python"
    main_py = "main.py"

    dataset = "imdb"
    seed = 42

    base_cmd = [
        python_exec, main_py,
        "--dataset", dataset,
        "--model", "hubnet_v2",
        "--epochs", "1",
        "--seed", str(seed),
        "--batch_size", "8",
        "--max_len", "128",
        "--val_ratio", "0.1",
        "--d_model", "64",
        "--HubNet_decay", "0.9",
        "--HubNet_dropout", "0.1",
        "--HubNet_score_clip", "20.0",
        "--HubNet_gate",
    ]

    tests = [
        {
            "label": "STEPS",
            "steps": 2,
            "slots": 32,
            "topk": 0,
        },
        {
            "label": "SLOTS",
            "steps": 4,
            "slots": 16,
            "topk": 0,
        },
        {
            "label": "TOPK",
            "steps": 4,
            "slots": 32,
            "topk": 2,
        },
    ]

    all_ok = True

    for t in tests:
        cmd = base_cmd + [
            "--HubNet_steps", str(t["steps"]),
            "--HubNet_slots", str(t["slots"]),
            "--HubNet_topk", str(t["topk"]),
        ]

        expected_json = expected_result_path(
            dataset=dataset,
            seed=seed,
            steps=t["steps"],
            slots=t["slots"],
            topk=t["topk"],
            gate=True,
        )

        ok = run_one(
            label=t["label"],
            cmd=cmd,
            expected_json=expected_json,
        )
        all_ok = all_ok and ok

    print("\nSmoke test finished.")
    print("FINAL STATUS:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
