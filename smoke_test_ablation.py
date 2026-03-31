import subprocess
import glob
import os
import time

RESULTS_DIR = "results"


def get_latest_result(before_files):
    after_files = set(glob.glob(f"{RESULTS_DIR}/*.json"))
    new_files = list(after_files - before_files)
    if not new_files:
        return None
    new_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return new_files[0]


def run_one(cmd):
    before = set(glob.glob(f"{RESULTS_DIR}/*.json"))

    print("\nRunning:", " ".join(cmd))
    start = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start

    if res.returncode != 0:
        print("FAILED: main.py crashed")
        return False

    latest = get_latest_result(before)

    if latest is None:
        print("FAILED: no result JSON detected")
        return False

    print(f"OK: result captured -> {latest} | time={elapsed:.2f}s")

    # optional cleanup
    os.remove(latest)

    return True


def main():
    python_exec = "python"
    main_py = "main.py"

    base_cmd = [
        python_exec, main_py,
        "--dataset", "imdb",
        "--model", "hubnet_v2",
        "--epochs", "1",
        "--seed", "42",
        "--batch_size", "8",
        "--max_len", "128",
        "--val_ratio", "0.1",
        "--d_model", "64",
        "--HubNet_decay", "0.9",
        "--HubNet_dropout", "0.1",
        "--HubNet_score_clip", "20.0",
        "--HubNet_gate"
    ]

    print("\n=== SMOKE TEST: STEPS ===")
    run_one(base_cmd + [
        "--HubNet_steps", "2",
        "--HubNet_slots", "32",
        "--HubNet_topk", "0"
    ])

    print("\n=== SMOKE TEST: SLOTS ===")
    run_one(base_cmd + [
        "--HubNet_steps", "4",
        "--HubNet_slots", "16",
        "--HubNet_topk", "0"
    ])

    print("\n=== SMOKE TEST: TOPK ===")
    run_one(base_cmd + [
        "--HubNet_steps", "4",
        "--HubNet_slots", "32",
        "--HubNet_topk", "2"
    ])

    print("\nSmoke test finished.")


if __name__ == "__main__":
    main()
