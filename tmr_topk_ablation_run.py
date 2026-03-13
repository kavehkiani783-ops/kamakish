import os
import subprocess
import itertools

# -------------------------------------------------------
# Experiment configuration
# -------------------------------------------------------

DATASETS = ["imdb", "listops_synth"]
SEEDS = [42, 123, 999]
TOPK_VALUES = [0, 2, 4, 8, 16]

TMR_STEPS = 4
TMR_SLOTS = 16

BASE_OUTPUT_DIR = "ablation_runs/topk_ablation"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

print("\nStarting TMR top-k binding ablation runs\n")

# -------------------------------------------------------
# Run experiments
# -------------------------------------------------------

for dataset, topk, seed in itertools.product(DATASETS, TOPK_VALUES, SEEDS):

    print("------------------------------------------------------------")
    print(f"Running dataset={dataset} | topk={topk} | seed={seed}")
    print("------------------------------------------------------------")

    run_output_dir = os.path.join(
        BASE_OUTPUT_DIR,
        f"{dataset}_topk{topk}_seed{seed}"
    )

    os.makedirs(run_output_dir, exist_ok=True)

    # Skip if result already exists
    existing_json = [f for f in os.listdir(run_output_dir) if f.endswith(".json")]
    if len(existing_json) > 0:
        print("Run already exists, skipping.")
        continue

    if dataset == "imdb":

        cmd = [
            "python",
            "main.py",
            "--dataset", "imdb",
            "--model", "tmr",
            "--epochs", "3",
            "--val_ratio", "0.1",
            "--seed", str(seed),
            "--tmr_slots", str(TMR_SLOTS),
            "--tmr_steps", str(TMR_STEPS),
            "--tmr_topk", str(topk),
            "--output_dir", run_output_dir
        ]

    else:

        cmd = [
            "python",
            "main.py",
            "--dataset", "listops_synth",
            "--model", "tmr",
            "--epochs", "3",
            "--batch_size", "64",
            "--max_len", "512",
            "--seed", str(seed),
            "--tmr_slots", str(TMR_SLOTS),
            "--tmr_steps", str(TMR_STEPS),
            "--tmr_topk", str(topk),
            "--output_dir", run_output_dir
        ]

    subprocess.run(cmd, check=True)

print("\nTop-k ablation runs finished.")
