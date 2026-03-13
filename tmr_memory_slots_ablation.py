import subprocess
import itertools
import os

datasets = ["imdb", "listops_synth"]
seeds = [42, 123, 999]
mem_slots = [16, 32, 64, 128, 256]

steps = 4

# Create dedicated output folder for this experiment
output_dir = "ablation_runs/memory_slots_ablation"
os.makedirs(output_dir, exist_ok=True)

for dataset, slots, seed in itertools.product(datasets, mem_slots, seeds):

    print("------------------------------------------------------------")
    print(f"Running dataset={dataset} | mem_slots={slots} | seed={seed}")
    print("------------------------------------------------------------")

    if dataset == "imdb":
        cmd = [
            "python",
            "main.py",
            "--dataset", "imdb",
            "--model", "tmr",
            "--epochs", "3",
            "--val_ratio", "0.1",
            "--seed", str(seed),
            "--tmr_slots", str(slots),
            "--tmr_steps", str(steps),
            "--output_dir", output_dir
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
            "--tmr_slots", str(slots),
            "--tmr_steps", str(steps),
            "--output_dir", output_dir
        ]

    subprocess.run(cmd, check=True)
