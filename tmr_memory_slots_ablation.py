import subprocess
import itertools

datasets = ["imdb", "listops_synth"]
seeds = [42, 123, 999]
mem_slots = [16, 32, 64, 128, 256]

steps = 4

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
            "--mem_slots", str(slots),
            "--steps", str(steps),
        ]

    elif dataset == "listops_synth":
        cmd = [
            "python",
            "main.py",
            "--dataset", "listops_synth",
            "--model", "tmr",
            "--epochs", "3",
            "--batch_size", "64",
            "--max_len", "512",
            "--seed", str(seed),
            "--mem_slots", str(slots),
            "--steps", str(steps),
        ]

    subprocess.run(cmd, check=True)
