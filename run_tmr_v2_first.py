import subprocess

runs = [
    # NUBNET v1 ListOps
    [
        "python", "main.py",
        "--dataset", "listops_synth",
        "--model", "NubNet",
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--seed", "42",
        "--d_model", "128",
        "--NubNet_slots", "32",
        "--NubNet_steps", "1",
    ],

    # NUBNET v2 ListOps
    [
        "python", "main.py",
        "--dataset", "listops_synth",
        "--model", "NubNet_v2",
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--seed", "42",
        "--d_model", "128",
        "--NubNet_slots", "32",
        "--NubNet_steps", "1",
    ],

    # NUBNET v1 IMDB
    [
        "python", "main.py",
        "--dataset", "imdb",
        "--model", "NubNet",
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--seed", "42",
        "--d_model", "128",
        "--NubNet_slots", "32",
        "--NubNet_steps", "1",
    ],

    # NUBNET v2 IMDB
    [
        "python", "main.py",
        "--dataset", "imdb",
        "--model", "NubNet_v2",
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--seed", "42",
        "--d_model", "128",
        "--NubNet_slots", "32",
        "--NubNet_steps", "1",
    ],
]

for run in runs:
    print("\nRunning:", " ".join(run))
    subprocess.run(run, check=True)

print("\nAll runs completed.")
