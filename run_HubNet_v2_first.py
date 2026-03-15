import subprocess

runs = [
    # HUBNET v1 ListOps
    [
        "python", "main.py",
        "--dataset", "listops_synth",
        "--model", "HubNet",
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--seed", "42",
        "--d_model", "128",
        "--HubNet_slots", "32",
        "--HubNet_steps", "1",
    ],

    # HUBNET v2 ListOps
    [
        "python", "main.py",
        "--dataset", "listops_synth",
        "--model", "HubNet_v2",
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--seed", "42",
        "--d_model", "128",
        "--HubNet_slots", "32",
        "--HubNet_steps", "1",
    ],

    # HUBNET v1 IMDB
    [
        "python", "main.py",
        "--dataset", "imdb",
        "--model", "HubNet",
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--seed", "42",
        "--d_model", "128",
        "--HubNet_slots", "32",
        "--HubNet_steps", "1",
    ],

    # HUBNET v2 IMDB
    [
        "python", "main.py",
        "--dataset", "imdb",
        "--model", "HubNet_v2",
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--seed", "42",
        "--d_model", "128",
        "--HubNet_slots", "32",
        "--HubNet_steps", "1",
    ],
]

for run in runs:
    print("\nRunning:", " ".join(run))
    subprocess.run(run, check=True)

print("\nAll runs completed.")
