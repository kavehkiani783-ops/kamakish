import subprocess

runs = [
    # TMR v1 ListOps
    [
        "python", "main.py",
        "--dataset", "listops_synth",
        "--model", "tmr",
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--seed", "42",
        "--d_model", "128",
        "--tmr_slots", "32",
        "--tmr_steps", "1",
    ],

    # TMR v2 ListOps
    [
        "python", "main.py",
        "--dataset", "listops_synth",
        "--model", "tmr_v2",
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--seed", "42",
        "--d_model", "128",
        "--tmr_slots", "32",
        "--tmr_steps", "1",
    ],

    # TMR v1 IMDB
    [
        "python", "main.py",
        "--dataset", "imdb",
        "--model", "tmr",
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--seed", "42",
        "--d_model", "128",
        "--tmr_slots", "32",
        "--tmr_steps", "1",
    ],

    # TMR v2 IMDB
    [
        "python", "main.py",
        "--dataset", "imdb",
        "--model", "tmr_v2",
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--seed", "42",
        "--d_model", "128",
        "--tmr_slots", "32",
        "--tmr_steps", "1",
    ],
]

for run in runs:
    print("\nRunning:", " ".join(run))
    subprocess.run(run, check=True)

print("\nAll runs completed.")
