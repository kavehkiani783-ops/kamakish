import subprocess
import sys

runs = [
    [sys.executable, "main.py", "--dataset", "imdb", "--model", "HubNet", "--epochs", "1", "--batch_size", "8", "--max_len", "128", "--seed", "42", "--d_model", "64", "--HubNet_slots", "8", "--HubNet_steps", "1"],
    [sys.executable, "main.py", "--dataset", "imdb", "--model", "HubNet_v2", "--epochs", "1", "--batch_size", "8", "--max_len", "128", "--seed", "42", "--d_model", "64", "--HubNet_slots", "8", "--HubNet_steps", "1", "--HubNet_gate"],
    [sys.executable, "main.py", "--dataset", "listops_synth", "--model", "HubNet", "--epochs", "1", "--batch_size", "8", "--max_len", "128", "--seed", "42", "--d_model", "64", "--HubNet_slots", "8", "--HubNet_steps", "1"],
    [sys.executable, "main.py", "--dataset", "listops_synth", "--model", "HubNet_v2", "--epochs", "1", "--batch_size", "8", "--max_len", "128", "--seed", "42", "--d_model", "64", "--HubNet_slots", "8", "--HubNet_steps", "1", "--HubNet_gate"],
]

for cmd in runs:
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)

print("\nAll smoke tests passed.")
