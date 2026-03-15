import subprocess
import sys


def run_cmd(cmd):
    print("\n" + "=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def main():
    dataset = "listops_synth"
    model = "NubNet_v2"
    epochs = 3
    batch_size = 64
    max_len = 512
    d_model = 128
    NubNet_slots = 32
    NubNet_topk = 0
    NubNet_dropout = 0.1
    NubNet_score_clip = 20.0

    seeds = [42, 43, 44]
    steps_list = [1, 2, 4]

    for seed in seeds:
        for steps in steps_list:
            cmd = [
                sys.executable, "main.py",
                "--dataset", dataset,
                "--model", model,
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--max_len", str(max_len),
                "--seed", str(seed),
                "--d_model", str(d_model),
                "--NubNet_slots", str(NubNet_slots),
                "--NubNet_steps", str(steps),
                "--NubNet_topk", str(NubNet_topk),
                "--NubNet_dropout", str(NubNet_dropout),
                "--NubNet_score_clip", str(NubNet_score_clip),
                "--NubNet_gate",
            ]
            run_cmd(cmd)

    print("\nAll NUBNET-v2 step/seed runs completed.")


if __name__ == "__main__":
    main()
