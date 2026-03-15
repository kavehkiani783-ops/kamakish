import subprocess
import sys


def run_cmd(cmd):
    print("\n" + "=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def main():
    dataset = "listops_synth"
    model = "HubNet_v2"
    epochs = 3
    batch_size = 64
    max_len = 512
    d_model = 128
    HubNet_slots = 32
    HubNet_topk = 0
    HubNet_dropout = 0.1
    HubNet_score_clip = 20.0

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
                "--HubNet_slots", str(HubNet_slots),
                "--HubNet_steps", str(steps),
                "--HubNet_topk", str(HubNet_topk),
                "--HubNet_dropout", str(HubNet_dropout),
                "--HubNet_score_clip", str(HubNet_score_clip),
                "--HubNet_gate",
            ]
            run_cmd(cmd)

    print("\nAll HUBNET-v2 step/seed runs completed.")


if __name__ == "__main__":
    main()
