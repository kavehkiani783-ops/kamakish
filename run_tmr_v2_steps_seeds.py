import subprocess
import sys


def run_cmd(cmd):
    print("\n" + "=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def main():
    dataset = "listops_synth"
    model = "tmr_v2"
    epochs = 3
    batch_size = 64
    max_len = 512
    d_model = 128
    tmr_slots = 32
    tmr_topk = 0
    tmr_dropout = 0.1
    tmr_score_clip = 20.0

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
                "--tmr_slots", str(tmr_slots),
                "--tmr_steps", str(steps),
                "--tmr_topk", str(tmr_topk),
                "--tmr_dropout", str(tmr_dropout),
                "--tmr_score_clip", str(tmr_score_clip),
                "--tmr_gate",
            ]
            run_cmd(cmd)

    print("\nAll TMR-v2 step/seed runs completed.")


if __name__ == "__main__":
    main()
