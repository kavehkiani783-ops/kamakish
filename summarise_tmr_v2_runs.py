import subprocess
import sys


def run_cmd(cmd):
    print("\n" + "=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def main():

    datasets = ["listops_synth", "imdb"]
    seeds = [42, 43, 44]
    steps_list = [1, 2, 4]

    for dataset in datasets:
        for steps in steps_list:
            for seed in seeds:

                cmd = [
                    sys.executable, "main.py",
                    "--dataset", dataset,
                    "--model", "tmr_v2",
                    "--epochs", "3",
                    "--batch_size", "64",
                    "--max_len", "512",
                    "--seed", str(seed),
                    "--d_model", "128",
                    "--tmr_slots", "32",
                    "--tmr_steps", str(steps),
                    "--tmr_topk", "0",
                    "--tmr_dropout", "0.1",
                    "--tmr_score_clip", "20.0",
                    "--tmr_gate",
                ]

                run_cmd(cmd)

    print("\nAll TMR-v2 experiments finished.")


if __name__ == "__main__":
    main()
