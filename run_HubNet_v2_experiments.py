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
                    "--model", "HubNet_v2",
                    "--epochs", "3",
                    "--batch_size", "64",
                    "--max_len", "512",
                    "--seed", str(seed),
                    "--d_model", "128",
                    "--HubNet_slots", "32",
                    "--HubNet_steps", str(steps),
                    "--HubNet_topk", "0",
                    "--HubNet_dropout", "0.1",
                    "--HubNet_score_clip", "20.0",
                    "--HubNet_gate",
                ]

                run_cmd(cmd)

    print("\nAll HUBNET-v2 experiments finished.")


if __name__ == "__main__":
    main()
