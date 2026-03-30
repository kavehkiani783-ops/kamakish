import subprocess
import sys


def run_cmd(cmd):
    print("\n" + "=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True)


def main():
    datasets = ["imdb", "listops_synth"]
    seeds = [42, 43, 44]

    base_args = [
        "--epochs", "3",
        "--batch_size", "64",
        "--max_len", "512",
        "--d_model", "128",
    ]

    simple_models = [
        ("meanpool", []),
        ("bilstm", []),
        ("tiny_transformer", []),
        ("transformer_base", []),
    ]

    HubNet_v1_args = [
        "--HubNet_slots", "32",
        "--HubNet_steps", "1",
        "--HubNet_topk", "0",
        "--HubNet_dropout", "0.1",
        "--HubNet_score_clip", "20.0",
    ]

    HubNet_v2_args = [
        "--HubNet_slots", "32",
        "--HubNet_steps", "4",
        "--HubNet_topk", "0",
        "--HubNet_dropout", "0.1",
        "--HubNet_score_clip", "20.0",
        "--HubNet_gate",
    ]

    for dataset in datasets:
        for seed in seeds:
            for model_name, extra in simple_models:
                cmd = [
                    sys.executable, "main.py",
                    "--dataset", dataset,
                    "--model", model_name,
                    "--seed", str(seed),
                    *base_args,
                    *extra,
                ]
                run_cmd(cmd)

            cmd = [
                sys.executable, "main.py",
                "--dataset", dataset,
                "--model", "HubNet_v1",
                "--seed", str(seed),
                *base_args,
                *HubNet_v1_args,
            ]
            run_cmd(cmd)

            cmd = [
                sys.executable, "main.py",
                "--dataset", dataset,
                "--model", "HubNet_v2",
                "--seed", str(seed),
                *base_args,
                *HubNet_v2_args,
            ]
            run_cmd(cmd)

    print("\nAll final model comparison runs completed.")


if __name__ == "__main__":
    main()
