# tmr_ablations.py

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any


ABLATION_AXES: Dict[str, List[Any]] = {
    "steps": [0, 1, 2, 4, 8],
    "mem_slots": [32, 64, 128, 256],
    "decay": [0.7, 0.85, 0.95],
    "gate": [False, True],
    "topk": [0, 1, 2, 4],   # 0 = softmax, >0 = sparse top-k
}


def make_base_tmr() -> Dict[str, Any]:
    return {
        "steps": 4,
        "mem_slots": 64,
        "decay": 0.9,
        "gate": False,
        "topk": 0,
    }


def build_run_name(dataset: str, seed: int, tmr_cfg: Dict[str, Any]) -> str:
    return (
        f"{dataset}"
        f"_seed{seed}"
        f"_steps{tmr_cfg['steps']}"
        f"_slots{tmr_cfg['mem_slots']}"
        f"_decay{tmr_cfg['decay']}"
        f"_gate{int(tmr_cfg['gate'])}"
        f"_topk{tmr_cfg['topk']}"
    )


def build_command(
    python_exec: str,
    main_py: str,
    dataset: str,
    seed: int,
    epochs: int,
    batch_size: int,
    max_len: int,
    val_ratio: float,
    d_model: int,
    lr: float,
    tmr_cfg: Dict[str, Any],
    output_dir: str,
) -> List[str]:
    cmd = [
        python_exec,
        main_py,
        "--dataset", str(dataset),
        "--model", "tmr",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--max_len", str(max_len),
        "--val_ratio", str(val_ratio),
        "--d_model", str(d_model),
        "--lr", str(lr),
        "--seed", str(seed),
        "--output_dir", str(output_dir),
        "--tmr_steps", str(tmr_cfg["steps"]),
        "--tmr_slots", str(tmr_cfg["mem_slots"]),
        "--tmr_decay", str(tmr_cfg["decay"]),
        "--tmr_topk", str(tmr_cfg["topk"]),
    ]

    if tmr_cfg["gate"]:
        cmd.append("--tmr_gate")

    return cmd


def parse_metrics_from_output(output_text: str) -> Dict[str, Any]:
    text = output_text.lower()

    test_acc = None
    epoch_time_min = None

    acc_patterns = [
        r"test\s+acc(?:uracy)?\s*[:=]?\s*([0-9]*\.?[0-9]+)",
        r"\bacc\s*=\s*([0-9]*\.?[0-9]+)",
        r"\bacc(?:uracy)?\s*[:=]?\s*([0-9]*\.?[0-9]+)",
    ]

    time_patterns = [
        r"epoch[_\s]?time\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*min",
        r"\btime\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*min",
        r"\btime\s*=\s*([0-9]*\.?[0-9]+)",
    ]

    for pat in acc_patterns:
        m = re.search(pat, text)
        if m:
            test_acc = float(m.group(1))
            break

    for pat in time_patterns:
        m = re.search(pat, text)
        if m:
            epoch_time_min = float(m.group(1))
            break

    return {
        "test_acc": test_acc,
        "epoch_time_min": epoch_time_min,
    }


def parse_metrics_from_json(json_path: Path) -> Dict[str, Any]:
    if not json_path.exists():
        return {"test_acc": None, "epoch_time_min": None, "raw_json": None}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"test_acc": None, "epoch_time_min": None, "raw_json": None}

    test_acc = None
    epoch_time_min = None

    acc_keys = [
        "test_acc",
        "accuracy",
        "acc",
        "test_accuracy",
    ]

    time_keys = [
        "epoch_time_min",
        "epoch_time",
        "time",
        "train_time_min",
    ]

    for key in acc_keys:
        if key in data and isinstance(data[key], (int, float)):
            test_acc = float(data[key])
            break

    for key in time_keys:
        if key in data and isinstance(data[key], (int, float)):
            epoch_time_min = float(data[key])
            break

    return {
        "test_acc": test_acc,
        "epoch_time_min": epoch_time_min,
        "raw_json": data,
    }


def run_one(
    python_exec: str,
    main_py: str,
    dataset: str,
    seed: int,
    epochs: int,
    batch_size: int,
    max_len: int,
    val_ratio: float,
    d_model: int,
    lr: float,
    tmr_cfg: Dict[str, Any],
    out_dir: Path,
    log_dir: Path,
    live: bool = False,
) -> Dict[str, Any]:
    run_name = build_run_name(dataset, seed, tmr_cfg)
    run_output_dir = out_dir / "model_outputs" / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(
        python_exec=python_exec,
        main_py=main_py,
        dataset=dataset,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        max_len=max_len,
        val_ratio=val_ratio,
        d_model=d_model,
        lr=lr,
        tmr_cfg=tmr_cfg,
        output_dir=str(run_output_dir),
    )

    log_path = log_dir / f"{run_name}.log"
    json_path = run_output_dir / f"tmr_{dataset}_{seed}.json"

    print("\n" + "=" * 100)
    print(
        f"RUN START | dataset={dataset} | seed={seed} | "
        f"steps={tmr_cfg['steps']} | mem_slots={tmr_cfg['mem_slots']} | "
        f"decay={tmr_cfg['decay']} | gate={tmr_cfg['gate']} | topk={tmr_cfg['topk']}"
    )
    print("-" * 100)
    print("Command:")
    print(" ".join(cmd))
    print("=" * 100)

    wall_start = time.time()

    if live:
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            collected_output = []

            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                collected_output.append(line)

            process.wait()
            returncode = process.returncode
            full_output = "".join(collected_output)
    else:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        returncode = proc.returncode
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        full_output = stdout_text + "\n" + stderr_text

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("COMMAND:\n")
            f.write(" ".join(cmd) + "\n\n")
            f.write("STDOUT:\n")
            f.write(stdout_text)
            f.write("\n\nSTDERR:\n")
            f.write(stderr_text)

    wall_time_min = round((time.time() - wall_start) / 60.0, 4)

    parsed_output = parse_metrics_from_output(full_output)
    parsed_json = parse_metrics_from_json(json_path)

    test_acc = parsed_output["test_acc"]
    epoch_time_min = parsed_output["epoch_time_min"]

    if test_acc is None:
        test_acc = parsed_json["test_acc"]
    if epoch_time_min is None:
        epoch_time_min = parsed_json["epoch_time_min"]

    result = {
        "dataset": dataset,
        "seed": seed,
        "steps": tmr_cfg["steps"],
        "mem_slots": tmr_cfg["mem_slots"],
        "decay": tmr_cfg["decay"],
        "gate": tmr_cfg["gate"],
        "topk": tmr_cfg["topk"],
        "returncode": returncode,
        "test_acc": test_acc,
        "epoch_time_min": epoch_time_min,
        "wall_time_min": wall_time_min,
        "log_file": str(log_path),
        "json_file": str(json_path),
        "run_output_dir": str(run_output_dir),
    }

    print("\n" + "-" * 100)
    if returncode != 0:
        print(
            f"RUN END   | FAILED | dataset={dataset} | seed={seed} | "
            f"steps={tmr_cfg['steps']} | log={log_path}"
        )
    else:
        print(
            f"RUN END   | OK | dataset={dataset} | seed={seed} | "
            f"steps={tmr_cfg['steps']} | mem_slots={tmr_cfg['mem_slots']} | "
            f"test_acc={result['test_acc']} | epoch_time_min={result['epoch_time_min']} | "
            f"wall_time_min={result['wall_time_min']}"
        )
        print(f"Log: {log_path}")
        print(f"JSON: {json_path}")
    print("-" * 100)

    return result


def write_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "dataset",
        "seed",
        "steps",
        "mem_slots",
        "decay",
        "gate",
        "topk",
        "returncode",
        "test_acc",
        "epoch_time_min",
        "wall_time_min",
        "log_file",
        "json_file",
        "run_output_dir",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def std(values: List[float]) -> float:
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def print_summary(results: List[Dict[str, Any]], axis: str) -> None:
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    ok_results = [r for r in results if r["returncode"] == 0]

    if not ok_results:
        print("No successful runs.")
        return

    grouped: Dict[str, Dict[Any, List[Dict[str, Any]]]] = {}

    for r in ok_results:
        dataset = r["dataset"]
        axis_value = r[axis]
        grouped.setdefault(dataset, {}).setdefault(axis_value, []).append(r)

    for dataset in grouped:
        print(f"\nDataset: {dataset}")
        print("-" * 100)
        print(
            f"{axis:>12} | {'n':>3} | {'mean_acc':>10} | {'std_acc':>10} | "
            f"{'mean_epoch_time':>16} | {'mean_wall_time':>15}"
        )
        print("-" * 100)

        values_sorted = sorted(grouped[dataset].keys(), key=lambda x: str(x))

        for axis_value in values_sorted:
            rows = grouped[dataset][axis_value]

            accs = [r["test_acc"] for r in rows if isinstance(r["test_acc"], (int, float))]
            epoch_times = [r["epoch_time_min"] for r in rows if isinstance(r["epoch_time_min"], (int, float))]
            wall_times = [r["wall_time_min"] for r in rows if isinstance(r["wall_time_min"], (int, float))]

            acc_mean = mean(accs) if accs else None
            acc_std = std(accs) if len(accs) > 0 else None
            epoch_mean = mean(epoch_times) if epoch_times else None
            wall_mean = mean(wall_times) if wall_times else None

            acc_mean_str = f"{acc_mean:.4f}" if acc_mean is not None else "NA"
            acc_std_str = f"{acc_std:.4f}" if acc_std is not None else "NA"
            epoch_mean_str = f"{epoch_mean:.4f}" if epoch_mean is not None else "NA"
            wall_mean_str = f"{wall_mean:.4f}" if wall_mean is not None else "NA"

            print(
                f"{str(axis_value):>12} | {len(rows):>3} | {acc_mean_str:>10} | "
                f"{acc_std_str:>10} | {epoch_mean_str:>16} | {wall_mean_str:>15}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Automated TMR ablation runner"
    )

    parser.add_argument("--main_py", type=str, default="main.py")
    parser.add_argument("--python_exec", type=str, default=sys.executable)

    parser.add_argument(
        "--axis",
        type=str,
        required=True,
        choices=["steps", "mem_slots", "decay", "gate", "topk"],
    )

    parser.add_argument("--datasets", nargs="+", default=["imdb", "listops_synth"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--out_dir", type=str, default="ablation_runs")
    parser.add_argument("--live", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_outputs").mkdir(parents=True, exist_ok=True)

    base_cfg = make_base_tmr()
    values = ABLATION_AXES[args.axis]

    csv_path = out_dir / f"results_{args.axis}.csv"
    results: List[Dict[str, Any]] = []

    print("=" * 100)
    print("STARTING TMR ABLATION SWEEP")
    print("=" * 100)
    print(f"Axis:       {args.axis}")
    print(f"Values:     {values}")
    print(f"Datasets:   {args.datasets}")
    print(f"Seeds:      {args.seeds}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max len:    {args.max_len}")
    print(f"d_model:    {args.d_model}")
    print(f"lr:         {args.lr}")
    print(f"Live mode:  {args.live}")
    print(f"Output dir: {out_dir}")
    print("=" * 100)

    total_runs = len(args.datasets) * len(args.seeds) * len(values)
    run_counter = 0

    for dataset in args.datasets:
        for seed in args.seeds:
            for value in values:
                run_counter += 1
                cfg = dict(base_cfg)
                cfg[args.axis] = value

                print(f"\nProgress: {run_counter}/{total_runs}")

                result = run_one(
                    python_exec=args.python_exec,
                    main_py=args.main_py,
                    dataset=dataset,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                    val_ratio=args.val_ratio,
                    d_model=args.d_model,
                    lr=args.lr,
                    tmr_cfg=cfg,
                    out_dir=out_dir,
                    log_dir=log_dir,
                    live=args.live,
                )

                results.append(result)
                write_csv(results, csv_path)

    print_summary(results, axis=args.axis)

    print("\n" + "=" * 100)
    print("DONE")
    print(f"CSV saved to: {csv_path}")
    print(f"Logs saved to: {log_dir}")
    print(f"Model outputs saved to: {out_dir / 'model_outputs'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
