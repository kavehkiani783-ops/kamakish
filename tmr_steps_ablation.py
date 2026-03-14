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
    "topk": [0, 1, 2, 4],
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
        "--dataset", dataset,
        "--model", "tmr",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--max_len", str(max_len),
        "--val_ratio", str(val_ratio),
        "--d_model", str(d_model),
        "--lr", str(lr),
        "--seed", str(seed),
        "--output_dir", output_dir,
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

    return {"test_acc": test_acc, "epoch_time_min": epoch_time_min}


def parse_metrics_from_json(json_path: Path) -> Dict[str, Any]:

    if not json_path.exists():
        return {"test_acc": None, "epoch_time_min": None}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"test_acc": None, "epoch_time_min": None}

    acc_keys = ["test_acc", "accuracy", "acc", "test_accuracy"]
    time_keys = ["epoch_time_min", "epoch_time", "time", "train_time_min"]

    test_acc = None
    epoch_time_min = None

    for key in acc_keys:
        if key in data:
            test_acc = float(data[key])
            break

    for key in time_keys:
        if key in data:
            epoch_time_min = float(data[key])
            break

    return {"test_acc": test_acc, "epoch_time_min": epoch_time_min}


def run_one(
    python_exec,
    main_py,
    dataset,
    seed,
    epochs,
    batch_size,
    max_len,
    val_ratio,
    d_model,
    lr,
    tmr_cfg,
    out_dir,
    log_dir,
):

    run_name = build_run_name(dataset, seed, tmr_cfg)

    run_output_dir = out_dir / "model_outputs" / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(
        python_exec,
        main_py,
        dataset,
        seed,
        epochs,
        batch_size,
        max_len,
        val_ratio,
        d_model,
        lr,
        tmr_cfg,
        str(run_output_dir),
    )

    log_path = log_dir / f"{run_name}.log"
    json_path = run_output_dir / f"tmr_{dataset}_{seed}.json"

    wall_start = time.time()

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    full_output = proc.stdout + "\n" + proc.stderr

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(full_output)

    wall_time_min = round((time.time() - wall_start) / 60.0, 4)

    parsed_output = parse_metrics_from_output(full_output)
    parsed_json = parse_metrics_from_json(json_path)

    test_acc = parsed_output["test_acc"] or parsed_json["test_acc"]
    epoch_time_min = parsed_output["epoch_time_min"] or parsed_json["epoch_time_min"]

    return {
        "dataset": dataset,
        "seed": seed,
        "steps": tmr_cfg["steps"],
        "mem_slots": tmr_cfg["mem_slots"],
        "decay": tmr_cfg["decay"],
        "gate": tmr_cfg["gate"],
        "topk": tmr_cfg["topk"],
        "returncode": proc.returncode,
        "test_acc": test_acc,
        "epoch_time_min": epoch_time_min,
        "wall_time_min": wall_time_min,
    }


def write_csv(results, csv_path):

    fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def mean(values):
    return sum(values) / len(values)


def std(values):
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def write_summary_csv(results, axis, csv_path):

    ok = [r for r in results if r["returncode"] == 0]

    grouped: Dict[str, Dict[Any, List[Dict[str, Any]]]] = {}

    for r in ok:
        grouped.setdefault(r["dataset"], {}).setdefault(r[axis], []).append(r)

    rows = []

    for dataset in grouped:
        for val in grouped[dataset]:

            runs = grouped[dataset][val]

            accs = [r["test_acc"] for r in runs if r["test_acc"] is not None]
            epochs = [r["epoch_time_min"] for r in runs if r["epoch_time_min"] is not None]
            walls = [r["wall_time_min"] for r in runs if r["wall_time_min"] is not None]

            rows.append({
                "dataset": dataset,
                axis: val,
                "n": len(runs),
                "mean_acc": mean(accs) if accs else None,
                "std_acc": std(accs) if accs else None,
                "mean_epoch_time": mean(epochs) if epochs else None,
                "mean_wall_time": mean(walls) if walls else None,
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--axis", required=True, choices=list(ABLATION_AXES.keys()))
    parser.add_argument("--datasets", nargs="+", default=["imdb", "listops_synth"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--main_py", default="main.py")
    parser.add_argument("--python_exec", default=sys.executable)

    parser.add_argument("--out_dir", default="ablation_runs/steps_ablation")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    log_dir = out_dir / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_outputs").mkdir(parents=True, exist_ok=True)

    base_cfg = make_base_tmr()
    values = ABLATION_AXES[args.axis]

    results = []

    for dataset in args.datasets:
        for seed in args.seeds:
            for val in values:

                cfg = dict(base_cfg)
                cfg[args.axis] = val

                result = run_one(
                    args.python_exec,
                    args.main_py,
                    dataset,
                    seed,
                    args.epochs,
                    args.batch_size,
                    args.max_len,
                    args.val_ratio,
                    args.d_model,
                    args.lr,
                    cfg,
                    out_dir,
                    log_dir,
                )

                results.append(result)

    raw_csv = out_dir / f"results_{args.axis}.csv"
    write_csv(results, raw_csv)

    summary_csv = out_dir / f"summary_{args.axis}.csv"
    write_summary_csv(results, args.axis, summary_csv)

    print("\nRaw runs saved to:", raw_csv)
    print("Summary comparison table saved to:", summary_csv)


if __name__ == "__main__":
    main()
