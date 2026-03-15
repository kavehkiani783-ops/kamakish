import argparse
import os
import json
from datetime import datetime

import torch

from data.datasets import get_dataset
from training.runner import train_and_evaluate

from models.NubNet_config import NUBNETConfig
from models.NubNet_block import NUBNETModel, NUBNETBlockV2
from models.simple_models import MeanPool, BiLSTM
from models.transformer_models import TinyTransformer, TransformerBase

print("MAIN STARTED")


def build_model(args, vocab_size, num_classes, pad_id):
    name = args.model.lower()

    if name == "NubNet":
        cfg = NUBNETConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            max_len=args.max_len,
            d_model=args.d_model,
            mem_slots=args.NubNet_slots,
            steps=0 if args.NubNet_no_settle else args.NubNet_steps,
            decay=args.NubNet_decay,
            gate=args.NubNet_gate,
            topk=args.NubNet_topk,
            dropout=args.NubNet_dropout,
            score_clip=args.NubNet_score_clip,
        )
        return NUBNETModel(args.d_model, num_classes, cfg)

    if name == "NubNet_v2":
        cfg = NUBNETConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            max_len=args.max_len,
            d_model=args.d_model,
            mem_slots=args.NubNet_slots,
            steps=0 if args.NubNet_no_settle else args.NubNet_steps,
            decay=args.NubNet_decay,
            gate=args.NubNet_gate,
            topk=args.NubNet_topk,
            dropout=args.NubNet_dropout,
            score_clip=args.NubNet_score_clip,
        )
        return NUBNETBlockV2(cfg)

    if name == "meanpool":
        return MeanPool(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_classes=num_classes,
            pad_id=pad_id,
        )

    if name == "bilstm":
        return BiLSTM(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_classes=num_classes,
            pad_id=pad_id,
        )

    if name == "tiny_transformer":
        return TinyTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_classes=num_classes,
            pad_id=pad_id,
        )

    if name == "transformer_base":
        return TransformerBase(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_classes=num_classes,
            pad_id=pad_id,
        )

    raise ValueError(f"Unknown model: {args.model}")


def build_output_filename(args):
    """
    Build informative filenames so NUBNET/NUBNET-v2 ablations do not overwrite each other.
    """
    parts = [args.model, args.dataset, f"seed{args.seed}"]

    if args.model.lower() in {"NubNet", "NubNet_v2"}:
        steps = 0 if args.NubNet_no_settle else args.NubNet_steps
        parts.extend([
            f"steps{steps}",
            f"slots{args.NubNet_slots}",
            f"topk{args.NubNet_topk}",
            f"gate{int(args.NubNet_gate)}",
        ])

    return "_".join(parts) + ".json"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "NubNet",
            "NubNet_v2",
            "meanpool",
            "bilstm",
            "tiny_transformer",
            "transformer_base",
        ],
        default="NubNet",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_size", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")

    # NUBNET controls
    parser.add_argument("--NubNet_steps", type=int, default=4)
    parser.add_argument("--NubNet_slots", type=int, default=64)
    parser.add_argument("--NubNet_decay", type=float, default=0.9)
    parser.add_argument("--NubNet_gate", action="store_true")
    parser.add_argument("--NubNet_topk", type=int, default=0)
    parser.add_argument("--NubNet_no_settle", action="store_true")
    parser.add_argument("--NubNet_dropout", type=float, default=0.1)
    parser.add_argument("--NubNet_score_clip", type=float, default=20.0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model} | d_model={args.d_model} | lr={args.lr}")
    print(f"Dataset: {args.dataset} | max_len={args.max_len} | batch_size={args.batch_size}")

    if args.model.lower() in {"NubNet", "NubNet_v2"}:
        print(
            f"NUBNET config | slots={args.NubNet_slots} | "
            f"steps={0 if args.NubNet_no_settle else args.NubNet_steps} | "
            f"decay={args.NubNet_decay} | gate={args.NubNet_gate} | "
            f"topk={args.NubNet_topk} | dropout={args.NubNet_dropout} | "
            f"score_clip={args.NubNet_score_clip}"
        )

    print("-" * 70)

    train_loader, val_loader, test_loader, meta = get_dataset(
        name=args.dataset,
        batch_size=args.batch_size,
        max_len=args.max_len,
        seed=args.seed,
        val_ratio=args.val_ratio,
        val_size=args.val_size,
    )

    print(
        f"Split sizes | train={meta['train_size']} | "
        f"val={meta['val_size']} | test={meta['test_size']}"
    )
    print("-" * 70)

    model = build_model(
        args=args,
        vocab_size=meta["vocab_size"],
        num_classes=meta["num_classes"],
        pad_id=meta["pad_id"],
    ).to(device)

    results = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_filename = build_output_filename(args)
    out_path = os.path.join(args.output_dir, out_filename)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("-" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
