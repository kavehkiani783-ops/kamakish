import argparse
import os
import json
from datetime import datetime

import torch

from data.datasets import get_dataset
from training.runner import train_and_evaluate

from models.tmr_config import TMRConfig
from models.tmr_block import TMRModel
from models.simple_models import MeanPool, BiLSTM
from models.transformer_models import TinyTransformer, TransformerBase


def build_model(args, vocab_size, num_classes, pad_id):
    name = args.model.lower()

    if name == "tmr":
        topk = int(args.tmr_topk)
        if topk < 0:
            raise ValueError("--tmr_topk must be >= 0 (0 means softmax).")

        cfg = TMRConfig(
            d_model=args.d_model,
            vocab_size=vocab_size,
            num_classes=num_classes,
            pad_id=pad_id,  # REQUIRED by your TMRConfig
            num_slots=args.tmr_slots,
            num_steps=0 if args.tmr_no_settle else args.tmr_steps,
            decay=args.tmr_decay,
            use_gate=args.tmr_gate,
            topk=topk,
        )
        return TMRModel(vocab_size, num_classes, cfg)

    if name == "meanpool":
        # Make MeanPool accept pad_id=None in its __init__ (see patch below)
        return MeanPool(vocab_size=vocab_size, d_model=args.d_model, num_classes=num_classes, pad_id=pad_id)

    if name == "bilstm":
        return BiLSTM(vocab_size=vocab_size, d_model=args.d_model, num_classes=num_classes, pad_id=pad_id)

    if name == "tiny_transformer":
        return TinyTransformer(vocab_size=vocab_size, d_model=args.d_model, num_classes=num_classes, pad_id=pad_id)

    if name == "transformer_base":
        return TransformerBase(vocab_size=vocab_size, d_model=args.d_model, num_classes=num_classes, pad_id=pad_id)

    raise ValueError(f"Unknown model: {args.model}")


def main():
    parser = argparse.ArgumentParser()

    # Core
    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument(
        "--model",
        type=str,
        choices=["tmr", "meanpool", "bilstm", "tiny_transformer", "transformer_base"],
        default="tmr",
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

    # TMR ablations
    parser.add_argument("--tmr_steps", type=int, default=4)
    parser.add_argument("--tmr_slots", type=int, default=64)
    parser.add_argument("--tmr_decay", type=float, default=0.9)
    parser.add_argument("--tmr_gate", action="store_true")
    parser.add_argument("--tmr_topk", type=int, default=0)  # 0 => softmax
    parser.add_argument("--tmr_no_settle", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.dataset} | max_len={args.max_len} | batch_size={args.batch_size}")
    print("-" * 70)

    train_loader, val_loader, test_loader, meta = get_dataset(
        name=args.dataset,
        batch_size=args.batch_size,
        max_len=args.max_len,
        seed=args.seed,
        val_ratio=args.val_ratio,
        val_size=args.val_size,
    )

    print(f"Split sizes | train={meta['train_size']} | val={meta['val_size']} | test={meta['test_size']}")
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
    out_path = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
