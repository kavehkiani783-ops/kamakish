import argparse
import os
import json
from datetime import datetime

import torch

from data.datasets import get_dataset
from training.runner import train_and_evaluate

from models.HubNet_config import HUBNETConfig
from models.HubNet_block import HUBNETModel, HUBNETBlockV2
from models.simple_models import MeanPool, BiLSTM
from models.transformer_models import TinyTransformer, TransformerBase

print("MAIN STARTED")


def normalise_model_name(name: str) -> str:
    """
    Make model selection case-insensitive and allow legacy aliases.
    """
    name = name.strip().lower()

    aliases = {
        "hubnet": "hubnet",
        "hubnet_v2": "hubnet_v2",
        "tmr": "hubnet",
        "tmr_v2": "hubnet_v2",
        "meanpool": "meanpool",
        "bilstm": "bilstm",
        "tiny_transformer": "tiny_transformer",
        "transformer_base": "transformer_base",
    }

    if name not in aliases:
        raise ValueError(f"Unknown model: {name}")

    return aliases[name]


def build_model(args, vocab_size, num_classes, pad_id):
    name = normalise_model_name(args.model)

    if name == "hubnet":
        cfg = HUBNETConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            max_len=args.max_len,
            d_model=args.d_model,
            mem_slots=args.HubNet_slots,
            steps=0 if args.HubNet_no_settle else args.HubNet_steps,
            decay=args.HubNet_decay,
            gate=args.HubNet_gate,
            topk=args.HubNet_topk,
            dropout=args.HubNet_dropout,
            score_clip=args.HubNet_score_clip,
        )
        return HUBNETModel(args.d_model, num_classes, cfg)

    if name == "hubnet_v2":
        cfg = HUBNETConfig(
            vocab_size=vocab_size,
            num_classes=num_classes,
            max_len=args.max_len,
            d_model=args.d_model,
            mem_slots=args.HubNet_slots,
            steps=0 if args.HubNet_no_settle else args.HubNet_steps,
            decay=args.HubNet_decay,
            gate=args.HubNet_gate,
            topk=args.HubNet_topk,
            dropout=args.HubNet_dropout,
            score_clip=args.HubNet_score_clip,
        )
        return HUBNETBlockV2(cfg)

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
    Build informative filenames so HubNet / HubNet-v2 runs do not overwrite each other.
    """
    model_name = normalise_model_name(args.model)
    parts = [model_name, args.dataset, f"seed{args.seed}"]

    if model_name in {"hubnet", "hubnet_v2"}:
        steps = 0 if args.HubNet_no_settle else args.HubNet_steps
        parts.extend([
            f"steps{steps}",
            f"slots{args.HubNet_slots}",
            f"topk{args.HubNet_topk}",
            f"gate{int(args.HubNet_gate)}",
        ])

    return "_".join(parts) + ".json"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="imdb")
    parser.add_argument(
        "--model",
        type=str,
        default="hubnet",
        help=(
            "Supported: hubnet, hubnet_v2, meanpool, bilstm, "
            "tiny_transformer, transformer_base. "
            "Legacy aliases tmr and tmr_v2 are also accepted."
        ),
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

    # HubNet controls
    parser.add_argument("--HubNet_steps", type=int, default=4)
    parser.add_argument("--HubNet_slots", type=int, default=64)
    parser.add_argument("--HubNet_decay", type=float, default=0.9)
    parser.add_argument("--HubNet_gate", action="store_true")
    parser.add_argument("--HubNet_topk", type=int, default=0)
    parser.add_argument("--HubNet_no_settle", action="store_true")
    parser.add_argument("--HubNet_dropout", type=float, default=0.1)
    parser.add_argument("--HubNet_score_clip", type=float, default=20.0)

    args = parser.parse_args()
    model_name = normalise_model_name(args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_name} | d_model={args.d_model} | lr={args.lr}")
    print(f"Dataset: {args.dataset} | max_len={args.max_len} | batch_size={args.batch_size}")

    if model_name in {"hubnet", "hubnet_v2"}:
        print(
            f"HubNet config | slots={args.HubNet_slots} | "
            f"steps={0 if args.HubNet_no_settle else args.HubNet_steps} | "
            f"decay={args.HubNet_decay} | gate={args.HubNet_gate} | "
            f"topk={args.HubNet_topk} | dropout={args.HubNet_dropout} | "
            f"score_clip={args.HubNet_score_clip}"
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
