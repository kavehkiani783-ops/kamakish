import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


# ----------------------------
# Shared utilities
# ----------------------------
def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class DatasetMeta:
    num_classes: int
    label_key: str
    train_size: int
    val_size: int
    test_size: int
    vocab_size: int


class _TokenDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, label_key="label"):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.label_key = label_key

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            self.label_key: self.labels[idx],
        }


def _make_loaders(train_ds, val_ds, test_ds, batch_size: int):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


# ----------------------------
# Synthetic ListOps generator
# ----------------------------
VOCAB_SIZE = 30522
PAD_ID = 0

TOKEN_TO_ID = {
    "<pad>": PAD_ID,
    "(": 1,
    ")": 2,
    "MAX": 3,
    "MIN": 4,
    "SUM": 5,
    "0": 10,
    "1": 11,
    "2": 12,
    "3": 13,
    "4": 14,
    "5": 15,
    "6": 16,
    "7": 17,
    "8": 18,
    "9": 19,
}
OPS = ["MAX", "MIN", "SUM"]


def _apply_op(op: str, args: List[int]) -> int:
    if op == "MAX":
        return int(max(args))
    if op == "MIN":
        return int(min(args))
    if op == "SUM":
        return int(sum(args) % 10)  # keep labels 0..9
    raise ValueError(op)


def _gen_expr(rng: random.Random, max_depth: int, min_args: int, max_args: int, p_leaf: float) -> Tuple[List[str], int]:
    """
    Expression grammar:
      digit
      ( OP expr expr ... )
    Returns tokens and value in 0..9
    """
    if max_depth <= 0 or rng.random() < p_leaf:
        v = rng.randint(0, 9)
        return [str(v)], v

    op = rng.choice(OPS)
    n_args = rng.randint(min_args, max_args)

    toks = ["(", op]
    vals = []
    for _ in range(n_args):
        t, v = _gen_expr(rng, max_depth - 1, min_args, max_args, p_leaf)
        toks.extend(t)
        vals.append(v)
    toks.append(")")
    return toks, _apply_op(op, vals)


def _encode_tokens(tokens: List[str], max_len: int) -> Tuple[List[int], List[int]]:
    ids = [TOKEN_TO_ID[t] for t in tokens]
    # IMPORTANT: no truncation here — generator guarantees len(ids) <= max_len
    attn = [1] * len(ids)

    if len(ids) < max_len:
        pad_n = max_len - len(ids)
        ids += [PAD_ID] * pad_n
        attn += [0] * pad_n

    return ids, attn


def _build_synth_listops_length_controlled(
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    max_len: int,
    max_depth: int,
    min_args: int,
    max_args: int,
    p_leaf: float,
    max_tries_per_example: int = 200,
) -> Tuple[_TokenDataset, _TokenDataset, _TokenDataset, DatasetMeta, Dict]:
    """
    Key difference vs your old generator:
      - We reject any sample whose token length > max_len
      - So labels always correspond to fully-observed programs (no truncation noise)
    """
    rng = random.Random(seed)

    def make_split(n: int):
        X_ids, X_mask, y = [], [], []
        lengths = []
        rejected = 0
        total_tries = 0

        for _ in range(n):
            ok = False
            for _try in range(max_tries_per_example):
                total_tries += 1
                tokens, val = _gen_expr(rng, max_depth=max_depth, min_args=min_args, max_args=max_args, p_leaf=p_leaf)
                if len(tokens) <= max_len:
                    ids, attn = _encode_tokens(tokens, max_len=max_len)
                    X_ids.append(ids)
                    X_mask.append(attn)
                    y.append(val)
                    lengths.append(sum(attn))
                    ok = True
                    break
                else:
                    rejected += 1

            if not ok:
                # Fallback: if generation is too hard under constraints, force a leaf.
                # This avoids infinite loops while preserving correctness.
                v = rng.randint(0, 9)
                tokens = [str(v)]
                ids, attn = _encode_tokens(tokens, max_len=max_len)
                X_ids.append(ids)
                X_mask.append(attn)
                y.append(v)
                lengths.append(1)

        X_ids = torch.tensor(X_ids, dtype=torch.long)
        X_mask = torch.tensor(X_mask, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        stats = {
            "mean": float(np.mean(lengths)),
            "p95": float(np.percentile(lengths, 95)),
            "max": int(np.max(lengths)),
            "rejected": int(rejected),
            "tries": int(total_tries),
            "rejection_rate": float(rejected / max(1, total_tries)),
        }
        return _TokenDataset(X_ids, X_mask, y, label_key="label"), stats

    train_ds, train_stats = make_split(n_train)
    val_ds, val_stats = make_split(n_val)
    test_ds, test_stats = make_split(n_test)

    meta = DatasetMeta(
        num_classes=10,
        label_key="label",
        train_size=n_train,
        val_size=n_val,
        test_size=n_test,
        vocab_size=VOCAB_SIZE,
    )

    return train_ds, val_ds, test_ds, meta, {"train": train_stats, "val": val_stats, "test": test_stats}


# ----------------------------
# IMDB loader (simple char->id mapping)
# ----------------------------
def _simple_char_tokenise(text: str, max_len: int):
    ids = [min(ord(c), 255) for c in text][:max_len]
    ids = [i if i != 0 else 1 for i in ids]
    attn = [1] * len(ids)
    if len(ids) < max_len:
        pad_n = max_len - len(ids)
        ids += [PAD_ID] * pad_n
        attn += [0] * pad_n
    return ids, attn


def _build_imdb(seed: int, batch_size: int, max_len: int, val_ratio: float, val_size: Optional[int]):
    if load_dataset is None:
        raise RuntimeError("datasets library not available. Install with: pip install datasets")

    _set_seed(seed)
    ds = load_dataset("imdb")
    train_raw = ds["train"]
    test_raw = ds["test"]

    n_train = len(train_raw)
    if val_size is None:
        v = int(round(n_train * val_ratio))
    else:
        v = int(val_size)
    v = max(1, min(v, n_train - 1))

    idx = np.arange(n_train)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    val_idx = idx[:v]
    tr_idx = idx[v:]

    def encode_split(raw_split, indices=None):
        texts = raw_split["text"] if indices is None else [raw_split["text"][i] for i in indices]
        labels = raw_split["label"] if indices is None else [raw_split["label"][i] for i in indices]

        X_ids, X_mask = [], []
        for t in texts:
            ids, attn = _simple_char_tokenise(t, max_len=max_len)
            X_ids.append(ids)
            X_mask.append(attn)

        X_ids = torch.tensor(X_ids, dtype=torch.long)
        X_mask = torch.tensor(X_mask, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        return _TokenDataset(X_ids, X_mask, y, label_key="label")

    train_ds = encode_split(train_raw, tr_idx)
    val_ds = encode_split(train_raw, val_idx)
    test_ds = encode_split(test_raw, None)

    meta = DatasetMeta(
        num_classes=2,
        label_key="label",
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
        vocab_size=VOCAB_SIZE,
    )

    return _make_loaders(train_ds, val_ds, test_ds, batch_size), meta


# ----------------------------
# Public API
# ----------------------------
def get_dataset(
    name: str,
    batch_size: int,
    max_len: int,
    seed: int,
    val_ratio: float = 0.1,
    val_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    name = name.lower().strip()

    if name in ["listops_synth", "synthetic_listops", "synth_listops"]:
        n_train, n_val, n_test = 50000, 2000, 5000

        # Difficulty knobs (you can change these later)
        # IMPORTANT: we now guarantee sequences fit max_len (no truncation)
        max_depth = 6
        min_args = 2
        max_args = 5
        p_leaf = 0.35

        train_ds, val_ds, test_ds, meta, stats = _build_synth_listops_length_controlled(
            seed=seed,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            max_len=max_len,
            max_depth=max_depth,
            min_args=min_args,
            max_args=max_args,
            p_leaf=p_leaf,
            max_tries_per_example=200,
        )

        train_loader, val_loader, test_loader = _make_loaders(train_ds, val_ds, test_ds, batch_size)

        print("Generating data (length-controlled; NO truncation)...")
        tr = stats["train"]
        print(
            f"Length stats (train): mean={tr['mean']:.1f} | p95={tr['p95']:.0f} | max={tr['max']} | "
            f"rejection_rate={tr['rejection_rate']*100:.2f}%"
        )

        metadata = {
            "num_classes": meta.num_classes,
            "label_key": meta.label_key,
            "train_size": meta.train_size,
            "val_size": meta.val_size,
            "test_size": meta.test_size,
            "vocab_size": meta.vocab_size,
            "pad_id": PAD_ID,
            "dataset_name": "Synthetic-ListOps (length-controlled)",
        }
        return train_loader, val_loader, test_loader, metadata

    if name == "imdb":
        (train_loader, val_loader, test_loader), meta = _build_imdb(
            seed=seed,
            batch_size=batch_size,
            max_len=max_len,
            val_ratio=val_ratio,
            val_size=val_size,
        )
        metadata = {
            "num_classes": meta.num_classes,
            "label_key": meta.label_key,
            "train_size": meta.train_size,
            "val_size": meta.val_size,
            "test_size": meta.test_size,
            "vocab_size": meta.vocab_size,
            "pad_id": PAD_ID,
            "dataset_name": "imdb",
        }
        return train_loader, val_loader, test_loader, metadata

    raise ValueError(f"Unknown dataset: {name}. Use 'imdb' or 'listops_synth'.")
