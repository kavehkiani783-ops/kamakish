import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)


def _safe_float(x):
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return float(x)


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc_in = accuracies[in_bin].float().mean()
            conf_in = confidences[in_bin].mean()
            ece += torch.abs(conf_in - acc_in) * prop

    return float(ece.item())


@torch.no_grad()
def evaluate(model, loader, device, criterion, label_key="label"):
    model.eval()

    all_logits = []
    all_labels = []
    delta_norms_acc = []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch[label_key].to(device)

        try:
            out = model(input_ids, attention_mask, return_deltas=True)
        except TypeError:
            out = model(input_ids, attention_mask)

        if isinstance(out, tuple) and len(out) == 2:
            logits, delta_norms = out
            if delta_norms is not None:
                delta_norms_acc.append(delta_norms.detach())
        else:
            logits = out

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    nll = F.nll_loss(torch.log(probs + 1e-12), labels).item()

    ece = compute_ece(probs, labels)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean().item()
    mean_conf = probs.max(dim=1)[0].mean().item()

    # Binary-only metrics: guard for single-class splits
    auroc = None
    auprc = None
    brier = None
    y_prob_pos = None

    if probs.shape[1] == 2:
        pos_probs = probs[:, 1]
        y_prob_pos_np = pos_probs.cpu().numpy()
        y_prob_pos = y_prob_pos_np.tolist()

        # Brier
        brier = torch.mean((pos_probs - labels.float()) ** 2).item()

        # AUROC/AUPRC only defined if both classes present
        unique = np.unique(y_true)
        if unique.size == 2:
            try:
                auroc = roc_auc_score(y_true, y_prob_pos_np)
                auprc = average_precision_score(y_true, y_prob_pos_np)
            except Exception:
                auroc, auprc = None, None

    # Delta norms aggregation
    resonance_delta_norms = None
    if len(delta_norms_acc) > 0:
        lengths = [d.numel() for d in delta_norms_acc]
        Kmax = max(lengths)
        padded = []
        for d in delta_norms_acc:
            if d.numel() < Kmax:
                pad = torch.full((Kmax - d.numel(),), float("nan"), device=d.device)
                d = torch.cat([d, pad], dim=0)
            padded.append(d)
        delta_mean = torch.stack(padded, dim=0).nanmean(dim=0)
        resonance_delta_norms = delta_mean.cpu().tolist()

        # If nanmean still yields nan (pathological), replace with None later via main.py sanitiser

    return {
        "loss": _safe_float(total_loss / len(loader.dataset)),
        "accuracy": _safe_float(acc),
        "macro_f1": _safe_float(macro_f1),
        "weighted_f1": _safe_float(weighted_f1),
        "balanced_accuracy": _safe_float(bal_acc),
        "auroc": _safe_float(auroc) if auroc is not None else None,
        "auprc": _safe_float(auprc) if auprc is not None else None,
        "nll": _safe_float(nll),
        "brier": _safe_float(brier) if brier is not None else None,
        "ece": _safe_float(ece),
        "entropy": _safe_float(entropy),
        "mean_confidence": _safe_float(mean_conf),
        "y_true": y_true.tolist(),
        "y_prob_pos": y_prob_pos,
        "resonance_delta_norms": resonance_delta_norms,
    }


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=3,
    lr=2e-4,
    seed=42,
    label_key="label",
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_metrics = None
    history = {"epochs": []}

    t_total0 = time.time()

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        total = 0
        correct = 0
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch[label_key].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        train_acc = correct / max(1, total)
        train_loss = total_loss / max(1, total)

        val_metrics = evaluate(model, val_loader, device, criterion, label_key=label_key)
        test_metrics = evaluate(model, test_loader, device, criterion, label_key=label_key)

        epoch_time_min = (time.time() - t0) / 60.0

        print(
            f"Epoch {ep:02d}/{epochs} | "
            f"train acc {train_acc:.4f} (loss {train_loss:.4f}) | "
            f"val acc {val_metrics['accuracy']:.4f} | "
            f"test acc {test_metrics['accuracy']:.4f} | "
            f"epoch_time {epoch_time_min:.2f} min"
        )

        history["epochs"].append({
            "epoch": ep,
            "train_loss": _safe_float(train_loss),
            "train_accuracy": _safe_float(train_acc),
            "val": val_metrics,
            "test": test_metrics,
            "epoch_time_min": _safe_float(epoch_time_min),
        })

        if val_metrics["accuracy"] is not None and val_metrics["accuracy"] > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_metrics = {"val": val_metrics, "test": test_metrics}

    total_minutes = (time.time() - t_total0) / 60.0

    return {
        "best_val_accuracy": _safe_float(best_val_acc),
        "best_metrics": best_metrics,
        "total_minutes": _safe_float(total_minutes),
        "history": history,
    }
