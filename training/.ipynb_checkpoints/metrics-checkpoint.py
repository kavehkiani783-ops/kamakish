```python
# metrics.py

import time
import math
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef
)


# -----------------------------------------------------------
# Performance Metrics
# -----------------------------------------------------------

def compute_classification_metrics(y_true, logits, num_classes):

    y_true = y_true.cpu().numpy()
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    preds = np.argmax(probs, axis=1)

    results = {}

    results["accuracy"] = accuracy_score(y_true, preds)
    results["macro_f1"] = f1_score(y_true, preds, average="macro")
    results["weighted_f1"] = f1_score(y_true, preds, average="weighted")
    results["balanced_accuracy"] = balanced_accuracy_score(y_true, preds)
    results["kappa"] = cohen_kappa_score(y_true, preds)
    results["mcc"] = matthews_corrcoef(y_true, preds)

    if num_classes == 2:
        try:
            results["auroc"] = roc_auc_score(y_true, probs[:, 1])
            results["auprc"] = average_precision_score(y_true, probs[:, 1])
        except:
            results["auroc"] = None
            results["auprc"] = None
    else:
        try:
            results["auroc"] = roc_auc_score(y_true, probs, multi_class="ovr")
            results["auprc"] = None
        except:
            results["auroc"] = None
            results["auprc"] = None

    return results


# -----------------------------------------------------------
# Calibration / Uncertainty
# -----------------------------------------------------------

def compute_nll(y_true, logits):
    return F.cross_entropy(logits, y_true).item()


def compute_brier_score(y_true, logits):
    probs = F.softmax(logits, dim=1)
    y_onehot = F.one_hot(y_true, num_classes=probs.shape[1]).float()
    return torch.mean(torch.sum((probs - y_onehot) ** 2, dim=1)).item()


def compute_entropy(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
    return torch.mean(entropy).item()


def compute_mean_confidence(logits):
    probs = F.softmax(logits, dim=1)
    confidence = torch.max(probs, dim=1)[0]
    return torch.mean(confidence).item()


def compute_ece(y_true, logits, n_bins=15):

    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, 1)

    accuracies = predictions.eq(y_true)

    ece = torch.zeros(1, device=logits.device)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        in_bin = confidences.gt(lower) * confidences.le(upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


# -----------------------------------------------------------
# Efficiency Metrics
# -----------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 ** 2)


def measure_throughput(model, dataloader, device, max_batches=20):

    model.eval()
    torch.cuda.empty_cache()

    total_tokens = 0
    total_time = 0.0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            x, y = batch
            x = x.to(device)

            torch.cuda.synchronize()
            start = time.time()

            _ = model(x)

            torch.cuda.synchronize()
            end = time.time()

            total_time += (end - start)
            total_tokens += x.numel()

    tokens_per_sec = total_tokens / total_time
    ms_per_batch = (total_time / max_batches) * 1000

    return {
        "tokens_per_sec": tokens_per_sec,
        "ms_per_batch": ms_per_batch
    }


def peak_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


# -----------------------------------------------------------
# Resonance Dynamics Metrics
# -----------------------------------------------------------

def compute_delta_norms(memory_states):
    """
    memory_states: list of tensors [M0, M1, ..., MK]
    """
    deltas = []
    for i in range(1, len(memory_states)):
        delta = memory_states[i] - memory_states[i - 1]
        deltas.append(torch.norm(delta).item())
    return deltas


def steps_to_converge(delta_norms, threshold):
    for i, d in enumerate(delta_norms):
        if d < threshold:
            return i + 1
    return None


def convergence_rate(delta_norms):
    if len(delta_norms) < 2:
        return None
    ratios = []
    for i in range(1, len(delta_norms)):
        if delta_norms[i - 1] > 0:
            ratios.append(delta_norms[i] / delta_norms[i - 1])
    if len(ratios) == 0:
        return None
    return sum(ratios) / len(ratios)
```
