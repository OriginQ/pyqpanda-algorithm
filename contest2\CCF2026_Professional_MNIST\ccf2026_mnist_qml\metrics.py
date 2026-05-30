import math
from typing import Dict, List, Sequence


def binary_log_loss(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    eps = 1e-9
    total = 0.0
    for y, p in zip(y_true, probabilities):
        p = min(1.0 - eps, max(eps, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / max(1, len(y_true))


def classification_metrics(y_true: Sequence[int], probabilities: Sequence[float]) -> Dict[str, float]:
    pred = [1 if p >= 0.5 else 0 for p in probabilities]
    tp = sum(1 for y, p in zip(y_true, pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, pred) if y == 1 and p == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
        "auc": roc_auc(y_true, probabilities),
        "log_loss": binary_log_loss(y_true, probabilities),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(1 for y in y_true if y == 1)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ordered = sorted(zip(scores, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_, y) in enumerate(ordered, start=1):
        if y == 1:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

