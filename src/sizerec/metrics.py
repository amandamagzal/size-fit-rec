"""
Metrics for 4-class fit_outcome classification.
"""

from typing import Dict, Tuple
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Confusion matrix [num_classes, num_classes] with counts (row=true, col=pred)."""
    cm = np.zeros((num_classes, num_classes), dtype = np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def precision_recall_f1_per_class(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, Dict[str, float]]:
    """Per-class precision/recall/F1 and macro averages."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    eps = 1e-12
    out = {}
    precisions, recalls, f1s = [], [], []
    for k in range(num_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        out[str(k)] = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    out["macro"] = {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }
    return out
