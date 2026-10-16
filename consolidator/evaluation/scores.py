from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    precision: float
    recall: float
    f1: float


def best_threshold(scores: np.ndarray, labels: np.ndarray, total_positives: int | None = None) -> ThresholdChoice:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum()) if total_positives is None else int(total_positives)
    if len(scores) == 0 or positives == 0:
        return ThresholdChoice(1.0, 0.0, 0.0, 0.0)
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    true_positives = np.cumsum(labels[order])
    predicted = np.arange(1, len(scores) + 1)
    last = np.r_[np.flatnonzero(np.diff(ordered) != 0), len(scores) - 1]
    precision = true_positives[last] / predicted[last]
    recall = true_positives[last] / positives
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(precision), where=total > 0)
    best = int(np.argmax(f1))
    return ThresholdChoice(float(ordered[last[best]]), float(precision[best]), float(recall[best]), float(f1[best]))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if not labels.any():
        return 0.0
    return float(average_precision_score(labels, np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-1.0)))
