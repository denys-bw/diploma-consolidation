from __future__ import annotations

from dataclasses import dataclass

import numpy as np

JEFFREYS_PSEUDOCOUNT = 0.5
PRIOR_ACCURACY = 0.5
MAX_ITERATIONS = 100
TOLERANCE = 1e-6


@dataclass(frozen=True, eq=False)
class Claims:
    items: np.ndarray
    values: np.ndarray
    sources: np.ndarray
    n_sources: int

    def __post_init__(self) -> None:
        if not len(self.items) == len(self.values) == len(self.sources):
            raise ValueError("Твердження мають однакову довжину масивів об'єктів, значень і джерел")

    @property
    def size(self) -> int:
        return len(self.items)


@dataclass(frozen=True, eq=False)
class TruthEstimate:
    accuracy: np.ndarray
    scores: np.ndarray
    probabilities: np.ndarray
    iterations: int
    converged: bool


def discover_truth(claims: Claims) -> TruthEstimate:
    if claims.size == 0:
        return TruthEstimate(np.full(claims.n_sources, PRIOR_ACCURACY), np.empty(0), np.empty(0), 0, True)
    pairs, group = np.unique(np.stack([claims.items, claims.values], axis=1), axis=0, return_inverse=True)
    group = group.ravel()
    _, pair_item = np.unique(pairs[:, 0], return_inverse=True)
    pair_item = pair_item.ravel()
    claim_item = pair_item[group]
    false_values = np.maximum(np.bincount(pair_item) - 1, 1)
    log_false = np.log(false_values)[claim_item]
    overlapping = np.bincount(claim_item)[claim_item] >= 2
    evidence = np.bincount(claims.sources[overlapping], minlength=claims.n_sources).astype(np.float64)
    votes = np.bincount(group, minlength=len(pairs)).astype(np.float64)
    pair_probabilities = votes / np.bincount(pair_item, weights=votes)[pair_item]
    probabilities = pair_probabilities[group]
    accuracy = _accuracy(claims, overlapping, probabilities, evidence)
    scores = np.log(votes)[group]
    converged = False
    iterations = 0
    for iterations in range(1, MAX_ITERATIONS + 1):
        weights = log_false + np.log(accuracy / (1.0 - accuracy))[claims.sources]
        pair_scores = np.bincount(group, weights=weights, minlength=len(pairs))
        pair_probabilities = _softmax_by_item(pair_scores, pair_item)
        scores, probabilities = pair_scores[group], pair_probabilities[group]
        updated = _accuracy(claims, overlapping, probabilities, evidence)
        change = float(np.abs(updated - accuracy).max())
        accuracy = updated
        if change < TOLERANCE:
            converged = True
            break
    return TruthEstimate(accuracy, scores, probabilities, iterations, converged)


def _accuracy(claims: Claims, overlapping: np.ndarray, probabilities: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    support = np.bincount(claims.sources[overlapping], weights=probabilities[overlapping], minlength=claims.n_sources)
    return (support + JEFFREYS_PSEUDOCOUNT) / (evidence + 2.0 * JEFFREYS_PSEUDOCOUNT)


def _softmax_by_item(scores: np.ndarray, items: np.ndarray) -> np.ndarray:
    top = np.full(int(items.max()) + 1, -np.inf)
    np.maximum.at(top, items, scores)
    exponent = np.exp(scores - top[items])
    return exponent / np.bincount(items, weights=exponent)[items]
