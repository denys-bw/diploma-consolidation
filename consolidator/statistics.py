from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit

MAJORITY = 0.5
SIGNIFICANCE = 0.05
PRIOR_SCALE = 1.0
MAX_OPTIMIZER_ITERATIONS = 1000


def is_majority(share: float) -> bool:
    return share > MAJORITY


def sturges_bins(count: int) -> int:
    return math.ceil(math.log2(count)) + 1 if count > 1 else 1


def holm(p_values: np.ndarray, alpha: float = SIGNIFICANCE) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values, kind="stable")
    accepted = np.zeros(len(p_values), dtype=bool)
    for rank, index in enumerate(order):
        if p_values[index] > alpha / (len(p_values) - rank):
            break
        accepted[index] = True
    return accepted


@dataclass(frozen=True)
class SignedRankTest:
    pairs: int
    statistic: float
    p_value: float


def signed_rank_greater(differences: np.ndarray) -> SignedRankTest:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences) & (differences != 0.0)]
    if len(differences) == 0:
        return SignedRankTest(0, 0.0, 1.0)
    result = stats.wilcoxon(differences, alternative="greater")
    return SignedRankTest(len(differences), float(result.statistic), float(result.pvalue))


def weighted_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.clip(np.asarray(positive, dtype=np.float64), 0.0, 1.0)
    negative = 1.0 - positive
    total_positive, total_negative = positive.sum(), negative.sum()
    if total_positive <= 0 or total_negative <= 0:
        return 0.5
    values, inverse = np.unique(scores, return_inverse=True)
    positive_mass = np.bincount(inverse, weights=positive, minlength=len(values))
    negative_mass = np.bincount(inverse, weights=negative, minlength=len(values))
    below = np.cumsum(negative_mass) - negative_mass
    wins = float((positive_mass * (below + 0.5 * negative_mass)).sum())
    return wins / (total_positive * total_negative)


class BayesianLogistic:
    def __init__(self, prior_scale: float = PRIOR_SCALE) -> None:
        self.prior_scale = prior_scale
        self.centre = np.zeros(0)
        self.scale = np.ones(0)
        self.coefficients = np.zeros(0)
        self.intercept = 0.0

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        offset: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        reference: np.ndarray | None = None,
    ) -> BayesianLogistic:
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        basis = features if reference is None else np.asarray(reference, dtype=np.float64)
        self.centre = basis.mean(axis=0) if len(basis) else np.zeros(features.shape[1])
        spread = basis.std(axis=0) if len(basis) else np.ones(features.shape[1])
        self.scale = np.where(spread > 0, spread, 1.0)
        design = (features - self.centre) / self.scale
        offset = np.zeros(len(targets)) if offset is None else np.asarray(offset, dtype=np.float64)
        weights = np.ones(len(targets)) if weights is None else np.asarray(weights, dtype=np.float64)
        start = np.zeros(design.shape[1] + 1)
        result = minimize(
            self._objective,
            start,
            args=(design, targets, offset, weights),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": MAX_OPTIMIZER_ITERATIONS},
        )
        self.intercept = float(result.x[0])
        self.coefficients = result.x[1:]
        return self

    def decision_function(self, features: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        design = (np.asarray(features, dtype=np.float64) - self.centre) / self.scale
        logits = self.intercept + design @ self.coefficients
        return logits if offset is None else logits + np.asarray(offset, dtype=np.float64)

    def predict_proba(self, features: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        return expit(self.decision_function(features, offset))

    def _objective(
        self,
        parameters: np.ndarray,
        design: np.ndarray,
        targets: np.ndarray,
        offset: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        intercept, coefficients = parameters[0], parameters[1:]
        logits = offset + intercept + design @ coefficients
        residual = weights * (expit(logits) - targets)
        precision = self._precision(len(parameters))
        loss = float((weights * (np.logaddexp(0.0, logits) - targets * logits)).sum())
        loss += 0.5 * float((precision * parameters * parameters).sum())
        gradient = np.empty_like(parameters)
        gradient[0] = residual.sum()
        gradient[1:] = design.T @ residual
        gradient += precision * parameters
        return loss, gradient

    def _precision(self, size: int) -> np.ndarray:
        variance = np.full(size, self.prior_scale * self.prior_scale)
        if size > 1:
            variance[1:] /= size - 1
        return 1.0 / variance
