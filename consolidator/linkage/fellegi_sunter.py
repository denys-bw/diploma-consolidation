from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

JEFFREYS_PSEUDOCOUNT = 0.5
MAX_ITERATIONS = 500
TOLERANCE = 1e-6
PRIOR_BOUNDS = (1e-9, 1 - 1e-9)
MAX_CONSTRAINT_PASSES = 100
CONSTRAINT_TOLERANCE = 1e-6


@dataclass(frozen=True, eq=False)
class FellegiSunterModel:
    attributes: tuple[str, ...]
    m: tuple[np.ndarray, ...]
    u: tuple[np.ndarray, ...]
    priors: tuple[float, ...]
    pairs: tuple[int, ...]
    iterations: int
    converged: bool
    log_likelihood: float

    @property
    def prior(self) -> float:
        total = sum(self.pairs)
        return float(np.dot(self.priors, self.pairs) / total) if total else self.priors[0]

    @property
    def weights(self) -> tuple[np.ndarray, ...]:
        return tuple(np.log2(m / u) for m, u in zip(self.m, self.u))

    @property
    def parameters(self) -> int:
        return sum(2 * (len(m) - 1) for m in self.m) + sum(1 for prior in self.priors if prior > 0)

    def attribute_weights(self, levels: np.ndarray) -> np.ndarray:
        result = np.zeros(levels.shape, dtype=np.float64)
        for k, weights in enumerate(self.weights):
            column = levels[:, k]
            present = column >= 0
            result[present, k] = weights[column[present]]
        return result

    def match_weights(self, levels: np.ndarray) -> np.ndarray:
        return self.attribute_weights(levels).sum(axis=1)

    def parameters_frame(self) -> pd.DataFrame:
        rows = []
        for attribute, m, u, weights in zip(self.attributes, self.m, self.u, self.weights):
            for level, (m_value, u_value, weight) in enumerate(zip(m, u, weights)):
                rows.append({"attribute": attribute, "level": level, "m": m_value, "u": u_value, "weight": weight})
        return pd.DataFrame(rows, columns=["attribute", "level", "m", "u", "weight"])


class OneToOneConstraint:
    def __init__(self, pairs: np.ndarray, source_codes: np.ndarray, size: int) -> None:
        sources = int(source_codes.max()) + 1 if len(source_codes) else 1
        self.keys = tuple(pairs[:, side] * sources + source_codes[pairs[:, 1 - side]] for side in (0, 1))
        self.slots = size * sources
        sizes = np.bincount(source_codes, minlength=sources)
        self.bound = float(sum(min(sizes[a], sizes[b]) for a in range(sources) for b in range(a + 1, sources)))

    def violated(self, probabilities: np.ndarray) -> bool:
        return float(probabilities.sum()) > self.bound

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        adjusted = probabilities.copy()
        for _ in range(MAX_CONSTRAINT_PASSES):
            excess = 0.0
            for keys in self.keys:
                totals = np.bincount(keys, weights=adjusted, minlength=self.slots)
                excess = max(excess, float(totals.max(initial=0.0)) - 1.0)
                adjusted *= np.minimum(1.0, 1.0 / np.maximum(totals, np.finfo(float).tiny))[keys]
            if excess <= CONSTRAINT_TOLERANCE:
                break
        return adjusted


def probability_from_weight(weight: np.ndarray | float) -> np.ndarray:
    return expit(np.asarray(weight, dtype=np.float64) * np.log(2.0))


def weight_from_probability(probability: np.ndarray | float) -> np.ndarray | float:
    probability = np.asarray(probability, dtype=np.float64)
    with np.errstate(divide="ignore"):
        result = np.log2(probability) - np.log2(1.0 - probability)
    return float(result) if result.ndim == 0 else result


def posterior(weights: np.ndarray, prior: np.ndarray | float) -> np.ndarray:
    return probability_from_weight(weights + weight_from_probability(prior))


def estimate_prior(
    weights: np.ndarray, initial: float = 0.5, max_iterations: int = MAX_ITERATIONS, tolerance: float = TOLERANCE
) -> float:
    if len(weights) == 0:
        return initial
    prior = float(np.clip(initial, *PRIOR_BOUNDS))
    for _ in range(max_iterations):
        updated = float(np.clip(posterior(weights, prior).mean(), *PRIOR_BOUNDS))
        if abs(updated - prior) < tolerance:
            return updated
        prior = updated
    return prior


def estimate_group_priors(
    weights: np.ndarray, groups: np.ndarray, initial: Sequence[float], closed: Collection[int] = ()
) -> tuple[float, ...]:
    return tuple(
        0.0 if group in closed else estimate_prior(weights[groups == group], start)
        for group, start in enumerate(initial)
    )


def pair_priors(groups: np.ndarray, priors: Sequence[float]) -> np.ndarray:
    return np.asarray(priors, dtype=np.float64)[groups]


def frequency_adjustments(frequencies: np.ndarray) -> np.ndarray:
    adjustment = np.zeros(frequencies.shape, dtype=np.float64)
    for k in range(frequencies.shape[1]):
        column = frequencies[:, k].astype(np.float64)
        known = np.isfinite(column) & (column > 0)
        if known.any():
            typical = float(np.exp(np.log(column[known]).mean()))
            adjustment[known, k] = np.log2(typical / column[known])
    return adjustment


def frequency_adjustment(frequencies: np.ndarray) -> np.ndarray:
    return frequency_adjustments(frequencies).sum(axis=1)


def fit_fellegi_sunter(
    levels: np.ndarray,
    n_levels: Sequence[int],
    attributes: Sequence[str],
    prior: float,
    groups: np.ndarray | None = None,
    closed: Collection[int] = (),
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
    constraint: OneToOneConstraint | None = None,
) -> FellegiSunterModel:
    if levels.ndim != 2 or levels.shape[1] == 0:
        raise ValueError("Для моделі Фелегі–Зунтера потрібен хоча б один атрибут порівняння")
    if len(levels) == 0:
        raise ValueError("Немає пар-кандидатів для оцінювання моделі")
    groups = np.zeros(len(levels), dtype=np.int64) if groups is None else np.asarray(groups, dtype=np.int64)
    n_groups = int(groups.max()) + 1
    keyed = np.column_stack([levels.astype(np.int64), groups])
    patterns, inverse, counts = np.unique(keyed, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    counts = counts.astype(np.float64)
    pattern_groups = patterns[:, -1]
    codes = patterns[:, :-1]
    open_pattern = ~np.isin(pattern_groups, list(closed))
    group_sizes = np.bincount(pattern_groups, weights=counts, minlength=n_groups)
    matched = _initial_matches(codes, n_levels, counts, open_pattern, prior)
    priors, m, u = _maximize(codes, n_levels, counts, matched, pattern_groups, n_groups, group_sizes, closed)
    converged = False
    iteration = 0
    log_likelihood = previous = -np.inf
    for iteration in range(1, max_iterations + 1):
        log_m = _log_probability(codes, m)
        log_u = _log_probability(codes, u)
        clipped = np.clip(priors[pattern_groups], *PRIOR_BOUNDS)
        log_prior, log_rest = np.log(clipped), np.log1p(-clipped)
        responsibility = np.where(open_pattern, expit(log_prior - log_rest + log_m - log_u), 0.0)
        mixture = np.where(open_pattern, np.logaddexp(log_prior + log_m, log_rest + log_u), log_u)
        log_likelihood = float((counts * mixture).sum())
        if constraint is None:
            matched = counts * responsibility
        else:
            matched = np.bincount(inverse, weights=constraint.apply(responsibility[inverse]), minlength=len(patterns))
        new_priors, new_m, new_u = _maximize(
            codes, n_levels, counts, matched, pattern_groups, n_groups, group_sizes, closed
        )
        change = max(
            float(np.abs(new_priors - priors).max()),
            max(float(np.abs(a - b).max()) for a, b in zip(new_m, m)),
            max(float(np.abs(a - b).max()) for a, b in zip(new_u, u)),
        )
        priors, m, u = new_priors, new_m, new_u
        improvement = log_likelihood - previous
        previous = log_likelihood
        if change < tolerance or 0 <= improvement < tolerance * abs(log_likelihood):
            converged = True
            break
    return FellegiSunterModel(
        tuple(attributes),
        tuple(m),
        tuple(u),
        tuple(float(value) for value in priors),
        tuple(int(size) for size in group_sizes),
        iteration,
        converged,
        log_likelihood,
    )


def fit_non_matches(
    levels: np.ndarray,
    n_levels: Sequence[int],
    attributes: Sequence[str],
    m: Sequence[np.ndarray],
    prior: float,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
) -> FellegiSunterModel:
    codes, counts = np.unique(levels.astype(np.int64), axis=0, return_counts=True)
    counts = counts.astype(np.float64)
    log_m = _log_probability(codes, m)
    u = _marginals(codes, counts, n_levels)
    share = float(np.clip(prior, *PRIOR_BOUNDS))
    converged = False
    iteration = 0
    log_likelihood = previous = -np.inf
    for iteration in range(1, max_iterations + 1):
        log_u = _log_probability(codes, u)
        responsibility = expit(np.log(share) - np.log1p(-share) + log_m - log_u)
        log_likelihood = float((counts * np.logaddexp(np.log(share) + log_m, np.log1p(-share) + log_u)).sum())
        new_share = float(np.clip((counts * responsibility).sum() / counts.sum(), *PRIOR_BOUNDS))
        new_u = _marginals(codes, counts * (1.0 - responsibility), n_levels)
        change = max(abs(new_share - share), max(float(np.abs(a - b).max()) for a, b in zip(new_u, u)))
        share, u = new_share, new_u
        improvement = log_likelihood - previous
        previous = log_likelihood
        if change < tolerance or 0 <= improvement < tolerance * abs(log_likelihood):
            converged = True
            break
    return FellegiSunterModel(
        tuple(attributes), tuple(m), tuple(u), (share,), (int(counts.sum()),), iteration, converged, log_likelihood
    )


def independent_log_likelihood(levels: np.ndarray, n_levels: Sequence[int]) -> float:
    codes, counts = np.unique(levels.astype(np.int64), axis=0, return_counts=True)
    counts = counts.astype(np.float64)
    return float((counts * _log_probability(codes, _marginals(codes, counts, n_levels))).sum())


def _marginals(codes: np.ndarray, weights: np.ndarray, n_levels: Sequence[int]) -> list[np.ndarray]:
    distributions = []
    for k, size in enumerate(n_levels):
        present = codes[:, k] >= 0
        distributions.append(_distribution(codes[present, k], weights[present], size))
    return distributions


def _initial_matches(
    codes: np.ndarray, n_levels: Sequence[int], counts: np.ndarray, open_pattern: np.ndarray, prior: float
) -> np.ndarray:
    scale = np.maximum(np.asarray(n_levels, dtype=np.float64) - 1.0, 1.0)
    present = codes >= 0
    relative = np.where(present, codes / scale, 0.0)
    score = relative.sum(axis=1) / np.maximum(present.sum(axis=1), 1)
    score = np.where(open_pattern, score, -np.inf)
    order = np.lexsort((np.arange(len(score)), -score))
    budget = prior * counts[open_pattern].sum()
    before = np.cumsum(counts[order]) - counts[order]
    taken = np.clip(budget - before, 0.0, counts[order])
    matched = np.zeros(len(counts))
    matched[order] = taken
    return np.where(open_pattern, matched, 0.0)


def _maximize(
    codes: np.ndarray,
    n_levels: Sequence[int],
    counts: np.ndarray,
    matched: np.ndarray,
    pattern_groups: np.ndarray,
    n_groups: int,
    group_sizes: np.ndarray,
    closed: Collection[int],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    group_matches = np.bincount(pattern_groups, weights=matched, minlength=n_groups)
    is_closed = np.isin(np.arange(n_groups), list(closed))
    priors = np.where(is_closed, 0.0, np.clip(group_matches / np.maximum(group_sizes, 1.0), *PRIOR_BOUNDS))
    m, u = [], []
    for k, size in enumerate(n_levels):
        column = codes[:, k]
        present = column >= 0
        u_k = _distribution(column[present], (counts - matched)[present], size)
        m_k = _distribution(column[present], matched[present], size)
        u.append(u_k)
        m.append(_monotone(m_k, u_k))
    return priors, m, u


def _distribution(levels: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    mass = np.bincount(levels, weights=weights, minlength=size)
    return (mass + JEFFREYS_PSEUDOCOUNT) / (mass.sum() + JEFFREYS_PSEUDOCOUNT * size)


def _log_probability(codes: np.ndarray, probabilities: Sequence[np.ndarray]) -> np.ndarray:
    total = np.zeros(len(codes), dtype=np.float64)
    for k, values in enumerate(probabilities):
        column = codes[:, k]
        present = column >= 0
        total[present] += np.log(values)[column[present]]
    return total


def _monotone(m: np.ndarray, u: np.ndarray) -> np.ndarray:
    log_ratio = np.log(m / u)
    fitted = _isotonic(log_ratio, m + u)
    adjusted = u * np.exp(fitted)
    return adjusted / adjusted.sum()


def _isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    blocks: list[list[float]] = []
    for value, weight in zip(values, weights):
        blocks.append([float(value) * float(weight), float(weight), 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, mass, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += mass
            blocks[-1][2] += count
    fitted: list[float] = []
    for total, mass, count in blocks:
        fitted.extend([total / mass] * int(count))
    return np.array(fitted, dtype=np.float64)
