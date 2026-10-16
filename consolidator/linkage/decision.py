from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from consolidator.linkage.fellegi_sunter import weight_from_probability
from consolidator.linkage.pairs import EMPTY_PAIRS, encode
from consolidator.linkage.records import RecordId, RecordSet
from consolidator.statistics import MAJORITY

RecordPair = tuple[RecordId, RecordId]
TOLERATED_ERRORS = 0.05


class MatchStatus(IntEnum):
    NON_MATCH = 0
    POSSIBLE = 1
    MATCH = 2


@dataclass(frozen=True)
class DecisionRule:
    match_probability: float = MAJORITY
    false_match_rate: float = TOLERATED_ERRORS
    missed_match_rate: float = TOLERATED_ERRORS

    def __post_init__(self) -> None:
        if not 0.0 < self.match_probability < 1.0:
            raise ValueError("Поріг імовірності збігу має бути в інтервалі (0; 1)")
        if not (0.0 <= self.false_match_rate < 1.0 and 0.0 <= self.missed_match_rate < 1.0):
            raise ValueError("Допустимі частки помилок мають бути в інтервалі [0; 1)")

    def accepted(self, probabilities: np.ndarray) -> np.ndarray:
        return probabilities >= self.match_probability

    def errors(self, probabilities: np.ndarray, contested: np.ndarray | None = None) -> np.ndarray:
        errors = np.where(self.accepted(probabilities), 1.0 - probabilities, probabilities)
        if contested is not None:
            errors = np.where(contested, np.maximum(errors, MAJORITY), errors)
        return errors

    def statuses(self, probabilities: np.ndarray, contested: np.ndarray | None = None) -> np.ndarray:
        statuses = np.full(len(probabilities), MatchStatus.POSSIBLE, dtype=np.int8)
        accepted = self.accepted(probabilities)
        errors = self.errors(probabilities, contested)
        matches = np.flatnonzero(accepted)
        order = matches[np.lexsort((matches, errors[matches]))]
        budget = np.maximum(self.false_match_rate * np.arange(1, len(order) + 1), 1.0)
        statuses[order[: _prefix(np.cumsum(errors[order]), budget)]] = MatchStatus.MATCH
        rejected = np.flatnonzero(~accepted)
        order = rejected[np.lexsort((rejected, errors[rejected]))]
        budget = np.full(len(order), max(self.missed_match_rate * float(probabilities.sum()), 1.0))
        statuses[order[: _prefix(np.cumsum(errors[order]), budget)]] = MatchStatus.NON_MATCH
        return statuses

    def probability_bounds(self, probabilities: np.ndarray, statuses: np.ndarray) -> tuple[float, float]:
        matches = probabilities[statuses == MatchStatus.MATCH]
        rejected = probabilities[statuses == MatchStatus.NON_MATCH]
        upper = float(matches.min()) if len(matches) else 1.0
        lower = float(rejected.max()) if len(rejected) else 0.0
        return upper, lower

    def weight_thresholds(self, probabilities: np.ndarray, statuses: np.ndarray, prior: float) -> tuple[float, float]:
        shift = weight_from_probability(prior)
        upper, lower = self.probability_bounds(probabilities, statuses)
        return weight_from_probability(upper) - shift, weight_from_probability(lower) - shift


def _prefix(cumulative: np.ndarray, budget: np.ndarray) -> int:
    within = np.flatnonzero(cumulative <= budget)
    return int(within[-1]) + 1 if len(within) else 0


@dataclass(frozen=True)
class ClericalReview:
    decisions: Mapping[RecordPair, bool] = field(default_factory=dict)

    @classmethod
    def of(cls, decisions: Iterable[tuple[RecordId, RecordId, bool]]) -> ClericalReview:
        return cls({ordered_pair(left, right): bool(is_match) for left, right, is_match in decisions})

    def __len__(self) -> int:
        return len(self.decisions)

    def on(self, records: RecordSet) -> tuple[np.ndarray, np.ndarray]:
        lookup = records.position_lookup()
        known = [
            (sorted((lookup[left], lookup[right])), is_match)
            for (left, right), is_match in self.decisions.items()
            if left in lookup and right in lookup
        ]
        if not known:
            return EMPTY_PAIRS, np.zeros(0, dtype=bool)
        pairs = np.array([pair for pair, _ in known], dtype=np.int64)
        return pairs, np.array([is_match for _, is_match in known], dtype=bool)


def ordered_pair(left: RecordId, right: RecordId) -> RecordPair:
    left, right = (str(left[0]), str(left[1])), (str(right[0]), str(right[1]))
    return (left, right) if left <= right else (right, left)


def locate(candidates: np.ndarray, pairs: np.ndarray, size: int) -> np.ndarray:
    if len(pairs) == 0 or len(candidates) == 0:
        return np.zeros(len(pairs), dtype=np.int64) - 1
    codes = encode(candidates, size)
    probe = encode(pairs, size)
    positions = np.minimum(np.searchsorted(codes, probe), len(codes) - 1)
    return np.where(codes[positions] == probe, positions, -1)
