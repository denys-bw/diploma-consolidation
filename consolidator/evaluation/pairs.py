from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from consolidator.linkage.pairs import canonical_pairs, encode


@dataclass(frozen=True)
class MatchMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def found(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def expected(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float:
        return self.true_positives / self.found if self.found else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.expected if self.expected else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "tp": self.true_positives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class BlockingMetrics:
    candidates: int
    total_pairs: int
    found_matches: int
    matches: int

    @property
    def reduction_ratio(self) -> float:
        return 1.0 - self.candidates / self.total_pairs if self.total_pairs else 0.0

    @property
    def pairs_completeness(self) -> float:
        return self.found_matches / self.matches if self.matches else 0.0

    @property
    def pairs_quality(self) -> float:
        return self.found_matches / self.candidates if self.candidates else 0.0

    @property
    def f_measure(self) -> float:
        rr, pc = self.reduction_ratio, self.pairs_completeness
        return 2 * rr * pc / (rr + pc) if rr + pc else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "candidates": self.candidates,
            "total_pairs": self.total_pairs,
            "rr": self.reduction_ratio,
            "pc": self.pairs_completeness,
            "pq": self.pairs_quality,
            "f": self.f_measure,
        }


def evaluate_positions(found: np.ndarray, gold: np.ndarray, size: int) -> MatchMetrics:
    found_codes = np.unique(encode(canonical_pairs(found), size))
    gold_codes = np.unique(encode(canonical_pairs(gold), size))
    true_positives = int(np.isin(found_codes, gold_codes).sum())
    return MatchMetrics(true_positives, len(found_codes) - true_positives, len(gold_codes) - true_positives)


def evaluate_blocking(candidates: np.ndarray, gold: np.ndarray, size: int, total_pairs: int) -> BlockingMetrics:
    candidate_codes = np.unique(encode(canonical_pairs(candidates), size))
    gold_codes = np.unique(encode(canonical_pairs(gold), size))
    found = int(np.isin(gold_codes, candidate_codes).sum())
    return BlockingMetrics(len(candidate_codes), total_pairs, found, len(gold_codes))


def gold_labels(candidates: np.ndarray, gold: np.ndarray, size: int) -> np.ndarray:
    return np.isin(encode(canonical_pairs(candidates), size), encode(canonical_pairs(gold), size))
