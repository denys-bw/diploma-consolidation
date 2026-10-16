from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum

import numpy as np

from consolidator.linkage.comparison import ComparisonMatrix
from consolidator.linkage.fellegi_sunter import fit_fellegi_sunter, fit_non_matches, independent_log_likelihood
from consolidator.linkage.records import PairScope, RecordSet

WITHIN, ACROSS = 0, 1


class DuplicatePolicy(str, Enum):
    AUTO = "auto"
    EVERYWHERE = "everywhere"
    ACROSS = "across"
    ACROSS_CLEAN = "across_clean"


@dataclass(frozen=True)
class SourceStructure:
    scope: PairScope
    clean: bool
    decided: bool = False
    evidence: float | None = None

    @classmethod
    def declared(cls, policy: DuplicatePolicy, sources: int) -> SourceStructure | None:
        if sources <= 1:
            return cls(PairScope.ALL, False)
        if policy is DuplicatePolicy.ACROSS:
            return cls(PairScope.CROSS, False)
        if policy is DuplicatePolicy.ACROSS_CLEAN:
            return cls(PairScope.CROSS, True)
        if policy is DuplicatePolicy.EVERYWHERE:
            return cls(PairScope.ALL, False)
        return None

    @classmethod
    def detected(
        cls, comparison: ComparisonMatrix, groups: np.ndarray, reviewed: np.ndarray, prior: float
    ) -> SourceStructure:
        within = groups == WITHIN
        if (reviewed[within] == 1).any() or not (groups == ACROSS).any():
            return cls(PairScope.ALL, False, True, math.inf)
        if not within.any():
            return cls(PairScope.CROSS, True, True, None)
        evidence = internal_duplicates_evidence(comparison, groups, prior)
        if evidence > 0:
            return cls(PairScope.ALL, False, True, evidence)
        return cls(PairScope.CROSS, True, True, evidence)


def pair_groups(records: RecordSet, pairs: np.ndarray) -> np.ndarray:
    codes = records.source_codes
    return np.where(codes[pairs[:, 0]] == codes[pairs[:, 1]], WITHIN, ACROSS).astype(np.int64)


def mixed_groups(groups: np.ndarray, closed: Collection[int] = ()) -> bool:
    present = set(np.unique(groups).tolist())
    return WITHIN in present and ACROSS in present and ACROSS not in closed


def internal_duplicates_evidence(comparison: ComparisonMatrix, groups: np.ndarray, prior: float) -> float:
    levels, within = comparison.levels, groups == WITHIN
    across = fit_fellegi_sunter(levels[groups == ACROSS], comparison.n_levels, comparison.attributes, prior=prior)
    inside = fit_non_matches(levels[within], comparison.n_levels, comparison.attributes, across.m, prior)
    alone = independent_log_likelihood(levels[within], comparison.n_levels)
    return 2.0 * (inside.log_likelihood - alone) - math.log(int(within.sum()))
