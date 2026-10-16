from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from consolidator.linkage.records import RecordSet
from consolidator.profiling.numeric import numeric_values
from consolidator.profiling.types import ColumnType

NO_VALUE = -1


@dataclass(frozen=True, eq=False)
class FusionContext:
    records: RecordSet
    entity_ids: np.ndarray
    n_entities: int
    order: np.ndarray
    keys: dict[str, np.ndarray]
    truth_scores: dict[str, np.ndarray]

    def present(self, attribute: str) -> np.ndarray:
        return self.keys[attribute] != ""

    def select(self, scores: np.ndarray, eligible: np.ndarray, tiebreak: np.ndarray | None = None) -> np.ndarray:
        chosen = np.full(self.n_entities, NO_VALUE, dtype=np.int64)
        members = self.order[eligible[self.order]]
        if len(members) == 0:
            return chosen
        secondary = np.zeros(len(members)) if tiebreak is None else -tiebreak[members]
        ranked = members[np.lexsort((np.arange(len(members)), secondary, -scores[members], self.entity_ids[members]))]
        entities = self.entity_ids[ranked]
        first = np.r_[True, entities[1:] != entities[:-1]]
        chosen[entities[first]] = ranked[first]
        return chosen


class ConflictResolver(ABC):
    key = ""
    uses_source_accuracy = False

    @abstractmethod
    def choose(self, attribute: str, context: FusionContext) -> np.ndarray:
        raise NotImplementedError


class ScoringResolver(ConflictResolver):
    def choose(self, attribute: str, context: FusionContext) -> np.ndarray:
        return context.select(
            self.scores(attribute, context), context.present(attribute), self.tiebreak(attribute, context)
        )

    @abstractmethod
    def scores(self, attribute: str, context: FusionContext) -> np.ndarray:
        raise NotImplementedError

    def tiebreak(self, attribute: str, context: FusionContext) -> np.ndarray | None:
        if not context.records.types.get(attribute, ColumnType.TEXT).is_textual:
            return None
        values = context.records.values(attribute)
        return np.fromiter((len(str(value)) for value in values), dtype=np.float64, count=len(values))


class WeightedVote(ScoringResolver):
    key = "weighted_vote"
    uses_source_accuracy = True

    def scores(self, attribute: str, context: FusionContext) -> np.ndarray:
        return context.truth_scores[attribute]


class MajorityVote(ScoringResolver):
    key = "majority_vote"

    def scores(self, attribute: str, context: FusionContext) -> np.ndarray:
        codes = pd.factorize(context.keys[attribute])[0].astype(np.int64)
        combined = context.entity_ids * (int(codes.max()) + 1 if len(codes) else 1) + codes
        _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
        return counts[inverse.ravel()].astype(np.float64)


class Longest(ScoringResolver):
    key = "longest"

    def scores(self, attribute: str, context: FusionContext) -> np.ndarray:
        values = context.records.values(attribute)
        return np.fromiter((len(str(value)) for value in values), dtype=np.float64, count=len(values))


class SourcePriority(ScoringResolver):
    key = "source_priority"

    def scores(self, attribute: str, context: FusionContext) -> np.ndarray:
        return np.zeros(context.records.size, dtype=np.float64)

    def tiebreak(self, attribute: str, context: FusionContext) -> np.ndarray | None:
        return None


class Median(ConflictResolver):
    key = "median"

    def choose(self, attribute: str, context: FusionContext) -> np.ndarray:
        column_type = context.records.types.get(attribute, ColumnType.NUMBER)
        numbers = numeric_values(pd.Series(context.records.values(attribute), dtype=object), column_type)
        finite = context.present(attribute) & np.isfinite(numbers)
        chosen = np.full(context.n_entities, NO_VALUE, dtype=np.int64)
        members = context.order[finite[context.order]]
        if len(members):
            entity_ids = context.entity_ids
            ranked = members[np.lexsort((np.arange(len(members)), numbers[members], entity_ids[members]))]
            entities = entity_ids[ranked]
            starts = np.flatnonzero(np.r_[True, entities[1:] != entities[:-1]])
            sizes = np.diff(np.r_[starts, len(ranked)])
            chosen[entities[starts]] = ranked[starts + (sizes - 1) // 2]
        unresolved = chosen == NO_VALUE
        if unresolved.any():
            chosen[unresolved] = MajorityVote().choose(attribute, context)[unresolved]
        return chosen


RESOLVERS: dict[str, ConflictResolver] = {
    resolver.key: resolver for resolver in (WeightedVote(), MajorityVote(), Median(), Longest(), SourcePriority())
}
DEFAULT_RESOLVERS = {
    ColumnType.NUMBER: RESOLVERS[Median.key],
    ColumnType.MEASURE: RESOLVERS[Median.key],
}


def default_resolver(column_type: ColumnType) -> ConflictResolver:
    return DEFAULT_RESOLVERS.get(column_type, RESOLVERS[WeightedVote.key])
