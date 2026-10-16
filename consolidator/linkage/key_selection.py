from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from consolidator.linkage.blocking import Blocker, KeyBlocking, SortedNeighbourhood, TokenBlocking, scale_neighbours
from consolidator.linkage.comparison import comparable_attributes
from consolidator.linkage.keys import BlockingKey, KeyMethod
from consolidator.linkage.pairs import EMPTY_PAIRS, canonical_pairs, count_within_groups, encode
from consolidator.linkage.records import PairScope, RecordSet
from consolidator.profiling.types import ColumnType
from consolidator.values.phones import MIN_DIGITS
from consolidator.values.text import comparison_keys

KEY_METHODS = {
    ColumnType.TEXT: ((KeyMethod.FIRST_TOKEN, 0), (KeyMethod.SOUNDEX, 0), (KeyMethod.PREFIX, None)),
    ColumnType.CATEGORY: ((KeyMethod.EXACT, 0),),
    ColumnType.CODE: ((KeyMethod.EXACT, 0), (KeyMethod.PREFIX, None)),
    ColumnType.IDENTIFIER: ((KeyMethod.EXACT, 0),),
    ColumnType.EMAIL: ((KeyMethod.EXACT, 0),),
    ColumnType.URL: ((KeyMethod.EXACT, 0),),
    ColumnType.PHONE: ((KeyMethod.DIGITS, MIN_DIGITS),),
    ColumnType.YEAR: ((KeyMethod.YEAR, 0),),
    ColumnType.DATE: ((KeyMethod.YEAR, 0),),
    ColumnType.DATETIME: ((KeyMethod.YEAR, 0),),
}


@dataclass(frozen=True)
class KeyAssessment:
    key: BlockingKey
    candidates: int
    reduction_ratio: float
    estimated_completeness: float


@dataclass(frozen=True)
class KeySuggestion:
    keys: tuple[BlockingKey, ...]
    assessments: tuple[KeyAssessment, ...]
    candidates: int
    estimated_completeness: float
    budget: int


class KeySelector:
    def __init__(self, records: RecordSet, scope: PairScope) -> None:
        self.records = records
        self.scope = scope
        self.budget = records.size * scale_neighbours(records)
        self.reference = reference_matches(records, scope)
        self._groups: dict[BlockingKey, np.ndarray] = {}

    def candidate_keys(self) -> list[BlockingKey]:
        keys = []
        for attribute in comparable_attributes(self.records, self.scope):
            methods = KEY_METHODS.get(self.records.types[attribute], ())
            for method, length in methods:
                if length is None:
                    keys.extend(BlockingKey.of(attribute, method, size) for size in self._prefix_lengths(attribute))
                else:
                    keys.append(BlockingKey.of(attribute, method, length))
        return keys

    def assess(self, key: BlockingKey) -> KeyAssessment:
        candidates = count_within_groups(self.groups(key), self.records.source_codes, self.scope)
        total = self.records.total_pairs(self.scope)
        completeness = float(self.covers(key).mean()) if len(self.reference) else 0.0
        return KeyAssessment(key, candidates, 1.0 - candidates / total if total else 0.0, completeness)

    def suggest(self) -> KeySuggestion:
        singles = [self.assess(key) for key in self.candidate_keys()]
        composites = [
            self.assess(a.key.combined(b.key))
            for a, b in combinations(singles, 2)
            if a.key.parts[0].attribute != b.key.parts[0].attribute
        ]
        assessments = sorted(singles + composites, key=_rank)
        affordable = [item for item in assessments if 0 < item.candidates <= self.budget]
        chosen: list[KeyAssessment] = []
        covered = np.zeros(len(self.reference), dtype=bool)
        spent = 0
        while True:
            best, best_gain = None, 0
            for item in affordable:
                if item in chosen or spent + item.candidates > self.budget:
                    continue
                gain = int((self.covers(item.key) & ~covered).sum())
                if gain > best_gain:
                    best, best_gain = item, gain
            if best is None:
                break
            chosen.append(best)
            covered |= self.covers(best.key)
            spent += best.candidates
        return KeySuggestion(
            keys=tuple(item.key for item in chosen),
            assessments=tuple(assessments),
            candidates=spent,
            estimated_completeness=float(covered.mean()) if len(self.reference) else 0.0,
            budget=self.budget,
        )

    def groups(self, key: BlockingKey) -> np.ndarray:
        if key not in self._groups:
            self._groups[key] = key.groups(self.records)
        return self._groups[key]

    def covers(self, key: BlockingKey) -> np.ndarray:
        groups = self.groups(key)
        left, right = groups[self.reference[:, 0]], groups[self.reference[:, 1]]
        return (left >= 0) & (left == right)

    def _prefix_lengths(self, attribute: str) -> list[int]:
        keys = comparison_keys(self.records.values(attribute))
        lengths = np.array([len(key.replace(" ", "")) for key in keys if key])
        if len(lengths) == 0:
            return []
        median = float(np.median(lengths))
        sizes, size = [], 1
        while size < median:
            sizes.append(size)
            size *= 2
        return sizes


@dataclass(frozen=True)
class AutomaticKeys(Blocker):
    sorted_window: bool = False
    window: int | None = None

    @property
    def label(self) -> str:
        if not self.sorted_window:
            return "стандартне за автоматично підібраними ключами"
        size = "авто" if self.window is None else str(self.window)
        return f"сортоване сусідство (w={size}) за автоматично підібраними ключами"

    def resolve(self, records: RecordSet, scope: PairScope) -> Blocker:
        keys = KeySelector(records, scope).suggest().keys
        if not keys:
            raise ValueError("Не вдалося підібрати ключі блокування для цих даних. Оберіть блокування TF-IDF.")
        return SortedNeighbourhood(keys, self.window) if self.sorted_window else KeyBlocking(keys)

    def candidates(self, records: RecordSet, scope: PairScope) -> np.ndarray:
        return self.resolve(records, scope).candidates(records, scope)


def reference_matches(records: RecordSet, scope: PairScope) -> np.ndarray:
    search = TokenBlocking().search(records, scope)
    found = search.initial(1)
    pairs = []
    for group, part in enumerate(found):
        queries, targets, _ = search.plan[group]
        pairs.append(np.column_stack([queries[part.queries], targets[part.targets]]))
    if not pairs or not sum(len(part) for part in pairs):
        return EMPTY_PAIRS
    directed = np.concatenate(pairs)
    forward = encode(directed, records.size)
    backward = encode(directed[:, ::-1], records.size)
    mutual = directed[np.isin(forward, backward)]
    return canonical_pairs(mutual)


def _rank(item: KeyAssessment) -> tuple[float, float]:
    return -item.estimated_completeness, -item.reduction_ratio
