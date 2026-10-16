from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from consolidator.linkage.comparison import comparable_attributes
from consolidator.linkage.keys import BlockingKey
from consolidator.linkage.pairs import EMPTY_PAIRS, canonical_pairs, encode, in_scope, pairs_within_groups, union
from consolidator.linkage.records import PairScope, RecordSet
from consolidator.profiling.numeric import numeric_values
from consolidator.profiling.types import ColumnType
from consolidator.statistics import MAJORITY
from consolidator.values.text import comparison_keys
from consolidator.values.vectors import CHUNK_CELLS, tfidf

MAX_FULL_PAIRS = 5_000_000
CANDIDATE_LIMIT = 50_000_000
TOKEN_TYPES = frozenset(
    {
        ColumnType.TEXT,
        ColumnType.CATEGORY,
        ColumnType.CODE,
        ColumnType.IDENTIFIER,
        ColumnType.PHONE,
        ColumnType.EMAIL,
        ColumnType.URL,
    }
)

Judge = Callable[[np.ndarray], np.ndarray]


def scale_neighbours(records: RecordSet) -> int:
    return max(1, math.ceil(math.log2(records.size))) if records.size > 1 else 1


class Blocker(ABC):
    def resolve(self, records: RecordSet, scope: PairScope) -> Blocker:
        return self

    @abstractmethod
    def candidates(self, records: RecordSet, scope: PairScope) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def label(self) -> str:
        raise NotImplementedError

    @property
    def adaptive(self) -> bool:
        return False


@dataclass(frozen=True)
class FullComparison(Blocker):
    max_pairs: int = MAX_FULL_PAIRS

    @property
    def label(self) -> str:
        return "без блокування"

    def candidates(self, records: RecordSet, scope: PairScope) -> np.ndarray:
        total = records.total_pairs(scope)
        if total > self.max_pairs:
            raise ValueError(f"Без блокування довелося б порівняти {total} пар записів, допустимо до {self.max_pairs}")
        groups = np.zeros(records.size, dtype=np.int64)
        return pairs_within_groups(groups, records.source_codes, scope)


@dataclass(frozen=True)
class KeyBlocking(Blocker):
    keys: tuple[BlockingKey, ...]

    @property
    def label(self) -> str:
        return "стандартне за ключами: " + " | ".join(key.label for key in self.keys)

    def candidates(self, records: RecordSet, scope: PairScope) -> np.ndarray:
        return union(*(pairs_within_groups(key.groups(records), records.source_codes, scope) for key in self.keys))


@dataclass(frozen=True)
class SortedNeighbourhood(Blocker):
    keys: tuple[BlockingKey, ...]
    window: int | None = None

    @property
    def label(self) -> str:
        size = "авто" if self.window is None else str(self.window)
        return f"сортоване сусідство (w={size}): " + " | ".join(key.label for key in self.keys)

    def candidates(self, records: RecordSet, scope: PairScope) -> np.ndarray:
        window = self.window or scale_neighbours(records) + 1
        return union(*(self._pass(key, records, scope, window) for key in self.keys))

    def _pass(self, key: BlockingKey, records: RecordSet, scope: PairScope, window: int) -> np.ndarray:
        values = key.values(records)
        present = np.flatnonzero(values != "")
        if len(present) < 2:
            return EMPTY_PAIRS
        order = present[np.argsort(values[present], kind="stable")]
        pairs = [np.column_stack([order[:-offset], order[offset:]]) for offset in range(1, min(window, len(order)))]
        return in_scope(records, canonical_pairs(np.concatenate(pairs)), scope)


@dataclass(frozen=True)
class NumericNeighbourhood(Blocker):
    attributes: tuple[str, ...]
    window: int | None = None

    @property
    def label(self) -> str:
        return "сортоване сусідство за числовими атрибутами: " + ", ".join(self.attributes)

    def candidates(self, records: RecordSet, scope: PairScope) -> np.ndarray:
        window = self.window or scale_neighbours(records) + 1
        return union(*(self._pass(attribute, records, scope, window) for attribute in self.attributes))

    def _pass(self, attribute: str, records: RecordSet, scope: PairScope, window: int) -> np.ndarray:
        values = numeric_values(pd.Series(records.values(attribute)), records.types[attribute])
        present = np.flatnonzero(np.isfinite(values))
        if len(present) < 2:
            return EMPTY_PAIRS
        order = present[np.argsort(values[present], kind="stable")]
        pairs = [np.column_stack([order[:-offset], order[offset:]]) for offset in range(1, min(window, len(order)))]
        return in_scope(records, canonical_pairs(np.concatenate(pairs)), scope)


@dataclass(frozen=True)
class TokenBlocking(Blocker):
    attributes: tuple[str, ...] | None = None
    neighbours: int | None = None

    @property
    def label(self) -> str:
        count = "адаптивно" if self.neighbours is None else str(self.neighbours)
        return f"TF-IDF, найближчих сусідів: {count}"

    @property
    def adaptive(self) -> bool:
        return self.neighbours is None

    def resolve(self, records: RecordSet, scope: PairScope) -> Blocker:
        if self.attributes is not None:
            return self
        comparable = comparable_attributes(records, scope)
        if any(records.types.get(attribute) in TOKEN_TYPES for attribute in comparable):
            return self
        numeric = tuple(attribute for attribute in comparable if records.types[attribute].is_quantitative)
        return NumericNeighbourhood(numeric) if numeric else self

    def candidates(self, records: RecordSet, scope: PairScope) -> np.ndarray:
        search = self.search(records, scope)
        return search.pairs(search.initial(self.neighbours or scale_neighbours(records)))

    def adaptive_candidates(self, records: RecordSet, scope: PairScope, judge: Judge) -> tuple[np.ndarray, np.ndarray]:
        search = self.search(records, scope)
        found = search.initial(scale_neighbours(records))
        while True:
            pairs = search.pairs(found)
            probabilities = judge(pairs)
            likely = probabilities >= MAJORITY
            expanded = search.expand(found, encode(pairs[likely], records.size))
            if expanded is None or len(search.pairs(expanded)) > CANDIDATE_LIMIT:
                return pairs, probabilities
            found = expanded

    def search(self, records: RecordSet, scope: PairScope) -> NeighbourSearch:
        return NeighbourSearch(records, self.vectors(records, scope), _plan(records, scope))

    def vectors(self, records: RecordSet, scope: PairScope = PairScope.ALL) -> sparse.csr_matrix:
        attributes = self.attributes or token_attributes(records, scope)
        blocks = [tfidf(comparison_keys(records.values(attribute))) for attribute in attributes]
        blocks = [block for block in blocks if block.shape[1]]
        if not blocks:
            return sparse.csr_matrix((records.size, 0), dtype=np.float32)
        return normalize(sparse.hstack(blocks, format="csr", dtype=np.float32))


@dataclass(frozen=True, eq=False)
class Neighbours:
    queries: np.ndarray
    targets: np.ndarray
    ranks: np.ndarray
    depth: np.ndarray


class NeighbourSearch:
    def __init__(self, records: RecordSet, matrix: sparse.csr_matrix, plan: list[tuple[np.ndarray, np.ndarray, bool]]):
        self.records = records
        self.matrix = matrix
        self.plan = plan

    def initial(self, neighbours: int) -> list[Neighbours]:
        return [
            self._search(group, np.arange(len(queries)), np.full(len(queries), neighbours, dtype=np.int64))
            for group, (queries, _, _) in enumerate(self.plan)
        ]

    def expand(self, found: list[Neighbours], likely: np.ndarray) -> list[Neighbours] | None:
        likely = np.sort(likely)
        result, changed = [], False
        for group, part in enumerate(found):
            queries, targets, within = self.plan[group]
            limit = len(targets) - (1 if within else 0)
            deep = part.ranks > part.depth[part.queries] // 2
            positions = np.sort(np.column_stack([queries[part.queries], targets[part.targets]]), axis=1)
            pair_codes = encode(positions, self.records.size)
            hit = _contains(likely, pair_codes) & deep
            grow = np.unique(part.queries[hit])
            grow = grow[part.depth[grow] < limit]
            if len(grow) == 0:
                result.append(part)
                continue
            changed = True
            depth = part.depth.copy()
            depth[grow] = np.minimum(depth[grow] * 2, limit)
            fresh = self._search(group, grow, depth)
            keep = ~np.isin(part.queries, grow)
            result.append(
                Neighbours(
                    np.concatenate([part.queries[keep], fresh.queries]),
                    np.concatenate([part.targets[keep], fresh.targets]),
                    np.concatenate([part.ranks[keep], fresh.ranks]),
                    depth,
                )
            )
        return result if changed else None

    def pairs(self, found: list[Neighbours]) -> np.ndarray:
        parts = [
            np.column_stack([self.plan[group][0][part.queries], self.plan[group][1][part.targets]])
            for group, part in enumerate(found)
            if len(part.queries)
        ]
        return canonical_pairs(np.concatenate(parts)) if parts else EMPTY_PAIRS

    def _search(self, group: int, positions: np.ndarray, depth: np.ndarray) -> Neighbours:
        queries, targets, within = self.plan[group]
        if self.matrix.shape[1] == 0 or len(targets) == 0 or len(positions) == 0:
            empty = np.zeros(0, dtype=np.int64)
            return Neighbours(empty, empty, empty, depth)
        target_matrix = self.matrix[targets].T.tocsr()
        step = max(1, CHUNK_CELLS // len(targets))
        found_queries, found_targets, found_ranks = [], [], []
        for start in range(0, len(positions), step):
            chunk = positions[start : start + step]
            rows = queries[chunk]
            similarity = (self.matrix[rows] @ target_matrix).toarray()
            if within:
                similarity[np.arange(len(rows)), np.searchsorted(targets, rows)] = -1.0
            k = int(min(depth[chunk].max(), len(targets)))
            order = _top(similarity, k)
            ranked = np.take_along_axis(similarity, order, axis=1)
            ranks = np.broadcast_to(np.arange(1, k + 1), order.shape)
            keep = (ranked > 0) & (ranks <= depth[chunk][:, None])
            row_index, column = np.nonzero(keep)
            found_queries.append(chunk[row_index])
            found_targets.append(order[row_index, column])
            found_ranks.append(ranks[row_index, column])
        return Neighbours(
            np.concatenate(found_queries), np.concatenate(found_targets), np.concatenate(found_ranks), depth
        )


def token_attributes(records: RecordSet, scope: PairScope = PairScope.ALL) -> list[str]:
    comparable = comparable_attributes(records, scope)
    textual = [attribute for attribute in comparable if records.types.get(attribute) in TOKEN_TYPES]
    return textual or comparable


def _plan(records: RecordSet, scope: PairScope) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    everyone = np.arange(records.size)
    codes = records.source_codes
    sources = len(records.source_names)
    members = [everyone[codes == code] for code in range(sources)]
    across = [(group, everyone[codes != code], False) for code, group in enumerate(members)]
    if scope is PairScope.CROSS:
        return across
    if sources == 1:
        return [(everyone, everyone, True)]
    return [(group, group, True) for group in members] + across


def _top(similarity: np.ndarray, k: int) -> np.ndarray:
    columns = similarity.shape[1]
    if k < columns:
        kth = -np.partition(-similarity, k - 1, axis=1)[:, k - 1 : k]
        above = similarity > kth
        tied = similarity == kth
        chosen = above | (tied & (np.cumsum(tied, axis=1) <= k - above.sum(axis=1, keepdims=True)))
        candidates = np.nonzero(chosen)[1].reshape(-1, k)
    else:
        candidates = np.broadcast_to(np.arange(columns), similarity.shape).copy()
    values = np.take_along_axis(similarity, candidates, axis=1)
    order = np.lexsort((candidates, -values), axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def _contains(sorted_codes: np.ndarray, codes: np.ndarray) -> np.ndarray:
    if len(sorted_codes) == 0:
        return np.zeros(len(codes), dtype=bool)
    positions = np.minimum(np.searchsorted(sorted_codes, codes), len(sorted_codes) - 1)
    return sorted_codes[positions] == codes
