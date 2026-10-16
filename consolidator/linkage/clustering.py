from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from consolidator.linkage.pairs import EMPTY_PAIRS, canonical_pairs, pairs_within_groups
from consolidator.linkage.records import PairScope, RecordSet

AVERAGE_TOLERANCE = 1e-12
NOT_REVIEWED = -1


@dataclass(frozen=True, eq=False)
class ScoredPairs:
    pairs: np.ndarray
    weights: np.ndarray
    probabilities: np.ndarray
    reviewed: np.ndarray
    scope: PairScope = PairScope.ALL

    @property
    def effective(self) -> np.ndarray:
        return with_review(self.probabilities, self.reviewed)


@dataclass(frozen=True, eq=False)
class Clustering:
    labels: np.ndarray
    accepted: np.ndarray


class Clusterer(ABC):
    key = ""

    @abstractmethod
    def cluster(self, records: RecordSet, scored: ScoredPairs, threshold: float) -> Clustering:
        raise NotImplementedError


class ConnectedComponents(Clusterer):
    key = "connected_components"

    def cluster(self, records: RecordSet, scored: ScoredPairs, threshold: float) -> Clustering:
        accepted = scored.effective >= threshold
        return Clustering(components(records.size, scored.pairs[accepted]), accepted)


class AverageLinkage(Clusterer):
    key = "average_linkage"

    def cluster(self, records: RecordSet, scored: ScoredPairs, threshold: float) -> Clustering:
        probabilities, pairs = scored.effective, scored.pairs
        labels = self.labels(records.source_codes, pairs, probabilities, threshold, scored.scope)
        same = labels[pairs[:, 0]] == labels[pairs[:, 1]] if len(pairs) else np.zeros(0, dtype=bool)
        return Clustering(labels, (probabilities >= threshold) & same)

    def labels(
        self,
        sources: np.ndarray,
        pairs: np.ndarray,
        probabilities: np.ndarray,
        threshold: float,
        scope: PairScope = PairScope.ALL,
    ) -> np.ndarray:
        size = len(sources)
        if len(pairs):
            component = components(size, pairs[probabilities >= threshold])
            relevant = component[pairs[:, 0]] == component[pairs[:, 1]]
            pairs, probabilities = pairs[relevant], probabilities[relevant]
        parent = np.arange(size, dtype=np.int64)
        members = [Counter({int(source): 1}) for source in sources]
        links: list[dict[int, float]] = [{} for _ in range(size)]
        for (left, right), probability in zip(pairs.tolist(), probabilities.tolist()):
            links[left][right] = links[left].get(right, 0.0) + probability
            links[right][left] = links[right].get(left, 0.0) + probability
        heap = [
            (-probability, left, right)
            for (left, right), probability in zip(pairs.tolist(), probabilities.tolist())
            if probability >= threshold
        ]
        heapq.heapify(heap)
        while heap:
            negative, a, b = heapq.heappop(heap)
            if parent[a] != a or parent[b] != b or b not in links[a]:
                continue
            average = _average(links[a][b], members[a], members[b], scope)
            if abs(average + negative) > AVERAGE_TOLERANCE or average < threshold:
                continue
            if len(links[a]) < len(links[b]):
                a, b = b, a
            parent[b] = a
            members[a] += members[b]
            for neighbour, total in links[b].items():
                if neighbour == a:
                    continue
                links[a][neighbour] = links[a].get(neighbour, 0.0) + total
                del links[neighbour][b]
                links[neighbour][a] = links[a][neighbour]
            links[a].pop(b, None)
            links[b] = {}
            for neighbour, total in links[a].items():
                merged = _average(total, members[a], members[neighbour], scope)
                if merged >= threshold:
                    heapq.heappush(heap, (-merged, min(a, neighbour), max(a, neighbour)))
        return _compact(np.array([_root(parent, node) for node in range(size)], dtype=np.int64))


class OneToOneAssignment(Clusterer):
    key = "one_to_one"

    def cluster(self, records: RecordSet, scored: ScoredPairs, threshold: float) -> Clustering:
        effective = scored.effective
        eligible = np.flatnonzero(effective >= threshold)
        order = np.lexsort((-scored.weights[eligible], -effective[eligible]))
        labels, used = self.assign(records, scored.pairs[eligible], order)
        accepted = np.zeros(len(scored.pairs), dtype=bool)
        accepted[eligible[used]] = True
        return Clustering(labels, accepted)

    def assign(self, records: RecordSet, pairs: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parent = np.arange(records.size, dtype=np.int64)
        sources = {node: {int(records.source_codes[node])} for node in range(records.size)}
        used = np.zeros(len(pairs), dtype=bool)
        for index in order:
            a, b = _root(parent, int(pairs[index, 0])), _root(parent, int(pairs[index, 1]))
            if a == b or sources[a] & sources[b]:
                continue
            parent[b] = a
            sources[a] |= sources.pop(b)
            used[index] = True
        return _compact(np.array([_root(parent, node) for node in range(records.size)], dtype=np.int64)), used


CLUSTERERS: dict[str, Clusterer] = {
    clusterer.key: clusterer for clusterer in (AverageLinkage(), ConnectedComponents(), OneToOneAssignment())
}


def components(size: int, pairs: np.ndarray) -> np.ndarray:
    if size == 0:
        return np.zeros(0, dtype=np.int64)
    if len(pairs) == 0:
        return np.arange(size, dtype=np.int64)
    graph = sparse.coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(size, size))
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64)


def cluster_pairs(labels: np.ndarray, source_codes: np.ndarray, scope: PairScope) -> np.ndarray:
    if len(labels) == 0:
        return EMPTY_PAIRS
    sizes = np.bincount(labels)
    groups = np.where(sizes[labels] > 1, labels, -1)
    return canonical_pairs(pairs_within_groups(groups, source_codes, scope))


def with_review(probabilities: np.ndarray, reviewed: np.ndarray) -> np.ndarray:
    return np.where(reviewed == NOT_REVIEWED, probabilities, reviewed.astype(np.float64))


def _average(total: float, left: Counter[int], right: Counter[int], scope: PairScope) -> float:
    possible = sum(left.values()) * sum(right.values())
    if scope is PairScope.CROSS:
        possible -= sum(count * right[source] for source, count in left.items())
    return total / possible if possible else 0.0


def _root(parent: np.ndarray, node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = int(parent[node])
    return node


def _compact(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels, return_inverse=True)[1].astype(np.int64)
