from __future__ import annotations

import numpy as np

from consolidator.linkage.records import PairScope, RecordSet

EMPTY_PAIRS = np.empty((0, 2), dtype=np.int64)


def canonical_pairs(pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return EMPTY_PAIRS
    pairs = np.sort(np.asarray(pairs, dtype=np.int64).reshape(-1, 2), axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(pairs, axis=0)


def in_scope(records: RecordSet, pairs: np.ndarray, scope: PairScope) -> np.ndarray:
    if scope is PairScope.ALL or len(pairs) == 0:
        return pairs
    codes = records.source_codes
    return pairs[codes[pairs[:, 0]] != codes[pairs[:, 1]]]


def union(*groups: np.ndarray) -> np.ndarray:
    present = [group for group in groups if len(group)]
    return canonical_pairs(np.concatenate(present)) if present else EMPTY_PAIRS


def encode(pairs: np.ndarray, size: int) -> np.ndarray:
    return pairs[:, 0].astype(np.int64) * size + pairs[:, 1].astype(np.int64)


def random_pairs(records: RecordSet, scope: PairScope, count: int, seed: int = 0) -> np.ndarray:
    total = records.total_pairs(scope)
    if total == 0:
        return EMPTY_PAIRS
    count = min(count, total)
    rng = np.random.default_rng(seed)
    collected = EMPTY_PAIRS
    attempts = 0
    while len(collected) < count and attempts < 50:
        attempts += 1
        batch = max(2 * (count - len(collected)), 1024)
        left = rng.integers(0, records.size, batch)
        right = rng.integers(0, records.size, batch)
        candidate = canonical_pairs(np.column_stack([left, right]))
        collected = union(collected, in_scope(records, candidate, scope))
    if len(collected) > count:
        collected = collected[rng.choice(len(collected), count, replace=False)]
    return canonical_pairs(collected)


def pairs_within_groups(groups: np.ndarray, sources: np.ndarray, scope: PairScope) -> np.ndarray:
    valid = np.flatnonzero(groups >= 0)
    if len(valid) < 2:
        return EMPTY_PAIRS
    order = valid[np.lexsort((sources[valid], groups[valid]))]
    group = groups[order]
    source = sources[order]
    n = len(order)
    position = np.arange(n)
    group_end = _run_ends(group)
    if scope is PairScope.ALL:
        first_partner = position + 1
    else:
        first_partner = _run_ends(group * (int(source.max()) + 1) + source)
    counts = np.maximum(group_end - first_partner, 0)
    total = int(counts.sum())
    if total == 0:
        return EMPTY_PAIRS
    left = np.repeat(position, counts)
    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    right = np.repeat(first_partner, counts) + offsets
    return canonical_pairs(np.column_stack([order[left], order[right]]))


def count_within_groups(groups: np.ndarray, sources: np.ndarray, scope: PairScope) -> int:
    valid = groups >= 0
    if valid.sum() < 2:
        return 0
    group_sizes = np.bincount(groups[valid]).astype(np.int64)
    total = int((group_sizes * (group_sizes - 1) // 2).sum())
    if scope is PairScope.ALL:
        return total
    combined = groups[valid].astype(np.int64) * (int(sources.max()) + 1) + sources[valid]
    _, run_sizes = np.unique(combined, return_counts=True)
    same_source = int((run_sizes.astype(np.int64) * (run_sizes - 1) // 2).sum())
    return total - same_source


def _run_ends(values: np.ndarray) -> np.ndarray:
    n = len(values)
    boundaries = np.flatnonzero(np.diff(values)) + 1
    ends = np.append(boundaries, n)
    starts = np.insert(boundaries, 0, 0)
    return np.repeat(ends, ends - starts)
