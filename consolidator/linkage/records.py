from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from consolidator.profiling.types import ColumnType

RecordId = tuple[str, str]


class PairScope(str, Enum):
    ALL = "all"
    CROSS = "cross"


@dataclass(frozen=True, eq=False)
class RecordSet:
    frame: pd.DataFrame
    types: dict[str, ColumnType]
    source_codes: np.ndarray
    source_names: tuple[str, ...]
    keys: np.ndarray

    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame], types: Mapping[str, ColumnType]) -> RecordSet:
        attributes = list(types)
        parts, codes, keys = [], [], []
        for code, frame in enumerate(frames.values()):
            parts.append(frame.reindex(columns=attributes).reset_index(drop=True))
            codes.append(np.full(len(frame), code, dtype=np.int32))
            keys.append(frame.index.astype(str).to_numpy(dtype=object))
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=attributes)
        frame = frame.fillna("").astype(object)
        return cls(
            frame=frame,
            types=dict(types),
            source_codes=np.concatenate(codes) if codes else np.array([], dtype=np.int32),
            source_names=tuple(frames),
            keys=np.concatenate(keys) if keys else np.array([], dtype=object),
        )

    @property
    def size(self) -> int:
        return len(self.frame)

    @property
    def attributes(self) -> list[str]:
        return list(self.frame.columns)

    @property
    def source_sizes(self) -> np.ndarray:
        return np.bincount(self.source_codes, minlength=len(self.source_names))

    def values(self, attribute: str) -> np.ndarray:
        return self.frame[attribute].to_numpy(dtype=object)

    def record_id(self, position: int) -> RecordId:
        return self.source_names[self.source_codes[position]], str(self.keys[position])

    def positions(self, ids: Iterable[RecordId]) -> np.ndarray:
        lookup = self.position_lookup()
        return np.array([lookup[(str(source), str(key))] for source, key in ids], dtype=np.int64)

    def position_lookup(self) -> dict[RecordId, int]:
        names = np.asarray(self.source_names, dtype=object)[self.source_codes]
        return {(str(source), str(key)): position for position, (source, key) in enumerate(zip(names, self.keys))}

    def total_pairs(self, scope: PairScope) -> int:
        return pair_count(self.source_sizes, scope)

    def pair_ids(self, pairs: np.ndarray) -> set[tuple[RecordId, RecordId]]:
        names = np.asarray(self.source_names, dtype=object)
        result = set()
        for left, right in pairs:
            a = (names[self.source_codes[left]], str(self.keys[left]))
            b = (names[self.source_codes[right]], str(self.keys[right]))
            result.add((a, b) if a <= b else (b, a))
        return result

    def pairs_from_ids(self, ids: Iterable[Sequence[RecordId]]) -> np.ndarray:
        lookup = self.position_lookup()
        rows = [(lookup[tuple(map(str, a))], lookup[tuple(map(str, b))]) for a, b in ids]
        if not rows:
            return np.empty((0, 2), dtype=np.int64)
        pairs = np.array(rows, dtype=np.int64)
        return np.sort(pairs, axis=1)


def pair_count(sizes: Sequence[int] | np.ndarray, scope: PairScope) -> int:
    sizes = np.asarray(sizes, dtype=np.int64)
    total = int(sizes.sum())
    if scope is PairScope.ALL:
        return total * (total - 1) // 2
    return (total * total - int((sizes * sizes).sum())) // 2
