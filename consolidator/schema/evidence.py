from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from consolidator.profiling.numeric import numeric_values
from consolidator.profiling.types import ColumnType
from consolidator.schema.mapping import joined_values
from consolidator.statistics import SignedRankTest, signed_rank_greater
from consolidator.values.missing import missing_mask
from consolidator.values.numbers import format_number
from consolidator.values.phones import NON_DIGITS
from consolidator.values.text import comparison_keys
from consolidator.values.vectors import mutual_nearest, qgram_vectors, row_cosine, tfidf

EVIDENCE_RECORDS = 5000
SEED = 0

Columns = tuple[str, ...]


@dataclass(frozen=True)
class PairEvidence:
    test: SignedRankTest
    effect: float

    @property
    def p_value(self) -> float:
        return self.test.p_value


NO_EVIDENCE = PairEvidence(SignedRankTest(0, 0.0, 1.0), 0.0)


class DuplicateEvidence:
    def __init__(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        left_types: Mapping[str, ColumnType],
        right_types: Mapping[str, ColumnType],
    ) -> None:
        generator = np.random.default_rng(SEED)
        if len(left) > EVIDENCE_RECORDS:
            left = left.iloc[np.sort(generator.choice(len(left), EVIDENCE_RECORDS, replace=False))]
        self.left = left.reset_index(drop=True)
        self.right = right.reset_index(drop=True)
        self.left_types = dict(left_types)
        self.right_types = dict(right_types)
        self._keys = {
            "left": {column: _keys(self.left[column], self.left_types.get(column)) for column in self.left.columns},
            "right": {column: _keys(self.right[column], self.right_types.get(column)) for column in self.right.columns},
        }
        documents = _documents(self._keys["left"], len(self.left)) + _documents(self._keys["right"], len(self.right))
        matrix = tfidf(documents)
        self.pairs = mutual_nearest(matrix[: len(self.left)], matrix[len(self.left) :])
        self.shuffled = _derangement(len(self.pairs), generator)
        self._vectors: dict[tuple[str, Columns], tuple[object, np.ndarray]] = {}

    @property
    def size(self) -> int:
        return len(self.pairs)

    def correspondence(self, left: Columns, right: Columns) -> PairEvidence:
        if self.size < 2:
            return NO_EVIDENCE
        true = self._similarity(left, right, self.pairs[:, 1])
        false = self._similarity(left, right, self.pairs[self.shuffled, 1])
        return _evidence(true - false)

    def improvement(self, base: tuple[Columns, Columns], extended: tuple[Columns, Columns]) -> PairEvidence:
        if self.size < 2:
            return NO_EVIDENCE
        before = self._similarity(*base, self.pairs[:, 1])
        after = self._similarity(*extended, self.pairs[:, 1])
        return _evidence(after - before)

    def _similarity(self, left: Columns, right: Columns, partners: np.ndarray) -> np.ndarray:
        left_vectors, left_present = self._side("left", left)
        right_vectors, right_present = self._side("right", right)
        rows = self.pairs[:, 0]
        similarity = row_cosine(left_vectors[rows], right_vectors[partners])
        return np.where(left_present[rows] & right_present[partners], similarity, np.nan)

    def _side(self, side: str, columns: Columns) -> tuple[object, np.ndarray]:
        key = (side, columns)
        if key not in self._vectors:
            if len(columns) == 1:
                keys = self._keys[side][columns[0]]
            else:
                frame = self.left if side == "left" else self.right
                types = self.left_types if side == "left" else self.right_types
                keys = _keys(joined_values(frame, list(columns)), types.get(columns[0]))
            self._vectors[key] = (qgram_vectors(keys), keys != "")
        return self._vectors[key]


def _evidence(differences: np.ndarray) -> PairEvidence:
    finite = differences[np.isfinite(differences)]
    if len(finite) == 0:
        return NO_EVIDENCE
    return PairEvidence(signed_rank_greater(finite), float(finite.mean()))


def _keys(values: pd.Series, column_type: ColumnType | None) -> np.ndarray:
    text = values.fillna("").astype(str)
    text = text.where(~missing_mask(text), "")
    if column_type is ColumnType.PHONE:
        mapping = {value: NON_DIGITS.sub("", value) for value in pd.unique(text)}
        return text.map(mapping).to_numpy(dtype=object)
    keys = comparison_keys(text)
    if column_type is None or not column_type.is_quantitative:
        return keys
    numbers = numeric_values(text, column_type)
    finite = np.isfinite(numbers)
    canonical = {number: format_number(number) for number in pd.unique(numbers[finite])}
    keys[finite] = [canonical[number] for number in numbers[finite]]
    return keys


def _documents(keys: Mapping[str, np.ndarray], size: int) -> list[str]:
    if not keys:
        return [""] * size
    return [" ".join(part for part in parts if part) for parts in zip(*keys.values())]


def _derangement(size: int, generator: np.random.Generator) -> np.ndarray:
    if size < 2:
        return np.arange(size)
    order = generator.permutation(size)
    shuffled = np.empty(size, dtype=np.int64)
    shuffled[order] = np.roll(order, -1)
    return shuffled
