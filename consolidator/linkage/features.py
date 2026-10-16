from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from consolidator.linkage.comparison import COMPARATORS, COMPARISON_BATCH, Comparator, candidate_metrics
from consolidator.linkage.records import RecordSet

PRESENT = "є значення"
MAX_FEATURE_VALUES = 150_000_000


@dataclass(frozen=True, eq=False)
class MetricValues:
    names: tuple[tuple[str, str], ...]
    attributes: tuple[str, ...]
    similarities: np.ndarray

    def column(self, attribute: str, metric: str) -> np.ndarray:
        return self.similarities[:, self.names.index((attribute, metric))]

    def metrics(self, attribute: str) -> list[str]:
        return [metric for owner, metric in self.names if owner == attribute]

    def rows(self, selection: np.ndarray) -> MetricValues:
        return MetricValues(self.names, self.attributes, self.similarities[selection])

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f"{attribute}: {metric}" for attribute, metric in self.names) + tuple(
            f"{attribute}: {PRESENT}" for attribute in self.attributes
        )

    def features(self) -> np.ndarray:
        present = np.zeros((len(self.similarities), len(self.attributes)), dtype=np.float32)
        for index, attribute in enumerate(self.attributes):
            columns = [position for position, (owner, _) in enumerate(self.names) if owner == attribute]
            if columns:
                present[:, index] = ~np.isnan(self.similarities[:, columns[0]])
        return np.column_stack([np.nan_to_num(self.similarities, nan=0.0), present])


class MetricView:
    def __init__(self, comparators: Mapping[tuple[str, str], Comparator], attributes: Sequence[str]) -> None:
        self.comparators = dict(comparators)
        self.attributes = tuple(attributes)

    @classmethod
    def for_records(cls, records: RecordSet, attributes: Sequence[str]) -> MetricView:
        comparators = {
            (attribute, metric): COMPARATORS[metric]().fit(records.values(attribute))
            for attribute in attributes
            for metric in candidate_metrics(records.types[attribute])
        }
        return cls(comparators, attributes)

    @property
    def empty(self) -> bool:
        return not self.comparators

    @property
    def width(self) -> int:
        return len(self.comparators) + len(self.attributes)

    def affordable(self, pairs: int) -> bool:
        return pairs * self.width <= MAX_FEATURE_VALUES

    def compute(self, pairs: np.ndarray) -> MetricValues:
        values = np.zeros((len(pairs), len(self.comparators)), dtype=np.float32)
        for start in range(0, len(pairs), COMPARISON_BATCH):
            batch = slice(start, start + COMPARISON_BATCH)
            left, right = pairs[batch, 0], pairs[batch, 1]
            for index, comparator in enumerate(self.comparators.values()):
                values[batch, index] = comparator.similarity(left, right)
        return MetricValues(tuple(self.comparators), self.attributes, values)
