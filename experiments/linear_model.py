from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from consolidator.linkage import ComparisonMatrix


@dataclass(frozen=True)
class LinearModel:
    weights: Mapping[str, float]
    threshold: float = 0.5

    @classmethod
    def equal(cls, attributes: Sequence[str], threshold: float = 0.5) -> LinearModel:
        return cls(dict.fromkeys(attributes, 1.0), threshold)

    def scores(self, comparison: ComparisonMatrix) -> np.ndarray:
        attributes = [attribute for attribute in comparison.attributes if self.weights.get(attribute, 0.0) > 0]
        if not attributes:
            return np.zeros(len(comparison.similarities), dtype=np.float64)
        subset = comparison.subset(attributes)
        weights = np.array([self.weights[attribute] for attribute in attributes], dtype=np.float64)
        similarities = subset.similarities.astype(np.float64)
        present = ~np.isnan(similarities)
        weighted = np.where(present, similarities, 0.0) @ weights
        mass = present.astype(np.float64) @ weights
        return np.divide(weighted, mass, out=np.zeros_like(weighted), where=mass > 0)

    def predict(self, comparison: ComparisonMatrix) -> np.ndarray:
        return self.scores(comparison) >= self.threshold
