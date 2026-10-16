from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
CONSISTENCY_LIMIT = 0.1


@dataclass(frozen=True, eq=False)
class AhpResult:
    criteria: tuple[str, ...]
    matrix: np.ndarray
    weights: np.ndarray
    lambda_max: float
    consistency_index: float
    consistency_ratio: float

    @property
    def consistent(self) -> bool:
        return self.consistency_ratio < CONSISTENCY_LIMIT

    def weight(self, criterion: str) -> float:
        return float(self.weights[self.criteria.index(criterion)])

    def as_dict(self) -> dict[str, float]:
        return {criterion: float(weight) for criterion, weight in zip(self.criteria, self.weights)}


def pairwise_matrix(criteria: Sequence[str], judgements: Mapping[tuple[str, str], float]) -> np.ndarray:
    index = {criterion: position for position, criterion in enumerate(criteria)}
    matrix = np.ones((len(criteria), len(criteria)), dtype=np.float64)
    for (left, right), value in judgements.items():
        if value <= 0:
            raise ValueError("Оцінка попарного порівняння має бути додатною")
        matrix[index[left], index[right]] = value
        matrix[index[right], index[left]] = 1.0 / value
    return matrix


def analyse(criteria: Sequence[str], matrix: np.ndarray) -> AhpResult:
    matrix = np.asarray(matrix, dtype=np.float64)
    size = len(criteria)
    if matrix.shape != (size, size):
        raise ValueError("Розмір матриці не відповідає кількості критеріїв")
    if not np.allclose(matrix * matrix.T, 1.0):
        raise ValueError("Матриця попарних порівнянь має бути оберненосиметричною: a[j][i] = 1 / a[i][j]")
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    principal = int(np.argmax(eigenvalues.real))
    weights = np.abs(eigenvectors[:, principal].real)
    weights = weights / weights.sum()
    lambda_max = float(eigenvalues[principal].real)
    consistency_index = (lambda_max - size) / (size - 1) if size > 1 else 0.0
    random_index = RANDOM_INDEX.get(size, RANDOM_INDEX[10])
    consistency_ratio = consistency_index / random_index if random_index else 0.0
    return AhpResult(tuple(criteria), matrix, weights, lambda_max, consistency_index, consistency_ratio)
