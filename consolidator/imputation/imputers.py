from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from consolidator.profiling.numeric import quantitative_frame
from consolidator.profiling.types import ColumnType
from consolidator.standardization.columns import standardizer_for
from consolidator.standardization.settings import StandardizationSettings
from consolidator.values.missing import missing_mask

IMPUTED_MARK = "imputed:"
FOLDS = 5
VALIDATION_ROWS = 2000
SEED = 0


class Imputer(ABC):
    key = ""
    uses_other_attributes = False

    @property
    def mark(self) -> str:
        return IMPUTED_MARK + self.key

    @property
    def min_observed(self) -> int:
        return 1

    @property
    def label(self) -> str:
        return self.key

    def resolve(self, numbers: pd.DataFrame) -> Imputer:
        return self

    @abstractmethod
    def estimate(self, numbers: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class MeanImputer(Imputer):
    key = "mean"

    def estimate(self, numbers: pd.DataFrame) -> pd.DataFrame:
        return numbers.fillna(numbers.mean())


class KnnImputer(Imputer):
    key = "knn"
    uses_other_attributes = True

    def __init__(self, neighbours: int) -> None:
        self.neighbours = neighbours

    @property
    def min_observed(self) -> int:
        return self.neighbours

    @property
    def label(self) -> str:
        return f"{self.key}, k = {self.neighbours}"

    def estimate(self, numbers: pd.DataFrame) -> pd.DataFrame:
        centre = numbers.mean()
        scale = numbers.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        scaled = (numbers - centre) / scale
        filled = KNNImputer(n_neighbors=self.neighbours, keep_empty_features=True).fit_transform(
            scaled.to_numpy(dtype=np.float64)
        )
        return pd.DataFrame(filled, index=numbers.index, columns=numbers.columns) * scale + centre


class AdaptiveImputer(Imputer):
    key = "auto"

    def resolve(self, numbers: pd.DataFrame) -> Imputer:
        candidates = self.candidates(numbers)
        if len(candidates) == 1:
            return candidates[0]
        errors = cross_validated_errors(numbers, candidates)
        return min(candidates, key=lambda candidate: (errors[candidate.label], candidates.index(candidate)))

    def estimate(self, numbers: pd.DataFrame) -> pd.DataFrame:
        return self.resolve(numbers).estimate(numbers)

    @staticmethod
    def candidates(numbers: pd.DataFrame) -> list[Imputer]:
        candidates: list[Imputer] = [MeanImputer()]
        complete_rows = int(numbers.notna().sum(axis=1).gt(0).sum())
        if numbers.shape[1] < 2 or complete_rows < 2:
            return candidates
        limit = max(1, math.isqrt(min(complete_rows, VALIDATION_ROWS)))
        neighbours = 1
        while neighbours <= limit:
            candidates.append(KnnImputer(neighbours))
            neighbours *= 2
        return candidates


def cross_validated_errors(numbers: pd.DataFrame, candidates: list[Imputer]) -> dict[str, float]:
    generator = np.random.default_rng(SEED)
    rows = numbers.index.to_numpy()
    if len(rows) > VALIDATION_ROWS:
        rows = np.sort(generator.choice(rows, VALIDATION_ROWS, replace=False))
    sample = numbers.loc[rows]
    values = sample.to_numpy(dtype=np.float64)
    observed = np.isfinite(values)
    others = observed.sum(axis=1, keepdims=True) - observed >= 1
    testable = np.argwhere(observed & others)
    scale = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    totals = {candidate.label: 0.0 for candidate in candidates}
    if len(testable) == 0:
        return totals
    folds = generator.integers(0, FOLDS, len(testable))
    for fold in range(FOLDS):
        hidden = testable[folds == fold]
        if len(hidden) == 0:
            continue
        masked = values.copy()
        masked[hidden[:, 0], hidden[:, 1]] = np.nan
        frame = pd.DataFrame(masked, columns=sample.columns)
        truth = values[hidden[:, 0], hidden[:, 1]]
        for candidate in candidates:
            estimates = candidate.estimate(frame).to_numpy(dtype=np.float64)[hidden[:, 0], hidden[:, 1]]
            error = np.abs(estimates - truth) / scale[hidden[:, 1]]
            totals[candidate.label] += float(np.where(np.isfinite(error), error, np.inf).sum())
    return {label: total / len(testable) for label, total in totals.items()}


@dataclass(frozen=True, eq=False)
class ImputationResult:
    frame: pd.DataFrame
    imputed: pd.DataFrame
    imputer: Imputer | None = None

    @property
    def count(self) -> int:
        return int(self.imputed.to_numpy().sum())

    def counts(self) -> pd.Series:
        return self.imputed.sum().astype(int)

    @property
    def mark(self) -> str:
        return self.imputer.mark if self.imputer is not None else IMPUTED_MARK


def impute(
    frame: pd.DataFrame,
    types: Mapping[str, ColumnType],
    imputer: Imputer,
    settings: StandardizationSettings | None = None,
    applicable: pd.DataFrame | None = None,
) -> ImputationResult:
    output = frame.copy()
    imputed = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    numbers = quantitative_frame(frame, types)
    observed = numbers.notna()
    present = [column for column in numbers.columns if observed[column].any()]
    if not present:
        return ImputationResult(output, imputed)
    chosen = imputer.resolve(numbers[present])
    usable = [column for column in present if observed[column].sum() >= chosen.min_observed]
    if not usable:
        return ImputationResult(output, imputed, chosen)
    estimates = chosen.estimate(numbers[usable])
    for column in usable:
        targets = missing_mask(frame[column])
        if applicable is not None:
            targets &= applicable[column].to_numpy(dtype=bool)
        if chosen.uses_other_attributes:
            targets &= _observed_elsewhere(observed, usable, column)
        if not targets.any():
            continue
        renderer = standardizer_for(types[column], settings).fit(frame[column][observed[column].to_numpy()])
        positions = np.flatnonzero(targets)
        values = [renderer.render(value) for value in estimates[column].to_numpy()[positions]]
        keep = np.fromiter((value is not None for value in values), dtype=bool, count=len(values))
        location = frame.columns.get_loc(column)
        output.iloc[positions[keep], location] = np.asarray(values, dtype=object)[keep]
        imputed.iloc[positions[keep], location] = True
    return ImputationResult(output, imputed, chosen)


def _observed_elsewhere(observed: pd.DataFrame, usable: list[str], column: str) -> np.ndarray:
    others = [other for other in usable if other != column]
    if not others:
        return np.zeros(len(observed), dtype=bool)
    return observed[others].any(axis=1).to_numpy()
