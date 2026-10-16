from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from consolidator.profiling.numeric import summarize

RANDOM_STATE = 0


class OutlierDetector(ABC):
    key = ""

    @abstractmethod
    def detect(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TukeyFences(OutlierDetector):
    key = "iqr"

    def detect(self, values: np.ndarray) -> np.ndarray:
        summary = summarize(values)
        if summary is None:
            return np.zeros(values.shape, dtype=bool)
        with np.errstate(invalid="ignore"):
            return (values < summary.lower_fence) | (values > summary.upper_fence)


class IsolationForestDetector(OutlierDetector):
    key = "isolation_forest"

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state

    def detect(self, values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        flags = np.zeros(len(values), dtype=bool)
        if finite.sum() < 2 or np.ptp(values[finite]) == 0:
            return flags
        sample = values[finite].reshape(-1, 1)
        model = IsolationForest(random_state=self.random_state).fit(sample)
        flags[finite] = model.predict(sample) == -1
        return flags


OUTLIER_DETECTORS: tuple[OutlierDetector, ...] = (TukeyFences(), IsolationForestDetector())


def detect_outliers(values: pd.DataFrame, detector: OutlierDetector, groups: np.ndarray | None = None) -> pd.DataFrame:
    data = values if groups is None else group_deviations(values, groups)
    flags = {column: detector.detect(data[column].to_numpy(dtype=np.float64)) for column in data.columns}
    return pd.DataFrame(flags, index=values.index, columns=list(data.columns))


def group_deviations(values: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    keys = pd.Series(np.asarray(groups), index=values.index)
    return values - values.groupby(keys).transform("median")
