from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from consolidator.profiling.types import ColumnType
from consolidator.values.dates import infer_dayfirst, parse_date, parse_time, parse_year
from consolidator.values.numbers import parse_number

TUKEY_FACTOR = 1.5
SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class NumericSummary:
    count: int
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1

    @property
    def lower_fence(self) -> float:
        return self.q1 - TUKEY_FACTOR * self.iqr

    @property
    def upper_fence(self) -> float:
        return self.q3 + TUKEY_FACTOR * self.iqr


def numeric_values(values: pd.Series, column_type: ColumnType) -> np.ndarray:
    text = values.fillna("").astype(str)
    converter = _converter(column_type, text)
    unique = pd.unique(text)
    mapping = {value: converter(value) for value in unique}
    return text.map(mapping).to_numpy(dtype=float)


def summarize(values: np.ndarray) -> NumericSummary | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    q1, median, q3 = np.percentile(finite, [25, 50, 75])
    return NumericSummary(
        int(finite.size), float(finite.min()), float(q1), float(median), float(q3), float(finite.max())
    )


def quantitative_frame(frame: pd.DataFrame, types: Mapping[str, ColumnType]) -> pd.DataFrame:
    columns = [column for column, column_type in types.items() if column_type.is_quantitative and column in frame]
    return pd.DataFrame(
        {column: numeric_values(frame[column], types[column]) for column in columns}, index=frame.index, columns=columns
    )


def _converter(column_type: ColumnType, values: pd.Series) -> Callable[[str], float]:
    if column_type in (ColumnType.NUMBER, ColumnType.MEASURE):
        return _number
    if column_type is ColumnType.YEAR:
        return _year
    if column_type in (ColumnType.DATE, ColumnType.DATETIME):
        dayfirst = infer_dayfirst(values)
        return lambda text: _date(text, dayfirst)
    if column_type is ColumnType.TIME:
        return _time
    return lambda text: np.nan


def _number(text: str) -> float:
    parsed = parse_number(text)
    return parsed.value if parsed is not None else np.nan


def _year(text: str) -> float:
    year = parse_year(text)
    return float(year) if year is not None else np.nan


def _date(text: str, dayfirst: bool | None) -> float:
    parsed = parse_date(text, dayfirst)
    if parsed is None or parsed.ambiguous:
        return np.nan
    seconds = 0 if parsed.clock is None else parsed.clock.hour * 3600 + parsed.clock.minute * 60 + parsed.clock.second
    return parsed.value.toordinal() + seconds / SECONDS_PER_DAY


def _time(text: str) -> float:
    parsed = parse_time(text)
    return np.nan if parsed is None else parsed.value.hour * 60 + parsed.value.minute + parsed.value.second / 60
