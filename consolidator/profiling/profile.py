from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from consolidator.io.source import Source
from consolidator.profiling.numeric import NumericSummary, numeric_values, summarize
from consolidator.profiling.outliers import OUTLIER_DETECTORS, OutlierDetector
from consolidator.profiling.types import ColumnType, infer_present_type
from consolidator.values.counts import frequencies
from consolidator.values.missing import missing_mask
from consolidator.values.patterns import value_shape

TOP_VALUES = 5
TOP_PATTERNS = 5
PATTERN_SAMPLE = 2000


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    type: ColumnType
    type_share: float
    rows: int
    missing: int
    disguised_missing: int
    distinct: int
    top_values: tuple[tuple[str, int], ...]
    top_patterns: tuple[tuple[str, float], ...]
    numeric: NumericSummary | None
    outliers: dict[str, int] = field(default_factory=dict)

    @property
    def present(self) -> int:
        return self.rows - self.missing

    @property
    def completeness(self) -> float:
        return self.present / self.rows if self.rows else 0.0

    @property
    def dominant_pattern_share(self) -> float:
        return self.top_patterns[0][1] if self.top_patterns else 0.0


@dataclass(frozen=True)
class TableProfile:
    source: str
    rows: int
    columns: tuple[ColumnProfile, ...]

    def column(self, name: str) -> ColumnProfile:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    @property
    def types(self) -> dict[str, ColumnType]:
        return {column.name: column.type for column in self.columns}

    @property
    def completeness(self) -> float:
        cells = self.rows * len(self.columns)
        return sum(column.present for column in self.columns) / cells if cells else 0.0


def profile_source(source: Source, detectors: Sequence[OutlierDetector] = OUTLIER_DETECTORS) -> TableProfile:
    return profile_frame(source.frame, source.name, detectors)


def profile_frame(
    frame: pd.DataFrame, name: str = "", detectors: Sequence[OutlierDetector] = OUTLIER_DETECTORS
) -> TableProfile:
    columns = tuple(profile_column(frame[column], str(column), detectors) for column in frame.columns)
    return TableProfile(name, len(frame), columns)


def profile_column(
    values: pd.Series, name: str, detectors: Sequence[OutlierDetector] = OUTLIER_DETECTORS
) -> ColumnProfile:
    text = values.fillna("").astype(str)
    missing = missing_mask(text)
    present = text[~missing].str.strip()
    inference = infer_present_type(present)
    counts = frequencies(present)
    numeric, outliers = None, {}
    if inference.type.is_quantitative:
        numbers = numeric_values(present, inference.type)
        numeric = summarize(numbers)
        outliers = {detector.key: int(detector.detect(numbers).sum()) for detector in detectors}
    return ColumnProfile(
        name=name,
        type=inference.type,
        type_share=inference.share,
        rows=len(text),
        missing=int(missing.sum()),
        disguised_missing=int((missing & (text.str.strip() != "").to_numpy()).sum()),
        distinct=len(counts),
        top_values=tuple((str(value), int(count)) for value, count in counts.head(TOP_VALUES).items()),
        top_patterns=_top_patterns(counts),
        numeric=numeric,
        outliers=outliers,
    )


def _top_patterns(counts: pd.Series) -> tuple[tuple[str, float], ...]:
    if counts.empty:
        return ()
    sample = counts.head(PATTERN_SAMPLE)
    shapes = sample.groupby(sample.index.map(value_shape), sort=False).sum().sort_values(ascending=False, kind="stable")
    total = float(sample.sum())
    return tuple((str(shape), float(count) / total) for shape, count in shapes.head(TOP_PATTERNS).items())
