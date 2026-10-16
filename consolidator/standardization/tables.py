from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from consolidator.profiling.types import ColumnType
from consolidator.standardization.columns import standardizer_for
from consolidator.standardization.settings import CellStatus, StandardizationSettings

STATUS_COLUMNS = {
    CellStatus.MISSING: "missing",
    CellStatus.UNCHANGED: "unchanged",
    CellStatus.CHANGED: "changed",
    CellStatus.INVALID: "invalid",
    CellStatus.AMBIGUOUS: "ambiguous",
}


@dataclass(frozen=True, eq=False)
class StandardizedTable:
    frame: pd.DataFrame
    status: pd.DataFrame
    parameters: dict[str, dict[str, str]]

    def summary(self) -> pd.DataFrame:
        rows = []
        for column in self.status.columns:
            counts = self.status[column].value_counts()
            rows.append(
                {"column": column, **{label: int(counts.get(int(code), 0)) for code, label in STATUS_COLUMNS.items()}}
            )
        return pd.DataFrame(rows, columns=["column", *STATUS_COLUMNS.values()])


def standardize_frame(
    frame: pd.DataFrame,
    types: Mapping[str, ColumnType],
    settings: StandardizationSettings | None = None,
) -> StandardizedTable:
    return standardize_frames([frame], types, settings)[0]


def standardize_frames(
    frames: Sequence[pd.DataFrame],
    types: Mapping[str, ColumnType],
    settings: StandardizationSettings | None = None,
    names: Sequence[str] | None = None,
) -> list[StandardizedTable]:
    names = list(names) if names is not None else [str(position) for position in range(len(frames))]
    columns: dict[str, None] = {}
    for frame in frames:
        columns.update(dict.fromkeys(str(column) for column in frame.columns))
    outputs = [({}, {}) for _ in frames]
    parameters: dict[str, dict[str, str]] = {}
    for column in columns:
        parts = {name: frame[column] for name, frame in zip(names, frames) if column in frame.columns}
        standardizer = standardizer_for(types.get(column, ColumnType.TEXT), settings).fit(parts)
        parameters[column] = standardizer.parameters
        for name, frame, (values, statuses) in zip(names, frames, outputs):
            if column in frame.columns:
                result = standardizer.transform(frame[column], name)
                values[column], statuses[column] = result.values, result.status
    return [
        StandardizedTable(
            pd.DataFrame(values, index=frame.index, columns=list(values)),
            pd.DataFrame(statuses, index=frame.index, columns=list(statuses)),
            {column: parameters[column] for column in values},
        )
        for frame, (values, statuses) in zip(frames, outputs)
    ]
