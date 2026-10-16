from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from consolidator.profiling.types import ColumnType
from consolidator.standardization.settings import CellStatus
from consolidator.values.missing import missing_mask

FORMATTED_TYPES = frozenset(
    {
        ColumnType.PHONE,
        ColumnType.EMAIL,
        ColumnType.URL,
        ColumnType.DATE,
        ColumnType.DATETIME,
        ColumnType.TIME,
        ColumnType.YEAR,
        ColumnType.NUMBER,
        ColumnType.MEASURE,
        ColumnType.BOOLEAN,
    }
)


@dataclass(frozen=True, eq=False)
class DataSnapshot:
    frames: Sequence[pd.DataFrame]
    statuses: Sequence[pd.DataFrame]
    types: Mapping[str, ColumnType]
    records: int
    entities: int


class QualityCriterion(ABC):
    key = ""

    @abstractmethod
    def measure(self, snapshot: DataSnapshot) -> float:
        raise NotImplementedError


class Completeness(QualityCriterion):
    key = "completeness"

    def measure(self, snapshot: DataSnapshot) -> float:
        present = cells = 0
        for frame in snapshot.frames:
            for column in frame.columns:
                missing = missing_mask(frame[column])
                present += int((~missing).sum())
                cells += len(missing)
        return present / cells if cells else 1.0


class RecordCompleteness(QualityCriterion):
    key = "completeness"

    def measure(self, snapshot: DataSnapshot) -> float:
        complete = records = 0
        for frame in snapshot.frames:
            incomplete = np.zeros(len(frame), dtype=bool)
            for column in frame.columns:
                incomplete |= missing_mask(frame[column])
            complete += int((~incomplete).sum())
            records += len(frame)
        return complete / records if records else 1.0


class Uniqueness(QualityCriterion):
    key = "uniqueness"

    def measure(self, snapshot: DataSnapshot) -> float:
        return snapshot.entities / snapshot.records if snapshot.records else 1.0


class Consistency(QualityCriterion):
    key = "consistency"

    def measure(self, snapshot: DataSnapshot) -> float:
        conforming = present = 0
        for status in snapshot.statuses:
            for column in status.columns:
                if snapshot.types.get(column) not in FORMATTED_TYPES:
                    continue
                codes = status[column].to_numpy()
                present += int((codes != CellStatus.MISSING).sum())
                conforming += int((codes == CellStatus.UNCHANGED).sum())
        return conforming / present if present else 1.0


CRITERIA: tuple[QualityCriterion, ...] = (Completeness(), Uniqueness(), Consistency())
CRITERION_KEYS = tuple(criterion.key for criterion in CRITERIA)
