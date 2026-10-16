from __future__ import annotations

import json
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
from sqlite3 import Row

from consolidator.evaluation import MatchMetrics
from consolidator.profiling import ColumnType


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"

    @property
    def done(self) -> bool:
        return self in (RunStatus.FINISHED, RunStatus.FAILED)


class AttributeRole(str, Enum):
    ATTRIBUTE = "attribute"
    IGNORED = "ignored"
    REFERENCE = "reference"


def parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass(frozen=True)
class Project:
    id: int
    name: str
    description: str
    options: dict[str, object]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: Row) -> Project:
        return cls(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            options=json.loads(row["options"]),
            created_at=parse_time(row["created_at"]),
            updated_at=parse_time(row["updated_at"]),
        )


@dataclass(frozen=True)
class ProjectOverview:
    project: Project
    source_count: int
    record_count: int
    run_count: int
    last_index: float | None


@dataclass(frozen=True)
class SourceInfo:
    id: int
    table_id: int
    file_id: int
    position: int
    part: str
    name: str
    format: str
    record_count: int
    column_count: int
    completeness: float
    priority: int
    included: bool

    @classmethod
    def from_row(cls, row: Row) -> SourceInfo:
        return cls(
            id=row["id"],
            table_id=row["table_id"],
            file_id=row["file_id"],
            position=row["position"],
            part=row["part"],
            name=row["name"],
            format=row["format"],
            record_count=row["record_count"],
            column_count=row["column_count"],
            completeness=row["completeness"],
            priority=row["priority"],
            included=bool(row["included"]),
        )


@dataclass(frozen=True)
class SourceTable:
    id: int
    file_id: int
    position: int
    name: str
    format: str
    columns: tuple[str, ...]
    details: dict[str, str]
    key_column: str | None
    split_column: str | None
    sources: tuple[SourceInfo, ...] = ()

    @classmethod
    def from_row(cls, row: Row, sources: tuple[SourceInfo, ...] = ()) -> SourceTable:
        return cls(
            id=row["id"],
            file_id=row["file_id"],
            position=row["position"],
            name=row["name"],
            format=row["format"],
            columns=tuple(json.loads(row["columns"])),
            details=json.loads(row["details"]),
            key_column=row["key_column"],
            split_column=row["split_column"],
            sources=sources,
        )


@dataclass(frozen=True)
class UploadedFile:
    id: int
    project_id: int
    original_name: str
    stored_name: str
    size_bytes: int
    sha256: str
    uploaded_at: datetime
    tables: tuple[SourceTable, ...] = ()

    @classmethod
    def from_row(cls, row: Row, tables: tuple[SourceTable, ...] = ()) -> UploadedFile:
        return cls(
            id=row["id"],
            project_id=row["project_id"],
            original_name=row["original_name"],
            stored_name=row["stored_name"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            uploaded_at=parse_time(row["uploaded_at"]),
            tables=tables,
        )

    @property
    def sources(self) -> tuple[SourceInfo, ...]:
        return tuple(source for table in self.tables for source in table.sources)


@dataclass(frozen=True)
class TableDraft:
    position: int
    name: str
    format: str
    columns: tuple[str, ...]
    details: dict[str, str]


@dataclass(frozen=True)
class SourceDraft:
    part: str
    name: str
    record_count: int
    column_count: int
    completeness: float
    profile: list[dict[str, object]]


@dataclass(frozen=True)
class AttributeState:
    name: str
    type: ColumnType
    confidence: float
    role: AttributeRole
    resolution: str | None
    columns: tuple[tuple[int, str], ...]

    @property
    def included(self) -> bool:
        return self.role is AttributeRole.ATTRIBUTE

    def column_in(self, source_id: int) -> str | None:
        for owner, column in self.columns:
            if owner == source_id:
                return column
        return None


@dataclass(frozen=True)
class PairDecision:
    left_source: str
    left_key: str
    right_source: str
    right_key: str
    is_match: bool | None


@dataclass(frozen=True)
class Run:
    id: int
    project_id: int
    status: RunStatus
    stage: str
    options: dict[str, object]
    error: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def from_row(cls, row: Row) -> Run:
        return cls(
            id=row["id"],
            project_id=row["project_id"],
            status=RunStatus(row["status"]),
            stage=row["stage"],
            options=json.loads(row["options"]),
            error=row["error"],
            created_at=parse_time(row["created_at"]),
            started_at=parse_time(row["started_at"]),
            finished_at=parse_time(row["finished_at"]),
        )

    @property
    def seconds(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class RunListItem:
    run: Run
    record_count: int | None
    entity_count: int | None
    index_before: float | None
    index_after: float | None


@dataclass(frozen=True)
class Report:
    run_id: int
    source_count: int
    record_count: int
    attribute_count: int
    entity_count: int
    merged_records: int
    candidate_pairs: int
    matched_pairs: int
    possible_pairs: int
    reviewed_pairs: int
    unified_cells: int
    unrecognized_cells: int
    conflicting_cells: int
    filled_cells: int
    imputed_cells: int
    index_before: float
    index_after: float
    seconds: float

    @classmethod
    def from_row(cls, row: Row) -> Report:
        return cls(**{field.name: row[field.name] for field in fields(cls)})


@dataclass(frozen=True)
class QualityScore:
    criterion: str
    weight: float
    before: float
    after: float


@dataclass(frozen=True)
class Evaluation:
    reference: str
    true_entities: int
    pairs: MatchMetrics

    @classmethod
    def from_row(cls, row: Row) -> Evaluation:
        correct = row["correct_pairs"]
        pairs = MatchMetrics(correct, row["found_pairs"] - correct, row["true_pairs"] - correct)
        return cls(row["reference"], row["true_entities"], pairs)


@dataclass(frozen=True)
class ReportFile:
    kind: str
    file_name: str
    size_bytes: int
