from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from consolidator.pipeline import PipelineResult
from consolidator.reporting import (
    RELIABILITY_ROWS,
    RELIABLE,
    conflicts_table,
    consolidated_table,
    export_excel,
    export_json,
    export_sqlite,
    fusion_table,
    linkage_table,
    mapping_table,
    model_table,
    provenance_table,
    quality_table,
    records_table,
    reliability_table,
    result_profile_table,
    settings_table,
    source_accuracy_table,
    sources_table,
    standardization_table,
    timings_table,
)
from webapp.models import ReportFile

CSV_ENCODING = "utf-8-sig"
TABLE_DIRECTORY = "tables"
RECORDS_FILE = "records.csv"
FLAGS_FILE = "flags.csv"
REVIEW_FILE = "review.csv"
EVIDENCE_PREFIX = "evidence:"
CONSOLIDATED_META = 3
PROVENANCE_META = 1
RECORD_META = 3
FLAG_META = 1
CHUNK_ROWS = 50_000
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
REVIEW_COLUMNS = [
    "left_position",
    "right_position",
    "left_source",
    "left_key",
    "right_source",
    "right_key",
    "probability",
    "weight",
    "accepted",
]


class CellFlag(str, Enum):
    FILLED = "filled"
    CONFLICT = "conflict"
    IMPUTED = "imputed"


FLAG_LABELS = {
    CellFlag.FILLED: "у частини записів значення бракувало — його взято з іншого джерела",
    CellFlag.CONFLICT: "джерела давали різні значення — обрано узгоджене",
    CellFlag.IMPUTED: "пропуск відновлено статистичним методом",
}


@dataclass(frozen=True)
class Download:
    file_name: str
    mimetype: str
    label: str


DOWNLOADS = {
    "consolidated": Download("consolidated.csv", "text/csv", "Консолідовані дані (CSV)"),
    "report": Download("report.xlsx", XLSX, "Повний звіт (Excel)"),
    "provenance": Download("provenance.csv", "text/csv", "Походження значень (CSV)"),
    "database": Download("consolidated.sqlite", "application/vnd.sqlite3", "База даних (SQLite)"),
    "summary": Download("summary.json", "application/json", "Підсумок (JSON)"),
}
TABLES: dict[str, Callable[[PipelineResult], pd.DataFrame]] = {
    "quality": quality_table,
    "sources": sources_table,
    "mapping": mapping_table,
    "standardization": standardization_table,
    "linkage": linkage_table,
    "reliability": reliability_table,
    "model": model_table,
    "fusion": fusion_table,
    "conflicts": conflicts_table,
    "accuracy": source_accuracy_table,
    "result_profile": result_profile_table,
    "settings": settings_table,
    "timings": timings_table,
}


@dataclass(frozen=True)
class RecordView:
    source: str
    key: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class EntityView:
    entity: int
    sources: str
    attributes: tuple[str, ...]
    values: tuple[str, ...]
    flags: tuple[str, ...]
    provenance: tuple[str, ...]
    records: tuple[RecordView, ...]


@dataclass(frozen=True)
class ReliabilityView:
    reliable: bool
    false_share: float
    false_tolerance: float
    missed_share: float
    missed_tolerance: float
    contested: int
    possible: int
    suggested: int
    reasons: str = ""


@dataclass(frozen=True)
class Preview:
    attributes: tuple[str, ...]
    rows: pd.DataFrame
    flags: pd.DataFrame
    total: int


class ReportWriter:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def write(self, result: PipelineResult) -> list[ReportFile]:
        tables = self.directory / TABLE_DIRECTORY
        tables.mkdir(parents=True, exist_ok=True)
        self._csv(consolidated_table(result), DOWNLOADS["consolidated"].file_name)
        self._csv(provenance_table(result), DOWNLOADS["provenance"].file_name)
        export_excel(result, self.directory / DOWNLOADS["report"].file_name)
        export_sqlite(result, self.directory / DOWNLOADS["database"].file_name)
        export_json(result, self.directory / DOWNLOADS["summary"].file_name)
        for key, build in TABLES.items():
            self._csv(build(result), f"{TABLE_DIRECTORY}/{key}.csv")
        self._csv(records_table(result), RECORDS_FILE)
        self._csv(flag_table(result), FLAGS_FILE)
        self._csv(review_pairs(result), REVIEW_FILE)
        return [
            ReportFile(kind, download.file_name, (self.directory / download.file_name).stat().st_size)
            for kind, download in DOWNLOADS.items()
        ]

    def _csv(self, frame: pd.DataFrame, name: str) -> None:
        frame.to_csv(self.directory / name, index=False, encoding=CSV_ENCODING)


class ReportReader:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def table(self, key: str) -> pd.DataFrame:
        return _read(self.directory / TABLE_DIRECTORY / f"{key}.csv")

    def reliability(self) -> ReliabilityView | None:
        path = self.directory / TABLE_DIRECTORY / "reliability.csv"
        if not path.is_file():
            return None
        table = _read(path)
        if table.empty:
            return None
        values = dict(zip(table["Показник"], table["Значення"]))
        value = {key: values.get(label, "") for key, label in RELIABILITY_ROWS.items()}
        return ReliabilityView(
            reliable=value["verdict"] == RELIABLE,
            false_share=float(value["false_share"] or 0),
            false_tolerance=float(value["false_tolerance"] or 0),
            missed_share=float(value["missed_share"] or 0),
            missed_tolerance=float(value["missed_tolerance"] or 0),
            contested=int(value["contested"] or 0),
            possible=int(value["possible"] or 0),
            suggested=int(value["suggested"] or 0),
            reasons="" if value["reasons"] in ("", "—") else str(value["reasons"]),
        )

    def attributes(self) -> tuple[str, ...]:
        return tuple(self.table("mapping")["Атрибут"])

    def preview(self, rows: int) -> Preview:
        consolidated = _read(self.directory / DOWNLOADS["consolidated"].file_name, nrows=rows)
        flags = _read(self.directory / FLAGS_FILE, nrows=rows)
        return Preview(self.attributes(), consolidated, flags, self.entity_count())

    def entity_count(self) -> int:
        with (self.directory / FLAGS_FILE).open(encoding=CSV_ENCODING) as handle:
            return max(sum(1 for _ in handle) - 1, 0)

    def duplicates(self, limit: int) -> tuple[pd.DataFrame, int]:
        consolidated = _read(self.directory / DOWNLOADS["consolidated"].file_name)
        merged = consolidated[pd.to_numeric(consolidated.iloc[:, 2]) > 1]
        return merged.head(limit), len(merged)

    def review_pairs(self) -> pd.DataFrame:
        return _read(self.directory / REVIEW_FILE)

    def records(self, positions: Iterable[int]) -> dict[int, RecordView]:
        wanted = set(positions)
        found = {}
        for start, chunk in self._record_chunks():
            for offset, row in enumerate(chunk.itertuples(index=False, name=None)):
                if start + offset in wanted:
                    found[start + offset] = RecordView(row[1], row[2], tuple(row[RECORD_META:]))
        return found

    def entity(self, entity: int) -> EntityView | None:
        if entity < 0:
            return None
        consolidated = _row(self.directory / DOWNLOADS["consolidated"].file_name, entity)
        if consolidated is None:
            return None
        flags = _row(self.directory / FLAGS_FILE, entity)
        provenance = _row(self.directory / DOWNLOADS["provenance"].file_name, entity)
        records = []
        for _, chunk in self._record_chunks():
            members = chunk[chunk.iloc[:, 0] == str(entity)]
            records.extend(
                RecordView(row[1], row[2], tuple(row[RECORD_META:]))
                for row in members.itertuples(index=False, name=None)
            )
        return EntityView(
            entity=entity,
            sources=str(consolidated.iloc[1]),
            attributes=self.attributes(),
            values=tuple(consolidated.iloc[CONSOLIDATED_META:]),
            flags=tuple(flags.iloc[FLAG_META:]) if flags is not None else (),
            provenance=tuple(provenance.iloc[PROVENANCE_META:]) if provenance is not None else (),
            records=tuple(records),
        )

    def _record_chunks(self) -> Iterable[tuple[int, pd.DataFrame]]:
        start = 0
        for chunk in pd.read_csv(
            self.directory / RECORDS_FILE, dtype=str, keep_default_na=False, encoding=CSV_ENCODING, chunksize=CHUNK_ROWS
        ):
            yield start, chunk
            start += len(chunk)


def flag_table(result: PipelineResult) -> pd.DataFrame:
    fusion, attributes = result.fusion, result.attributes
    entities = pd.RangeIndex(fusion.n_entities)
    codes = pd.DataFrame("", index=entities, columns=attributes, dtype=object)
    filled = fusion.filled_cells().groupby(fusion.entity_ids).any().reindex(entities, fill_value=False)
    codes = codes.mask(filled.to_numpy(dtype=bool), CellFlag.FILLED.value)
    for attribute, group in fusion.conflicts.groupby("attribute"):
        codes.loc[group["entity"].to_numpy(), attribute] = CellFlag.CONFLICT.value
    if result.imputation is not None:
        imputed = result.imputation.imputed[attributes].to_numpy(dtype=bool)
        codes = codes.mask(imputed, CellFlag.IMPUTED.value)
    codes.insert(0, "entity", entities)
    return codes


def review_pairs(result: PipelineResult) -> pd.DataFrame:
    linkage, records = result.linkage, result.records
    if linkage is None:
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    queue = linkage.review_queue()
    left, right = linkage.candidates[queue, 0], linkage.candidates[queue, 1]
    names = pd.Series(records.source_names, dtype=object)
    frame = pd.DataFrame(
        {
            "left_position": left,
            "right_position": right,
            "left_source": names[records.source_codes[left]].to_numpy(),
            "left_key": records.keys[left],
            "right_source": names[records.source_codes[right]].to_numpy(),
            "right_key": records.keys[right],
            "probability": linkage.probabilities[queue],
            "weight": linkage.weights[queue],
            "accepted": linkage.accepted[queue].astype(int),
        },
        columns=REVIEW_COLUMNS,
    )
    if linkage.model is None:
        return frame
    for index, (attribute, weights) in enumerate(zip(linkage.comparison.attributes, linkage.model.weights)):
        levels = linkage.comparison.levels[queue, index]
        frame[EVIDENCE_PREFIX + attribute] = np.where(levels >= 0, weights[np.maximum(levels, 0)], np.nan)
    return frame


def _read(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=CSV_ENCODING, nrows=nrows)


def _row(path: Path, position: int) -> pd.Series | None:
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding=CSV_ENCODING,
        skiprows=range(1, position + 1),
        nrows=1,
    )
    return frame.iloc[0] if len(frame) else None
