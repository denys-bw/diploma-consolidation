from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from functools import partial
from pathlib import Path

import pandas as pd

from consolidator.pipeline import PipelineResult
from consolidator.profiling import ColumnType
from consolidator.reporting.summary import summarize
from consolidator.reporting.tables import (
    conflicts_table,
    consolidated_table,
    fusion_table,
    linkage_table,
    mapping_table,
    model_table,
    provenance_table,
    quality_table,
    records_table,
    reliability_table,
    result_profile_table,
    review_table,
    settings_table,
    source_accuracy_table,
    sources_table,
    standardization_table,
    timings_table,
)

CSV_ENCODING = "utf-8-sig"
ENTITY = "Сутність"
NUMERIC_TYPES = {ColumnType.NUMBER, ColumnType.YEAR}


def report_tables(result: PipelineResult) -> dict[str, pd.DataFrame]:
    return {
        "Консолідовані дані": consolidated_table(result),
        "Походження значень": provenance_table(result),
        "Якість": quality_table(result),
        "Джерела": sources_table(result),
        "Зіставлення схем": mapping_table(result),
        "Уніфікація": standardization_table(result),
        "Зв'язування": linkage_table(result),
        "Надійність": reliability_table(result),
        "Модель Фелегі–Зунтера": model_table(result),
        "Можливі збіги": review_table(result),
        "Злиття": fusion_table(result),
        "Конфлікти значень": conflicts_table(result),
        "Точність джерел": source_accuracy_table(result),
        "Профіль результату": result_profile_table(result),
        "Налаштування": settings_table(result),
        "Час виконання": timings_table(result),
    }


def export_excel(result: PipelineResult, path: str | Path) -> Path:
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in report_tables(result).items():
            table.to_excel(writer, sheet_name=name, index=False)
    return path


def export_csv(result: PipelineResult, directory: str | Path) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "consolidated.csv": consolidated_table(result),
        "provenance.csv": provenance_table(result),
        "quality.csv": quality_table(result),
    }
    paths = []
    for name, table in tables.items():
        path = directory / name
        table.to_csv(path, index=False, encoding=CSV_ENCODING)
        paths.append(path)
    return paths


def export_json(result: PipelineResult, path: str | Path) -> Path:
    path = Path(path)
    payload = {
        "summary": summarize(result).as_dict(),
        "mapping": mapping_table(result).to_dict(orient="records"),
        "timings": result.timings,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_sqlite(result: PipelineResult, path: str | Path) -> Path:
    path = Path(path)
    path.unlink(missing_ok=True)
    numeric = {attribute for attribute, kind in result.records.types.items() if kind in NUMERIC_TYPES}
    with closing(sqlite3.connect(path)) as connection, connection:
        create = partial(_create_table, connection, numeric=numeric)
        create("entities", consolidated_table(result), primary=ENTITY)
        create("provenance", provenance_table(result), primary=ENTITY, parent="entities")
        create("source_records", records_table(result), parent="entities")
        create("quality", quality_table(result))
        connection.execute(f"CREATE INDEX source_records_entity ON source_records ({_quoted(ENTITY)})")
    return path


def _create_table(
    connection: sqlite3.Connection,
    name: str,
    frame: pd.DataFrame,
    numeric: set[str],
    primary: str | None = None,
    parent: str | None = None,
) -> None:
    columns = [
        f"{_quoted(column)} {_sql_type(frame[column], column in numeric)}{' PRIMARY KEY' if column == primary else ''}"
        for column in frame.columns
    ]
    if parent is not None:
        columns.append(f"FOREIGN KEY ({_quoted(ENTITY)}) REFERENCES {parent} ({_quoted(ENTITY)})")
    connection.execute(f"CREATE TABLE {name} ({', '.join(columns)})")
    placeholders = ", ".join("?" for _ in frame.columns)
    connection.executemany(f"INSERT INTO {name} VALUES ({placeholders})", _rows(frame))


def _sql_type(column: pd.Series, numeric: bool) -> str:
    if pd.api.types.is_integer_dtype(column):
        return "INTEGER"
    if pd.api.types.is_float_dtype(column):
        return "REAL"
    return "NUMERIC" if numeric else "TEXT"


def _rows(frame: pd.DataFrame) -> Iterator[tuple[object, ...]]:
    cells = frame.astype(object).where(frame.notna(), None)
    for row in cells.itertuples(index=False, name=None):
        yield tuple(None if value == "" else value for value in row)


def _quoted(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'
