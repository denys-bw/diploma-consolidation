from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from consolidator.io.formats import RawFile, ReadOptions, TableFormat
from consolidator.io.names import stem_of, unique_column_names
from consolidator.io.source import Source

SQLITE_HEADER = b"SQLite format 3\x00"
TABLES = "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY rowid"


class SqliteFormat(TableFormat):
    suffixes = (".sqlite", ".sqlite3", ".db", ".db3")

    def recognizes(self, file: RawFile) -> bool:
        return super().recognizes(file) or file.data.startswith(SQLITE_HEADER)

    def read(self, file: RawFile, options: ReadOptions) -> list[Source]:
        tables = parse_sqlite(file.data)
        return [
            Source(table if len(tables) > 1 else stem_of(file.name), frame, file.origin, "SQLite", {"table": table})
            for table, frame in tables
        ]


def parse_sqlite(data: bytes) -> list[tuple[str, pd.DataFrame]]:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "source.sqlite"
        path.write_bytes(data)
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        try:
            tables = [name for (name,) in connection.execute(TABLES)]
            return [(table, _table_frame(connection, table)) for table in tables]
        finally:
            connection.close()


def database_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _table_frame(connection: sqlite3.Connection, table: str) -> pd.DataFrame:
    quoted = '"' + table.replace('"', '""') + '"'
    cursor = connection.execute(f"SELECT * FROM {quoted}")
    columns = unique_column_names(description[0] for description in cursor.description)
    rows = [[database_text(value) for value in row] for row in cursor.fetchall()]
    return pd.DataFrame(rows, columns=columns, dtype=object)
