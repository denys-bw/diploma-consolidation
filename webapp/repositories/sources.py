from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence

from webapp.database import timestamp
from webapp.models import SourceDraft, SourceInfo, SourceTable, TableDraft, UploadedFile
from webapp.storage import StoredFile

SOURCES = """
SELECT source.*, source_table.file_id, source_table.position, source_table.format
  FROM source
  JOIN source_table ON source_table.id = source.table_id
  JOIN source_file ON source_file.id = source_table.file_id
 WHERE source_file.project_id = ?
"""
ORDER = " ORDER BY source_table.file_id, source_table.position, source.id"
TABLES = """
SELECT source_table.*
  FROM source_table JOIN source_file ON source_file.id = source_table.file_id
 WHERE source_file.project_id = ?
"""


class SourceRepository:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def add_file(self, project_id: int, original_name: str, stored: StoredFile) -> int:
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO source_file (project_id, original_name, stored_name, size_bytes, sha256, uploaded_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, original_name, stored.name, stored.size, stored.sha256, timestamp()),
            )
        return int(cursor.lastrowid)

    def add_table(self, file_id: int, draft: TableDraft) -> int:
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO source_table (file_id, position, name, format, columns, details) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    draft.position,
                    draft.name,
                    draft.format,
                    json.dumps(list(draft.columns), ensure_ascii=False),
                    json.dumps(draft.details, ensure_ascii=False),
                ),
            )
        return int(cursor.lastrowid)

    def replace_sources(self, project_id: int, table_id: int, drafts: Sequence[SourceDraft]) -> None:
        with self.db:
            row = self.db.execute(
                "SELECT min(priority) AS priority FROM source WHERE table_id = ?", (table_id,)
            ).fetchone()
            self.db.execute("DELETE FROM source WHERE table_id = ?", (table_id,))
            priority = row["priority"] if row["priority"] is not None else self._next_priority(project_id)
            self.db.executemany(
                "INSERT INTO source (table_id, part, name, record_count, column_count, completeness, priority, profile) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        table_id,
                        draft.part,
                        draft.name,
                        draft.record_count,
                        draft.column_count,
                        draft.completeness,
                        priority + offset,
                        json.dumps(draft.profile, ensure_ascii=False),
                    )
                    for offset, draft in enumerate(drafts)
                ],
            )

    def set_table_options(self, table_id: int, key_column: str | None, split_column: str | None) -> None:
        with self.db:
            self.db.execute(
                "UPDATE source_table SET key_column = ?, split_column = ? WHERE id = ?",
                (key_column, split_column, table_id),
            )

    def files(self, project_id: int) -> list[UploadedFile]:
        tables: dict[int, list[SourceTable]] = {}
        for table in self.tables(project_id):
            tables.setdefault(table.file_id, []).append(table)
        rows = self.db.execute("SELECT * FROM source_file WHERE project_id = ? ORDER BY id", (project_id,))
        return [UploadedFile.from_row(row, tuple(tables.get(row["id"], ()))) for row in rows]

    def file(self, project_id: int, file_id: int) -> UploadedFile | None:
        return next((uploaded for uploaded in self.files(project_id) if uploaded.id == file_id), None)

    def tables(self, project_id: int) -> list[SourceTable]:
        sources: dict[int, list[SourceInfo]] = {}
        for source in self.sources(project_id):
            sources.setdefault(source.table_id, []).append(source)
        rows = self.db.execute(TABLES + " ORDER BY source_table.file_id, source_table.position", (project_id,))
        return [SourceTable.from_row(row, tuple(sources.get(row["id"], ()))) for row in rows]

    def table(self, project_id: int, table_id: int) -> SourceTable | None:
        return next((table for table in self.tables(project_id) if table.id == table_id), None)

    def sources(self, project_id: int, included_only: bool = False) -> list[SourceInfo]:
        query = SOURCES + (" AND source.included = 1" if included_only else "") + ORDER
        return [SourceInfo.from_row(row) for row in self.db.execute(query, (project_id,))]

    def source(self, project_id: int, source_id: int) -> SourceInfo | None:
        row = self.db.execute(SOURCES + " AND source.id = ?", (project_id, source_id)).fetchone()
        return SourceInfo.from_row(row) if row else None

    def profile(self, source_id: int) -> list[dict[str, object]]:
        row = self.db.execute("SELECT profile FROM source WHERE id = ?", (source_id,)).fetchone()
        return json.loads(row["profile"]) if row else []

    def names(self, project_id: int) -> set[str]:
        return {source.name for source in self.sources(project_id)}

    def set_included(self, source_id: int, included: bool) -> None:
        with self.db:
            self.db.execute("UPDATE source SET included = ? WHERE id = ?", (int(included), source_id))

    def set_priorities(self, priorities: Mapping[int, int]) -> None:
        with self.db:
            self.db.executemany(
                "UPDATE source SET priority = ? WHERE id = ?",
                [(priority, source_id) for source_id, priority in priorities.items()],
            )

    def delete_file(self, file_id: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM source_file WHERE id = ?", (file_id,))

    def stored_names(self, project_id: int) -> list[str]:
        rows = self.db.execute("SELECT stored_name FROM source_file WHERE project_id = ?", (project_id,))
        return [row["stored_name"] for row in rows]

    def _next_priority(self, project_id: int) -> int:
        sources = self.sources(project_id)
        return max((source.priority for source in sources), default=0) + 1
