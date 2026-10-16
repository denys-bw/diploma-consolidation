from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence

from consolidator.profiling import ColumnType
from webapp.models import AttributeRole, AttributeState

COLUMNS = """
SELECT attribute_column.*
  FROM attribute_column JOIN attribute ON attribute.id = attribute_column.attribute_id
 WHERE attribute.project_id = ?
 ORDER BY attribute_column.source_id
"""


class AttributeRepository:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def load(self, project_id: int) -> list[AttributeState]:
        columns: dict[int, list[tuple[int, str]]] = {}
        for row in self.db.execute(COLUMNS, (project_id,)):
            columns.setdefault(row["attribute_id"], []).append((row["source_id"], row["column_name"]))
        rows = self.db.execute("SELECT * FROM attribute WHERE project_id = ? ORDER BY position", (project_id,))
        return [
            AttributeState(
                name=row["name"],
                type=ColumnType(row["type"]),
                confidence=row["confidence"],
                role=AttributeRole(row["role"]),
                resolution=row["resolution"],
                columns=tuple(columns.get(row["id"], ())),
            )
            for row in rows
        ]

    def replace(self, project_id: int, attributes: Sequence[AttributeState]) -> None:
        with self.db:
            self.db.execute("DELETE FROM attribute WHERE project_id = ?", (project_id,))
            for position, attribute in enumerate(attributes):
                cursor = self.db.execute(
                    "INSERT INTO attribute (project_id, position, name, type, confidence, role, resolution) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        project_id,
                        position,
                        attribute.name,
                        attribute.type.value,
                        min(max(attribute.confidence, 0.0), 1.0),
                        attribute.role.value,
                        attribute.resolution,
                    ),
                )
                self.db.executemany(
                    "INSERT INTO attribute_column (attribute_id, source_id, column_name) VALUES (?, ?, ?)",
                    [(cursor.lastrowid, source_id, column) for source_id, column in attribute.columns],
                )

    def set_resolutions(self, project_id: int, resolutions: Mapping[str, str | None]) -> None:
        with self.db:
            self.db.executemany(
                "UPDATE attribute SET resolution = ? WHERE project_id = ? AND name = ?",
                [(resolution, project_id, name) for name, resolution in resolutions.items()],
            )
