from __future__ import annotations

import json
import sqlite3

from webapp.database import timestamp
from webapp.models import Project, ProjectOverview

OVERVIEW = """
SELECT project.*,
       (SELECT count(*)
          FROM source
          JOIN source_table ON source_table.id = source.table_id
          JOIN source_file ON source_file.id = source_table.file_id
         WHERE source_file.project_id = project.id AND source.included = 1) AS source_count,
       (SELECT coalesce(sum(source.record_count), 0)
          FROM source
          JOIN source_table ON source_table.id = source.table_id
          JOIN source_file ON source_file.id = source_table.file_id
         WHERE source_file.project_id = project.id AND source.included = 1) AS record_count,
       (SELECT count(*) FROM run WHERE run.project_id = project.id) AS run_count,
       (SELECT report.index_after
          FROM run JOIN report ON report.run_id = run.id
         WHERE run.project_id = project.id AND run.status = 'finished'
         ORDER BY run.id DESC
         LIMIT 1) AS last_index
  FROM project
 ORDER BY project.updated_at DESC, project.id DESC
"""


class ProjectRepository:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def create(self, name: str, description: str) -> int:
        now = timestamp()
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO project (name, description, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (name, description, now, now),
            )
        return int(cursor.lastrowid)

    def get(self, project_id: int) -> Project | None:
        row = self.db.execute("SELECT * FROM project WHERE id = ?", (project_id,)).fetchone()
        return Project.from_row(row) if row else None

    def overviews(self) -> list[ProjectOverview]:
        return [
            ProjectOverview(
                project=Project.from_row(row),
                source_count=row["source_count"],
                record_count=row["record_count"],
                run_count=row["run_count"],
                last_index=row["last_index"],
            )
            for row in self.db.execute(OVERVIEW)
        ]

    def save_options(self, project_id: int, options: dict[str, object]) -> None:
        with self.db:
            self.db.execute(
                "UPDATE project SET options = ?, updated_at = ? WHERE id = ?",
                (json.dumps(options, ensure_ascii=False), timestamp(), project_id),
            )

    def touch(self, project_id: int) -> None:
        with self.db:
            self.db.execute("UPDATE project SET updated_at = ? WHERE id = ?", (timestamp(), project_id))

    def delete(self, project_id: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM project WHERE id = ?", (project_id,))
