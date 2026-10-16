from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from webapp.database import timestamp
from webapp.models import PairDecision

KEY = ("left_source", "left_key", "right_source", "right_key")


class ReviewRepository:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def decisions(self, project_id: int) -> list[PairDecision]:
        rows = self.db.execute("SELECT * FROM pair_review WHERE project_id = ? ORDER BY reviewed_at", (project_id,))
        return [PairDecision(*(row[column] for column in KEY), bool(row["is_match"])) for row in rows]

    def save(self, project_id: int, decisions: Sequence[PairDecision]) -> None:
        now = timestamp()
        decided = [decision for decision in decisions if decision.is_match is not None]
        with self.db:
            self.db.executemany(
                "DELETE FROM pair_review WHERE project_id = ? AND left_source = ? AND left_key = ? "
                "AND right_source = ? AND right_key = ?",
                [(project_id, *_key(decision)) for decision in decisions if decision.is_match is None],
            )
            self.db.executemany(
                "INSERT INTO pair_review (project_id, left_source, left_key, right_source, right_key, is_match, "
                "reviewed_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (project_id, left_source, left_key, right_source, right_key) "
                "DO UPDATE SET is_match = excluded.is_match, reviewed_at = excluded.reviewed_at",
                [(project_id, *_key(decision), int(decision.is_match), now) for decision in decided],
            )


def _key(decision: PairDecision) -> tuple[str, str, str, str]:
    return decision.left_source, decision.left_key, decision.right_source, decision.right_key
