from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict

from webapp.database import timestamp
from webapp.models import Evaluation, QualityScore, Report, ReportFile, Run, RunListItem, RunStatus

INTERRUPTED = "Обробку перервано: застосунок було перезапущено до її завершення."
PROJECT_RUNS = """
SELECT run.*, report.record_count, report.entity_count, report.index_before, report.index_after
  FROM run LEFT JOIN report ON report.run_id = run.id
 WHERE run.project_id = ?
 ORDER BY run.id DESC
"""
INSERT_REPORT = """
INSERT INTO report (
    run_id, source_count, record_count, attribute_count, entity_count, merged_records, candidate_pairs,
    matched_pairs, possible_pairs, reviewed_pairs, unified_cells, unrecognized_cells, conflicting_cells,
    filled_cells, imputed_cells, index_before, index_after, seconds
) VALUES (
    :run_id, :source_count, :record_count, :attribute_count, :entity_count, :merged_records, :candidate_pairs,
    :matched_pairs, :possible_pairs, :reviewed_pairs, :unified_cells, :unrecognized_cells, :conflicting_cells,
    :filled_cells, :imputed_cells, :index_before, :index_after, :seconds
)
"""


class RunRepository:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def create(self, project_id: int, options: dict[str, object]) -> int:
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO run (project_id, status, options, created_at) VALUES (?, ?, ?, ?)",
                (project_id, RunStatus.QUEUED.value, json.dumps(options, ensure_ascii=False), timestamp()),
            )
        return int(cursor.lastrowid)

    def get(self, run_id: int) -> Run | None:
        row = self.db.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
        return Run.from_row(row) if row else None

    def list_for(self, project_id: int) -> list[RunListItem]:
        return [
            RunListItem(
                run=Run.from_row(row),
                record_count=row["record_count"],
                entity_count=row["entity_count"],
                index_before=row["index_before"],
                index_after=row["index_after"],
            )
            for row in self.db.execute(PROJECT_RUNS, (project_id,))
        ]

    def start(self, run_id: int) -> None:
        with self.db:
            self.db.execute(
                "UPDATE run SET status = ?, started_at = ? WHERE id = ?",
                (RunStatus.RUNNING.value, timestamp(), run_id),
            )

    def set_stage(self, run_id: int, stage: str) -> None:
        with self.db:
            self.db.execute("UPDATE run SET stage = ? WHERE id = ?", (stage, run_id))

    def finish(
        self,
        report: Report,
        scores: Sequence[QualityScore],
        evaluation: Evaluation | None,
        files: Sequence[ReportFile],
    ) -> None:
        run_id = report.run_id
        with self.db:
            self.db.execute(INSERT_REPORT, asdict(report))
            self.db.executemany(
                "INSERT INTO quality_score (run_id, criterion, weight, before, after) VALUES (?, ?, ?, ?, ?)",
                [(run_id, score.criterion, score.weight, score.before, score.after) for score in scores],
            )
            if evaluation is not None:
                pairs = evaluation.pairs
                self.db.execute(
                    "INSERT INTO evaluation (run_id, reference, true_entities, true_pairs, found_pairs, correct_pairs) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        evaluation.reference,
                        evaluation.true_entities,
                        pairs.expected,
                        pairs.found,
                        pairs.true_positives,
                    ),
                )
            self.db.executemany(
                "INSERT INTO report_file (run_id, kind, file_name, size_bytes) VALUES (?, ?, ?, ?)",
                [(run_id, file.kind, file.file_name, file.size_bytes) for file in files],
            )
            self.db.execute(
                "UPDATE run SET status = ?, stage = '', finished_at = ? WHERE id = ?",
                (RunStatus.FINISHED.value, timestamp(), run_id),
            )

    def fail(self, run_id: int, message: str) -> None:
        with self.db:
            self.db.execute(
                "UPDATE run SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (RunStatus.FAILED.value, message, timestamp(), run_id),
            )

    def interrupt_unfinished(self) -> int:
        with self.db:
            cursor = self.db.execute(
                "UPDATE run SET status = ?, error = ?, finished_at = ? WHERE status IN (?, ?)",
                (RunStatus.FAILED.value, INTERRUPTED, timestamp(), RunStatus.QUEUED.value, RunStatus.RUNNING.value),
            )
        return cursor.rowcount

    def report(self, run_id: int) -> Report | None:
        row = self.db.execute("SELECT * FROM report WHERE run_id = ?", (run_id,)).fetchone()
        return Report.from_row(row) if row else None

    def scores(self, run_id: int) -> list[QualityScore]:
        rows = self.db.execute("SELECT * FROM quality_score WHERE run_id = ? ORDER BY rowid", (run_id,))
        return [QualityScore(row["criterion"], row["weight"], row["before"], row["after"]) for row in rows]

    def evaluation(self, run_id: int) -> Evaluation | None:
        row = self.db.execute("SELECT * FROM evaluation WHERE run_id = ?", (run_id,)).fetchone()
        return Evaluation.from_row(row) if row else None

    def files(self, run_id: int) -> list[ReportFile]:
        rows = self.db.execute("SELECT * FROM report_file WHERE run_id = ? ORDER BY rowid", (run_id,))
        return [ReportFile(row["kind"], row["file_name"], row["size_bytes"]) for row in rows]

    def delete(self, run_id: int) -> None:
        with self.db:
            self.db.execute("DELETE FROM run WHERE id = ?", (run_id,))
