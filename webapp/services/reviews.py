from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from consolidator.linkage import ClericalReview, RecordPair, ordered_pair
from webapp.models import PairDecision, Run
from webapp.repositories import Repositories
from webapp.services.reports import EVIDENCE_PREFIX, RecordView, ReportReader
from webapp.storage import Storage


@dataclass(frozen=True)
class ComparedValue:
    attribute: str
    left: str
    right: str
    evidence: float | None = None


@dataclass(frozen=True)
class PairView:
    left: RecordView
    right: RecordView
    values: tuple[ComparedValue, ...]
    probability: float
    accepted: bool
    decision: bool | None


@dataclass(frozen=True)
class ReviewPage:
    pairs: tuple[PairView, ...]
    number: int
    pages: int
    start: int
    total: int
    decided: int


class ReviewService:
    def __init__(self, repositories: Repositories, storage: Storage) -> None:
        self.repositories = repositories
        self.storage = storage

    def page(self, run: Run, number: int, size: int) -> ReviewPage:
        reader = ReportReader(self.storage.run_directory(run.id))
        queue = reader.review_pairs()
        decisions = self._decisions(run.project_id)
        keys = [
            ordered_pair((row.left_source, row.left_key), (row.right_source, row.right_key))
            for row in queue.itertuples(index=False)
        ]
        pages = max(math.ceil(len(queue) / size), 1)
        number = min(max(number, 1), pages)
        start = (number - 1) * size
        shown = queue.iloc[start : start + size]
        positions = [int(value) for value in (*shown["left_position"], *shown["right_position"])]
        records = reader.records(positions)
        attributes = reader.attributes()
        pairs = []
        for row, key in zip(shown.to_dict("records"), keys[start : start + size]):
            left, right = records[int(row["left_position"])], records[int(row["right_position"])]
            pairs.append(
                PairView(
                    left=left,
                    right=right,
                    values=_compared(attributes, left, right, _evidence(row)),
                    probability=float(row["probability"]),
                    accepted=row["accepted"] == "1",
                    decision=decisions.get(key),
                )
            )
        decided = sum(key in decisions for key in keys)
        return ReviewPage(tuple(pairs), number, pages, start, len(queue), decided)

    def save(self, project_id: int, decisions: Sequence[PairDecision]) -> None:
        self.repositories.reviews.save(project_id, [_canonical(decision) for decision in decisions])

    def clerical_review(self, project_id: int) -> ClericalReview:
        return ClericalReview(self._decisions(project_id))

    def _decisions(self, project_id: int) -> dict[RecordPair, bool]:
        return {
            ordered_pair((item.left_source, item.left_key), (item.right_source, item.right_key)): bool(item.is_match)
            for item in self.repositories.reviews.decisions(project_id)
        }


def _compared(
    attributes: Sequence[str], left: RecordView, right: RecordView, evidence: Mapping[str, float]
) -> tuple[ComparedValue, ...]:
    return tuple(
        ComparedValue(attribute, left_value, right_value, evidence.get(attribute))
        for attribute, left_value, right_value in zip(attributes, left.values, right.values)
        if left_value or right_value
    )


def _evidence(row: Mapping[str, str]) -> dict[str, float]:
    return {
        name.removeprefix(EVIDENCE_PREFIX): float(value)
        for name, value in row.items()
        if name.startswith(EVIDENCE_PREFIX) and value not in ("", "nan")
    }


def _canonical(decision: PairDecision) -> PairDecision:
    (left_source, left_key), (right_source, right_key) = ordered_pair(
        (decision.left_source, decision.left_key), (decision.right_source, decision.right_key)
    )
    return PairDecision(left_source, left_key, right_source, right_key, decision.is_match)
