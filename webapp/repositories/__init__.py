from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from webapp.repositories.attributes import AttributeRepository
from webapp.repositories.projects import ProjectRepository
from webapp.repositories.reviews import ReviewRepository
from webapp.repositories.runs import RunRepository
from webapp.repositories.sources import SourceRepository


@dataclass(frozen=True)
class Repositories:
    projects: ProjectRepository
    sources: SourceRepository
    attributes: AttributeRepository
    reviews: ReviewRepository
    runs: RunRepository

    @classmethod
    def on(cls, db: sqlite3.Connection) -> Repositories:
        return cls(
            projects=ProjectRepository(db),
            sources=SourceRepository(db),
            attributes=AttributeRepository(db),
            reviews=ReviewRepository(db),
            runs=RunRepository(db),
        )
