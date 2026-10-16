from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from webapp.jobs import RunQueue
from webapp.repositories import Repositories
from webapp.services.mapping import MappingService
from webapp.services.projects import ProjectService
from webapp.services.reviews import ReviewService
from webapp.services.runs import RunService
from webapp.services.sources import SourceService
from webapp.storage import Storage


@dataclass(frozen=True)
class Services:
    repositories: Repositories
    storage: Storage
    projects: ProjectService
    sources: SourceService
    mapping: MappingService
    reviews: ReviewService
    runs: RunService

    @classmethod
    def build(cls, db: sqlite3.Connection, storage: Storage, queue: RunQueue, database: Path) -> Services:
        repositories = Repositories.on(db)
        sources = SourceService(repositories, storage)
        mapping = MappingService(repositories, sources)
        return cls(
            repositories=repositories,
            storage=storage,
            projects=ProjectService(repositories, storage),
            sources=sources,
            mapping=mapping,
            reviews=ReviewService(repositories, storage),
            runs=RunService(repositories, storage, queue, database, mapping),
        )
