from __future__ import annotations

from webapp.repositories import Repositories
from webapp.storage import Storage

ACTIVE_RUN = "Дочекайтеся завершення обробки: проєкт з активним запуском видалити не можна."


class ProjectError(ValueError):
    pass


class ProjectService:
    def __init__(self, repositories: Repositories, storage: Storage) -> None:
        self.repositories = repositories
        self.storage = storage

    def delete(self, project_id: int) -> None:
        runs = self.repositories.runs.list_for(project_id)
        if any(not item.run.status.done for item in runs):
            raise ProjectError(ACTIVE_RUN)
        stored = self.repositories.sources.stored_names(project_id)
        self.repositories.projects.delete(project_id)
        for name in stored:
            self.storage.delete_upload(name)
        for item in runs:
            self.storage.delete_run(item.run.id)
