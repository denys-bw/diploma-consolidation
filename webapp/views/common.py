from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from flask import abort, current_app, flash, g

from webapp.database import database_path, get_db
from webapp.models import Project, Run, RunStatus, SourceInfo
from webapp.options import RunOptions
from webapp.services import Services

ERROR = "error"
SUCCESS = "success"
NOTICE = "notice"


@dataclass(frozen=True)
class ProjectProgress:
    sources: int
    attributes: int
    runs: int
    report_run_id: int | None


def services() -> Services:
    if "services" not in g:
        g.services = Services.build(
            get_db(),
            current_app.extensions["storage"],
            current_app.extensions["run_queue"],
            database_path(current_app),
        )
    return g.services


def project_progress(project_id: int) -> ProjectProgress:
    repositories = services().repositories
    runs = repositories.runs.list_for(project_id)
    finished = next((item.run.id for item in runs if item.run.status is RunStatus.FINISHED), None)
    return ProjectProgress(
        sources=len(repositories.sources.sources(project_id, included_only=True)),
        attributes=sum(state.included for state in repositories.attributes.load(project_id)),
        runs=len(runs),
        report_run_id=finished,
    )


def project_or_404(project_id: int) -> Project:
    project = services().repositories.projects.get(project_id)
    if project is None:
        abort(404, description="Проєкт не знайдено.")
    return project


def source_or_404(project_id: int, source_id: int) -> SourceInfo:
    source = services().repositories.sources.source(project_id, source_id)
    if source is None:
        abort(404, description="Джерело не знайдено.")
    return source


def run_or_404(run_id: int) -> Run:
    run = services().repositories.runs.get(run_id)
    if run is None:
        abort(404, description="Запуск не знайдено.")
    return run


def finished_or_404(run_id: int) -> Run:
    run = run_or_404(run_id)
    if run.status is not RunStatus.FINISHED:
        abort(404, description="Звіт ще не готовий.")
    return run


def project_options(project: Project) -> RunOptions:
    return RunOptions.from_dict(project.options)


def flash_all(messages: Iterable[str], category: str = ERROR) -> None:
    for message in messages:
        flash(message, category)
