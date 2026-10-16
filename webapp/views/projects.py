from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for
from werkzeug.wrappers import Response

from consolidator.io import SUPPORTED_SUFFIXES
from webapp.models import AttributeRole
from webapp.services.projects import ProjectError
from webapp.views.common import ERROR, SUCCESS, project_or_404, services
from webapp.views.sources import layout_fields

MAX_NAME = 120
MAX_DESCRIPTION = 1000
ACCEPTED_FILES = ",".join(sorted(SUPPORTED_SUFFIXES))

blueprint = Blueprint("projects", __name__)


@blueprint.get("/")
def index() -> str:
    return render_template(
        "projects/index.html",
        projects=services().repositories.projects.overviews(),
        max_name=MAX_NAME,
        max_description=MAX_DESCRIPTION,
    )


@blueprint.post("/projects")
def create() -> Response:
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()
    if not name:
        flash("Вкажіть назву проєкту.", ERROR)
        return redirect(url_for(".index"))
    project_id = services().repositories.projects.create(name[:MAX_NAME], description[:MAX_DESCRIPTION])
    flash("Проєкт створено. Завантажте файли з даними.", SUCCESS)
    return redirect(url_for(".show", project_id=project_id))


@blueprint.get("/projects/<int:project_id>")
def show(project_id: int) -> str:
    project = project_or_404(project_id)
    repositories = services().repositories
    files = repositories.sources.files(project_id)
    tables = [table for uploaded in files for table in uploaded.tables]
    attributes = repositories.attributes.load(project_id)
    return render_template(
        "projects/show.html",
        project=project,
        files=files,
        included=[source for uploaded in files for source in uploaded.sources if source.included],
        attributes=[attribute for attribute in attributes if attribute.included],
        reference=next((attribute for attribute in attributes if attribute.role is AttributeRole.REFERENCE), None),
        runs=repositories.runs.list_for(project_id),
        accepted=ACCEPTED_FILES,
        layout=layout_fields(tables) if tables else (),
    )


@blueprint.post("/projects/<int:project_id>/delete")
def delete(project_id: int) -> Response:
    project = project_or_404(project_id)
    try:
        services().projects.delete(project_id)
    except ProjectError as error:
        flash(str(error), ERROR)
        return redirect(url_for(".show", project_id=project_id))
    flash(f"Проєкт «{project.name}» видалено разом із файлами й звітами.", SUCCESS)
    return redirect(url_for(".index"))
