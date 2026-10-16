from __future__ import annotations

from collections.abc import Mapping

from flask import Blueprint, flash, redirect, render_template, request, url_for
from werkzeug.wrappers import Response

from consolidator.profiling import ColumnType
from consolidator.reporting.labels import TYPE_LABELS
from webapp.models import AttributeRole
from webapp.options import OptionsError
from webapp.services.mapping import AttributeEdit, MappingError
from webapp.views.common import ERROR, NOTICE, SUCCESS, flash_all, project_options, project_or_404, services

SAVE = "save"
MERGE = "merge"
REBUILD = "rebuild"
DETACH = "detach"
ROLE_LABELS = {
    AttributeRole.ATTRIBUTE: "у результаті",
    AttributeRole.IGNORED: "не використовувати",
    AttributeRole.REFERENCE: "еталон",
}

blueprint = Blueprint("mapping", __name__, url_prefix="/projects/<int:project_id>/mapping")


@blueprint.get("")
def edit(project_id: int) -> str:
    project = project_or_404(project_id)
    repositories = services().repositories
    return render_template(
        "mapping/edit.html",
        project=project,
        sources=repositories.sources.sources(project_id, included_only=True),
        attributes=repositories.attributes.load(project_id),
        type_labels=TYPE_LABELS,
        role_labels=ROLE_LABELS,
        alpha=project_options(project).schema_alpha,
    )


@blueprint.post("")
def update(project_id: int) -> Response:
    project = project_or_404(project_id)
    action = request.form.get("action", SAVE)
    if action == REBUILD:
        return _rebuild(project_id, request.form.get("schema_alpha", ""))
    mapping = services().mapping
    try:
        edits, selected = _edits(request.form)
        mapping.save_edits(project_id, edits)
        if action == MERGE:
            flash(f"Атрибути об'єднано в «{mapping.merge(project_id, selected)}».", SUCCESS)
        elif action.startswith(f"{DETACH}:"):
            _, row, source_id = action.split(":")
            name = mapping.detach(project_id, edits[int(row)].name.strip(), int(source_id))
            flash(f"Колонку від'єднано в окремий атрибут «{name}».", SUCCESS)
        else:
            flash("Зміни зіставлення збережено.", SUCCESS)
    except (MappingError, ValueError, IndexError) as error:
        flash(str(error) or "Не вдалося застосувати зміни.", ERROR)
    services().repositories.projects.touch(project.id)
    return redirect(url_for(".edit", project_id=project_id))


def _rebuild(project_id: int, alpha: str) -> Response:
    project = project_or_404(project_id)
    try:
        options = project_options(project).with_schema_alpha(alpha)
    except OptionsError as error:
        flash_all(error.messages)
        return redirect(url_for(".edit", project_id=project_id))
    services().repositories.projects.save_options(project_id, options.to_dict())
    count = services().mapping.rebuild(project_id, options.schema_alpha)
    flash(f"Зіставлення перебудовано автоматично, атрибутів: {count}. Попередні правки скинуто.", NOTICE)
    return redirect(url_for(".edit", project_id=project_id))


def _edits(form: Mapping[str, str]) -> tuple[list[AttributeEdit], list[str]]:
    edits, selected = [], []
    for row in range(int(form.get("rows", "0"))):
        current = form[f"current_{row}"]
        name = form.get(f"name_{row}", current)
        edits.append(
            AttributeEdit(
                current=current,
                name=name,
                type=ColumnType(form.get(f"type_{row}", "")),
                role=AttributeRole(form.get(f"role_{row}", AttributeRole.ATTRIBUTE.value)),
            )
        )
        if f"selected_{row}" in form:
            selected.append(name.strip())
    return edits, selected
