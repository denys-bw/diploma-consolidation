from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, url_for
from werkzeug.wrappers import Response

from webapp.models import Project, SourceTable
from webapp.services.sources import Keep, SourceError
from webapp.views.common import ERROR, NOTICE, SUCCESS, project_options, project_or_404, services, source_or_404

DETAIL_LABELS = {
    "encoding": "Кодування",
    "delimiter": "Роздільник",
    "quoting": "Лапки",
    "sheet": "Аркуш",
    "table": "Таблиця",
    "split_from": "Виділено з таблиці",
    "split_by": "За колонкою",
}
KEEP = "keep"
NONE = "none"
COLUMN = "column:"
DELIMITER_NAMES = {",": "кома", ";": "крапка з комою", "\t": "табуляція", "|": "вертикальна риска", " ": "пробіл"}
QUOTING_NAMES = {"standard": "стандартні (RFC 4180)", "none": "не використовуються"}

blueprint = Blueprint("sources", __name__, url_prefix="/projects/<int:project_id>")


@dataclass(frozen=True)
class LayoutField:
    name: str
    label: str
    current: str
    options: tuple[tuple[str, str], ...]


def layout_fields(tables: Sequence[SourceTable]) -> tuple[LayoutField, LayoutField]:
    counts = Counter(column for table in tables for column in table.columns)
    columns = sorted(counts, key=lambda column: (-counts[column], column))
    labels = tuple(
        (
            f"{COLUMN}{column}",
            column if counts[column] == len(tables) else f"{column} (у {counts[column]} з {len(tables)})",
        )
        for column in columns
    )
    return (
        _layout_field(
            "key_column", "Ідентифікатор запису", "номер рядка в таблиці", [t.key_column for t in tables], labels
        ),
        _layout_field(
            "split_column", "Розділити на джерела за колонкою", "не розділяти", [t.split_column for t in tables], labels
        ),
    )


def _layout_field(
    name: str, label: str, empty: str, values: Sequence[str | None], columns: tuple[tuple[str, str], ...]
) -> LayoutField:
    chosen = set(values)
    if len(chosen) > 1:
        return LayoutField(name, label, KEEP, ((KEEP, "різні — не змінювати"), (NONE, empty), *columns))
    value = next(iter(chosen), None)
    return LayoutField(name, label, f"{COLUMN}{value}" if value else NONE, ((NONE, empty), *columns))


@blueprint.post("/files")
def upload(project_id: int) -> Response:
    project = project_or_404(project_id)
    uploads = [item for item in request.files.getlist("files") if item.filename]
    if not uploads:
        flash("Виберіть щонайменше один файл.", ERROR)
        return redirect(url_for("projects.show", project_id=project_id))
    outcome = services().sources.add_uploads(project_id, uploads)
    for message in outcome.errors:
        flash(message, ERROR)
    if outcome.added:
        attributes = _rebuild(project)
        flash(
            f"Додано джерел: {len(outcome.added)} ({', '.join(outcome.added)}). "
            f"Колонки зіставлено автоматично, атрибутів: {attributes}.",
            SUCCESS,
        )
    return redirect(url_for("projects.show", project_id=project_id))


@blueprint.post("/files/<int:file_id>/delete")
def delete_file(project_id: int, file_id: int) -> Response:
    project = project_or_404(project_id)
    uploaded = services().repositories.sources.file(project_id, file_id)
    if uploaded is None:
        abort(404, description="Файл не знайдено.")
    services().sources.remove_file(uploaded)
    _rebuild(project)
    flash(f"Файл «{uploaded.original_name}» видалено, зіставлення колонок перебудовано.", NOTICE)
    return redirect(url_for("projects.show", project_id=project_id))


@blueprint.post("/sources/<int:source_id>/toggle")
def toggle(project_id: int, source_id: int) -> Response:
    project = project_or_404(project_id)
    source = source_or_404(project_id, source_id)
    services().sources.toggle(source)
    _rebuild(project)
    state = "вимкнено" if source.included else "увімкнено"
    flash(f"Джерело «{source.name}» {state}, зіставлення колонок перебудовано.", NOTICE)
    return redirect(url_for("projects.show", project_id=project_id))


@blueprint.get("/sources/<int:source_id>")
def show(project_id: int, source_id: int) -> str:
    project = project_or_404(project_id)
    source = source_or_404(project_id, source_id)
    repositories = services().repositories
    table = repositories.sources.table(project_id, source.table_id)
    loaded = services().sources.load_source(project_id, source)
    preview = loaded.frame.head(current_app.config["SOURCE_PREVIEW_ROWS"]).fillna("").astype(str)
    return render_template(
        "sources/show.html",
        project=project,
        source=source,
        table=table,
        uploaded=repositories.sources.file(project_id, source.file_id),
        details=_details({**table.details, **loaded.details}),
        profile=repositories.sources.profile(source_id),
        columns=list(preview.columns),
        rows=list(zip(preview.index.tolist(), preview.itertuples(index=False, name=None))),
        key_name=loaded.frame.index.name,
    )


@blueprint.post("/sources/<int:source_id>/table")
def configure(project_id: int, source_id: int) -> Response:
    project = project_or_404(project_id)
    source = source_or_404(project_id, source_id)
    table = services().repositories.sources.table(project_id, source.table_id)
    key_column = request.form.get("key_column") or None
    split_column = request.form.get("split_column") or None
    try:
        names = services().sources.configure(project_id, table, key_column, split_column)
    except SourceError as error:
        flash(str(error), ERROR)
        return redirect(url_for(".show", project_id=project_id, source_id=source_id))
    attributes = _rebuild(project)
    flash(
        f"Таблицю «{table.name}» оновлено. Джерел: {len(names)}, атрибутів: {attributes}. "
        "Зіставлення колонок перебудовано.",
        SUCCESS,
    )
    return redirect(url_for("projects.show", project_id=project_id))


@blueprint.post("/tables")
def configure_all(project_id: int) -> Response:
    project = project_or_404(project_id)
    key_column = _layout_value(request.form.get("key_column", KEEP))
    split_column = _layout_value(request.form.get("split_column", KEEP))
    if key_column is not Keep.CURRENT and key_column is not None and key_column == split_column:
        flash("Одна колонка не може бути одночасно ідентифікатором запису і ознакою джерела.", ERROR)
        return redirect(url_for("projects.show", project_id=project_id))
    outcome = services().sources.configure_all(project_id, key_column, split_column)
    for message in outcome.errors:
        flash(message, ERROR)
    if outcome.skipped:
        flash(f"Без змін, бо в таблиці немає вибраної колонки: {', '.join(outcome.skipped)}.", NOTICE)
    if outcome.updated:
        attributes = _rebuild(project)
        flash(
            f"Структуру оновлено для таблиць: {len(outcome.updated)}. "
            f"Зіставлення колонок перебудовано, атрибутів: {attributes}.",
            SUCCESS,
        )
    elif not outcome.errors and not outcome.skipped:
        flash("Структура таблиць уже така, як вибрано.", NOTICE)
    return redirect(url_for("projects.show", project_id=project_id))


def _layout_value(raw: str) -> str | None | Keep:
    if raw.startswith(COLUMN):
        return raw.removeprefix(COLUMN)
    return None if raw == NONE else Keep.CURRENT


def _rebuild(project: Project) -> int:
    count = services().mapping.rebuild(project.id, project_options(project).schema_alpha)
    services().repositories.projects.touch(project.id)
    return count


def _details(details: dict[str, str]) -> list[tuple[str, str]]:
    rows = []
    for key, label in DETAIL_LABELS.items():
        value = details.get(key)
        if value is None:
            continue
        if key == "delimiter":
            value = DELIMITER_NAMES.get(value, f"«{value}»")
        elif key == "quoting":
            value = QUOTING_NAMES.get(value, value)
        rows.append((label, value))
    return rows
