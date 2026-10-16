from __future__ import annotations

import re
from collections.abc import Mapping

from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.wrappers import Response

from consolidator.fusion import RESOLVERS, default_resolver
from consolidator.quality import CRITERION_KEYS
from consolidator.reporting.labels import CRITERION_LABELS, RESOLVER_LABELS, TYPE_LABELS
from webapp.models import Project, Run, RunStatus
from webapp.options import (
    AHP_PAIRS,
    BLOCKING_CHOICES,
    COMMON_REGIONS,
    DEFAULT_PRESET,
    DUPLICATE_CHOICES,
    EQUAL_PRESET,
    IMPUTATION_CHOICES,
    MODEL_CHOICES,
    POLICY_CHOICES,
    SAATY_VALUES,
    OptionsError,
    RunOptions,
    ahp_for,
    saaty_label,
)
from webapp.services.reports import DOWNLOADS, FLAG_LABELS, ReportReader
from webapp.services.runs import RUN_STAGE_LABELS, STAGES
from webapp.views.common import (
    ERROR,
    NOTICE,
    SUCCESS,
    finished_or_404,
    flash_all,
    project_options,
    project_or_404,
    run_or_404,
    services,
)

RUN = "run"
UNSAFE_FILE_CHARACTERS = re.compile(r"[^\w\-]+")
REPORT_TABLES = (
    "sources",
    "mapping",
    "standardization",
    "linkage",
    "model",
    "fusion",
    "accuracy",
    "conflicts",
    "result_profile",
    "settings",
    "timings",
)
TABLE_ROWS = 200
QUEUED_LABEL = "Очікування в черзі"
STARTING_LABEL = "Підготовка"
FINISHED_LABELS = {RunStatus.FINISHED: "Обробку завершено", RunStatus.FAILED: "Обробку перервано"}

blueprint = Blueprint("runs", __name__)


@blueprint.get("/projects/<int:project_id>/settings")
def settings(project_id: int) -> str:
    project = project_or_404(project_id)
    return _settings_page(project, project_options(project), {})


@blueprint.post("/projects/<int:project_id>/settings")
def save_settings(project_id: int) -> Response | tuple[str, int]:
    project = project_or_404(project_id)
    repositories = services().repositories
    try:
        options = RunOptions.from_form(request.form, project_options(project))
    except OptionsError as error:
        flash_all(error.messages)
        return _settings_page(project, project_options(project), request.form), 400
    repositories.projects.save_options(project_id, options.to_dict())
    repositories.attributes.set_resolutions(project_id, _resolutions(request.form))
    repositories.sources.set_priorities(_priorities(request.form, project_id))
    if request.form.get("action") != RUN:
        flash("Налаштування збережено.", SUCCESS)
        return redirect(url_for(".settings", project_id=project_id))
    return start_run(project_id, options)


@blueprint.get("/api/ahp")
def ahp_preview() -> Response:
    values = [request.args.get(f"ahp_{left}_{right}", "1") for left, right in AHP_PAIRS]
    if any(value not in SAATY_VALUES for value in values):
        abort(400, description="Недопустима оцінка попарного порівняння.")
    result = ahp_for(values)
    return jsonify(
        weights=result.as_dict(),
        lambda_max=result.lambda_max,
        consistency_index=result.consistency_index,
        consistency_ratio=result.consistency_ratio,
        consistent=result.consistent,
    )


@blueprint.get("/runs/<int:run_id>")
def show(run_id: int) -> str:
    run = run_or_404(run_id)
    project = project_or_404(run.project_id)
    if run.status is not RunStatus.FINISHED:
        return render_template(
            "runs/progress.html", project=project, run=run, stage_labels=RUN_STAGE_LABELS, **_progress(run)
        )
    repositories = services().repositories
    reader = ReportReader(services().storage.run_directory(run_id))
    config = current_app.config
    duplicates, duplicate_total = reader.duplicates(config["DUPLICATE_ROWS"])
    return render_template(
        "runs/report.html",
        project=project,
        run=run,
        ahp=RunOptions.from_dict(run.options).ahp(),
        report=repositories.runs.report(run_id),
        scores=repositories.runs.scores(run_id),
        evaluation=repositories.runs.evaluation(run_id),
        criterion_labels=CRITERION_LABELS,
        files=repositories.runs.files(run_id),
        downloads=DOWNLOADS,
        tables={key: reader.table(key) for key in REPORT_TABLES},
        table_rows=TABLE_ROWS,
        preview=reader.preview(config["PREVIEW_ROWS"]),
        duplicates=duplicates,
        duplicate_total=duplicate_total,
        reliability=reader.reliability(),
        flag_labels=_flag_labels(),
    )


@blueprint.get("/runs/<int:run_id>/status")
def status(run_id: int) -> Response:
    run = run_or_404(run_id)
    return jsonify(status=run.status.value, done=run.status.done, error=run.error, **_progress(run))


@blueprint.get("/runs/<int:run_id>/entities/<int:entity>")
def entity(run_id: int, entity: int) -> str:
    run = finished_or_404(run_id)
    reader = ReportReader(services().storage.run_directory(run_id))
    view = reader.entity(entity)
    if view is None:
        abort(404, description="Сутність не знайдено.")
    return render_template(
        "runs/entity.html",
        project=project_or_404(run.project_id),
        run=run,
        entity=view,
        entity_count=reader.entity_count(),
        flag_labels=_flag_labels(),
    )


@blueprint.get("/runs/<int:run_id>/download/<kind>")
def download(run_id: int, kind: str) -> Response:
    run = finished_or_404(run_id)
    if kind not in DOWNLOADS:
        abort(404, description="Такого файлу немає.")
    project = project_or_404(run.project_id)
    item = DOWNLOADS[kind]
    path = services().storage.run_directory(run_id) / item.file_name
    if not path.is_file():
        abort(404, description="Файл звіту не знайдено.")
    prefix = UNSAFE_FILE_CHARACTERS.sub("_", project.name).strip("_") or "project"
    return send_file(
        path, mimetype=item.mimetype, as_attachment=True, download_name=f"{prefix}_{run_id}_{item.file_name}"
    )


@blueprint.post("/runs/<int:run_id>/delete")
def delete(run_id: int) -> Response:
    run = run_or_404(run_id)
    if not run.status.done:
        flash("Запуск ще виконується, його не можна видалити.", ERROR)
        return redirect(url_for(".show", run_id=run_id))
    services().runs.remove(run_id)
    flash(f"Запуск №{run_id} і його звіт видалено.", NOTICE)
    return redirect(url_for("projects.show", project_id=run.project_id))


def start_run(project_id: int, options: RunOptions) -> Response:
    problems = services().runs.validate(project_id, options)
    if problems:
        flash_all(problems)
        return redirect(url_for("runs.settings", project_id=project_id))
    run_id = services().runs.start(project_id, options)
    return redirect(url_for("runs.show", run_id=run_id))


def _settings_page(project: Project, options: RunOptions, submitted: Mapping[str, str]) -> str:
    repositories = services().repositories
    sources = sorted(
        repositories.sources.sources(project.id, included_only=True), key=lambda item: (item.priority, item.id)
    )
    attributes = [state for state in repositories.attributes.load(project.id) if state.included]
    return render_template(
        "runs/settings.html",
        project=project,
        options=options,
        submitted=submitted,
        ahp=options.ahp(),
        sources=sources,
        attributes=attributes,
        problems=services().runs.validate(project.id, options),
        default_resolvers={state.name: default_resolver(state.type).key for state in attributes},
        resolvers=list(RESOLVERS),
        resolver_labels=RESOLVER_LABELS,
        type_labels=TYPE_LABELS,
        policy_choices=POLICY_CHOICES,
        duplicate_choices=DUPLICATE_CHOICES,
        blocking_choices=BLOCKING_CHOICES,
        model_choices=MODEL_CHOICES,
        imputation_choices=IMPUTATION_CHOICES,
        regions=COMMON_REGIONS,
        pairs=[
            (f"ahp_{left}_{right}", CRITERION_LABELS[left], CRITERION_LABELS[right], value)
            for (left, right), value in zip(AHP_PAIRS, options.judgements)
        ],
        saaty=[(value, [saaty_label(value, left, right) for left, right in AHP_PAIRS]) for value in SAATY_VALUES],
        presets={"default": list(DEFAULT_PRESET), "equal": list(EQUAL_PRESET)},
        criteria=[(criterion, CRITERION_LABELS[criterion]) for criterion in CRITERION_KEYS],
    )


def _flag_labels() -> dict[str, str]:
    return {flag.value: label for flag, label in FLAG_LABELS.items()}


def _resolutions(form: Mapping[str, str]) -> dict[str, str | None]:
    chosen = {}
    for row in range(int(form.get("attribute_rows", "0"))):
        name = form.get(f"attribute_{row}")
        value = form.get(f"resolution_{row}", "")
        if name:
            chosen[name] = value if value in RESOLVERS else None
    return chosen


def _priorities(form: Mapping[str, str], project_id: int) -> dict[int, int]:
    known = {source.id for source in services().repositories.sources.sources(project_id)}
    priorities = {}
    for key, value in form.items():
        source = key.removeprefix("priority_")
        number = value.strip()
        if key.startswith("priority_") and source.isdigit() and number.lstrip("-").isdigit() and int(source) in known:
            priorities[int(source)] = int(number)
    return priorities


def _progress(run: Run) -> dict[str, object]:
    stage = run.stage if run.stage in STAGES else ""
    if run.status.done:
        return {"stage": stage, "stage_label": FINISHED_LABELS[run.status], "progress": 1.0}
    if run.status is RunStatus.QUEUED:
        return {"stage": stage, "stage_label": QUEUED_LABEL, "progress": 0.0}
    position = STAGES.index(stage) + 1 if stage else 0
    return {
        "stage": stage,
        "stage_label": RUN_STAGE_LABELS.get(stage, STARTING_LABEL),
        "progress": position / (len(STAGES) + 1),
    }
