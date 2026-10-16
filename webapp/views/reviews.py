from __future__ import annotations

from collections.abc import Mapping

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from werkzeug.wrappers import Response

from webapp.models import PairDecision
from webapp.options import RunOptions
from webapp.views.common import SUCCESS, finished_or_404, project_or_404, services
from webapp.views.runs import start_run

SAVE = "save"
NEXT = "next"
RERUN = "rerun"
DECISIONS = {"match": True, "different": False}

blueprint = Blueprint("reviews", __name__)


@blueprint.get("/runs/<int:run_id>/review")
def show(run_id: int) -> str:
    run = finished_or_404(run_id)
    number = request.args.get("page", 1, type=int)
    page = services().reviews.page(run, number, current_app.config["REVIEW_PAGE_SIZE"])
    return render_template("runs/review.html", project=project_or_404(run.project_id), run=run, page=page)


@blueprint.post("/runs/<int:run_id>/review")
def save(run_id: int) -> Response:
    run = finished_or_404(run_id)
    services().reviews.save(run.project_id, _decisions(request.form))
    action = request.form.get("action", SAVE)
    if action == RERUN:
        return start_run(run.project_id, RunOptions.from_dict(run.options))
    flash("Рішення збережено. Під час наступного запуску модель навчиться на них.", SUCCESS)
    number = request.form.get("page", 1, type=int) + (action == NEXT)
    return redirect(url_for(".show", run_id=run_id, page=number))


def _decisions(form: Mapping[str, str]) -> list[PairDecision]:
    return [
        PairDecision(
            form[f"left_source_{row}"],
            form[f"left_key_{row}"],
            form[f"right_source_{row}"],
            form[f"right_key_{row}"],
            DECISIONS.get(form.get(f"decision_{row}", "")),
        )
        for row in range(int(form.get("pairs", "0")))
    ]
