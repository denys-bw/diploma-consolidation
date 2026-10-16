from __future__ import annotations

import logging
from contextlib import closing
from dataclasses import fields
from pathlib import Path

import numpy as np

from consolidator.evaluation import evaluate_positions, gold_pairs, true_entities
from consolidator.linkage import PairScope, cluster_pairs
from consolidator.pipeline import Pipeline, PipelineResult
from consolidator.reporting import summarize
from consolidator.reporting.labels import STAGE_LABELS
from consolidator.schema import Attribute
from webapp.database import connect
from webapp.jobs import RunQueue
from webapp.models import Evaluation, QualityScore, Report, Run
from webapp.options import RunOptions
from webapp.repositories import Repositories
from webapp.repositories.runs import RunRepository
from webapp.services.mapping import MappingService
from webapp.services.reports import ReportWriter
from webapp.services.reviews import ReviewService
from webapp.services.sources import SourceService
from webapp.storage import Storage

logger = logging.getLogger(__name__)

REPORT_STAGE = "report"
STAGES = (*STAGE_LABELS, REPORT_STAGE)
RUN_STAGE_LABELS = {**STAGE_LABELS, REPORT_STAGE: "Формування звіту"}
NO_ATTRIBUTES = "У зіставленні немає жодного атрибута для обробки."
OUT_OF_MEMORY = "Недостатньо оперативної пам'яті для такого обсягу даних. Вимкніть частину джерел або атрибутів."


class RunError(ValueError):
    pass


class RunService:
    def __init__(
        self,
        repositories: Repositories,
        storage: Storage,
        queue: RunQueue,
        database: Path,
        mapping: MappingService,
    ) -> None:
        self.repositories = repositories
        self.storage = storage
        self.queue = queue
        self.database = database
        self.mapping = mapping

    def validate(self, project_id: int, options: RunOptions) -> list[str]:
        return run_problems(self.repositories, self.mapping, project_id, options)

    def start(self, project_id: int, options: RunOptions) -> int:
        run_id = self.repositories.runs.create(project_id, options.to_dict())
        self.queue.submit(ConsolidationJob(self.database, self.storage, run_id))
        return run_id

    def remove(self, run_id: int) -> None:
        self.repositories.runs.delete(run_id)
        self.storage.delete_run(run_id)


class RunProgress:
    def __init__(self, runs: RunRepository, run_id: int) -> None:
        self.runs = runs
        self.run_id = run_id

    def stage_started(self, stage: str) -> None:
        self.runs.set_stage(self.run_id, stage)


class ConsolidationJob:
    def __init__(self, database: Path, storage: Storage, run_id: int) -> None:
        self.database = database
        self.storage = storage
        self.run_id = run_id

    def __call__(self) -> None:
        with closing(connect(self.database)) as db:
            repositories = Repositories.on(db)
            run = repositories.runs.get(self.run_id)
            if run is None:
                return
            try:
                repositories.runs.start(run.id)
                self._execute(repositories, run)
            except MemoryError:
                self.storage.delete_run(run.id)
                repositories.runs.fail(run.id, OUT_OF_MEMORY)
            except Exception as error:
                logger.exception("Обробка №%s завершилася з помилкою", run.id)
                self.storage.delete_run(run.id)
                repositories.runs.fail(run.id, failure_message(error))

    def _execute(self, repositories: Repositories, run: Run) -> None:
        sources = SourceService(repositories, self.storage)
        mapping = MappingService(repositories, sources)
        reviews = ReviewService(repositories, self.storage)
        options = RunOptions.from_dict(run.options)
        problems = run_problems(repositories, mapping, run.project_id, options)
        if problems:
            raise RunError(" ".join(problems))
        infos = repositories.sources.sources(run.project_id, included_only=True)
        priority = [info.name for info in sorted(infos, key=lambda info: (info.priority, info.id))]
        resolutions = {
            state.name: state.resolution
            for state in repositories.attributes.load(run.project_id)
            if state.included and state.resolution
        }
        pipeline = Pipeline(options.pipeline_settings(resolutions, priority), [RunProgress(repositories.runs, run.id)])
        result = pipeline.run(
            sources.load_sources(run.project_id),
            mapping.mapping(run.project_id),
            review=reviews.clerical_review(run.project_id),
        )
        repositories.runs.set_stage(run.id, REPORT_STAGE)
        files = ReportWriter(self.storage.run_directory(run.id)).write(result)
        reference = mapping.reference(run.project_id)
        evaluation = evaluate(result, reference) if reference is not None else None
        repositories.runs.finish(report_of(run.id, result), scores_of(result), evaluation, files)


def run_problems(
    repositories: Repositories, mapping: MappingService, project_id: int, options: RunOptions
) -> list[str]:
    sources = repositories.sources.sources(project_id, included_only=True)
    problems = options.problems([source.record_count for source in sources])
    if sources and not mapping.mapping(project_id).attributes:
        problems.append(NO_ATTRIBUTES)
    return problems


def evaluate(result: PipelineResult, reference: Attribute) -> Evaluation:
    records = result.records
    truth = reference_labels(result, reference)
    scope = result.linkage.scope if result.linkage is not None else PairScope.ALL
    expected = gold_pairs(truth, records.source_codes, scope)
    found = cluster_pairs(result.fusion.entity_ids, records.source_codes, scope)
    return Evaluation(reference.name, true_entities(truth), evaluate_positions(found, expected, records.size))


def reference_labels(result: PipelineResult, reference: Attribute) -> np.ndarray:
    records = result.records
    labels = np.full(records.size, "", dtype=object)
    frames = {source.name: source.frame for source in result.sources}
    for code, name in enumerate(records.source_names):
        column = reference.column_in(name)
        if column is None:
            continue
        values = frames[name][column].fillna("").astype(str)
        values.index = values.index.astype(str)
        positions = np.flatnonzero(records.source_codes == code)
        labels[positions] = values.reindex(records.keys[positions]).fillna("").to_numpy(dtype=object)
    return labels


def report_of(run_id: int, result: PipelineResult) -> Report:
    summary = summarize(result)
    values = {field.name: getattr(summary, field.name) for field in fields(Report) if field.name != "run_id"}
    return Report(run_id=run_id, **values)


def scores_of(result: PipelineResult) -> list[QualityScore]:
    weights, before, after = result.settings.weights, result.quality_before, result.quality_after
    return [QualityScore(criterion, weights[criterion], before[criterion], after[criterion]) for criterion in before]


def failure_message(error: Exception) -> str:
    if isinstance(error, ValueError):
        return str(error)
    return f"Внутрішня помилка ({type(error).__name__}): {error}"
