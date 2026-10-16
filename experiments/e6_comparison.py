from __future__ import annotations

import time

import pandas as pd

from benchmarks import Dataset, Task
from benchmarks.dataset import ROOT
from consolidator.evaluation import (
    MatchMetrics,
    error_cells,
    evaluate_correction,
    evaluate_detection,
    evaluate_positions,
)
from consolidator.linkage import Linker
from consolidator.pipeline import Pipeline, PipelineResult
from consolidator.standardization import FormatPolicy, StandardizationSettings
from experiments.common import (
    MAIN_DATASETS,
    available,
    gold_positions,
    linkage_settings,
    linked_pairs,
    pipeline_settings,
    prepare,
    save,
    sources_of,
)

PUBLISHED = ROOT / "data" / "published"
LINKAGE_DATASETS = ["dblp_acm", "dblp_scholar", "abt_buy", "amazon_google"]
CLEANING_DATASETS = ["flights", "beers"]
ALL_DATASETS = [*MAIN_DATASETS, "amazon_google", "dblp_scholar", "flights", "beers"]
THIS_WORK = "ця робота: Фелегі–Зунтер і логістична регресія (спільне навчання), без розмітки"
THIS_WORK_CLEANING = "ця робота: уніфікація й злиття джерел, без розмітки"


def main() -> None:
    save(entity_resolution(), "e6_entity_resolution.csv")
    save(error_correction(), "e6_error_correction.csv")
    save(all_datasets(), "e6_all_datasets.csv")


def entity_resolution() -> pd.DataFrame:
    published = pd.read_csv(PUBLISHED / "kopcke_2010_entity_resolution.csv")
    rows = []
    for dataset in available(LINKAGE_DATASETS):
        print(f"[4.3.3] {dataset.title}")
        metrics, _ = _linkage(dataset)
        rows.append(
            {
                "набір": dataset.name,
                "підхід": THIS_WORK,
                "тип підходу": "без навчання",
                "атрибутів": "усі",
                "F1, %": 100 * metrics.f1,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "джерело": "results/e6_entity_resolution.csv",
            }
        )
    return pd.concat([pd.DataFrame(rows), published], ignore_index=True)


def error_correction() -> pd.DataFrame:
    published = pd.read_csv(PUBLISHED / "mahdavi_2020_error_correction.csv")
    rows = []
    for dataset in available(CLEANING_DATASETS):
        print(f"[4.3.3] {dataset.title}")
        detection, correction, _ = _cleaning(dataset)
        rows.append(
            {
                "набір": dataset.name,
                "система": THIS_WORK_CLEANING,
                "виявлення помилок": "без розмітки",
                "розмічених кортежів": 0,
                "precision": correction.precision,
                "recall": correction.recall,
                "F1": correction.f1,
                "precision виявлення": detection.precision,
                "recall виявлення": detection.recall,
                "F1 виявлення": detection.f1,
                "джерело": "results/e6_error_correction.csv",
            }
        )
    return pd.concat([pd.DataFrame(rows), published], ignore_index=True)


def all_datasets() -> pd.DataFrame:
    rows = []
    for dataset in available(ALL_DATASETS):
        print(f"[4.3.3] усі набори: {dataset.title}")
        if dataset.task is Task.CLEAN:
            _, metrics, seconds = _cleaning(dataset)
            measure = "виправлення клітинок"
            sources = "—"
        else:
            metrics, seconds = _linkage(dataset)
            measure = "пари дублікатів"
            sources = len(dataset.sources)
        rows.append(
            {
                "набір": dataset.name,
                "домен": dataset.domain,
                "записів": dataset.n_records,
                "джерел": sources,
                "що оцінюється": measure,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "F1": metrics.f1,
                "час, с": seconds,
            }
        )
    return pd.DataFrame(rows)


def _linkage(dataset: Dataset) -> tuple[MatchMetrics, float]:
    started = time.perf_counter()
    records = prepare(dataset).records
    result = Linker(linkage_settings()).run(records)
    seconds = time.perf_counter() - started
    return evaluate_positions(linked_pairs(dataset, result), gold_positions(dataset, records), records.size), seconds


def _cleaning(dataset: Dataset) -> tuple[MatchMetrics, MatchMetrics, float]:
    dirty, clean = dataset.sources["dirty"], dataset.sources["clean"]
    started = time.perf_counter()
    result = _cleaning_run(dataset)
    seconds = time.perf_counter() - started
    values = result.record_values()
    rows = values["key"].astype(int).to_numpy()
    repairs = {
        (int(row), column): value
        for column in result.attributes
        if column in dirty.columns
        for row, value in zip(rows, values[column])
    }
    original = {column: dirty[column].to_numpy(dtype=object) for column in dirty.columns}
    changed = {(row, column) for (row, column), value in repairs.items() if value != original[column][row]}
    errors = error_cells(dirty, clean)
    return evaluate_detection(changed, errors), evaluate_correction(repairs, dirty, clean), seconds


def _cleaning_run(dataset: Dataset) -> PipelineResult:
    dominant = StandardizationSettings(policy=FormatPolicy.DOMINANT)
    return Pipeline(pipeline_settings(standardization=dominant)).run(sources_of(dataset))


if __name__ == "__main__":
    main()
