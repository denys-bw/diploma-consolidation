from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from benchmarks import Dataset, Task
from consolidator.evaluation import entity_uniqueness
from consolidator.linkage import components
from consolidator.pipeline import Pipeline, PipelineResult
from consolidator.quality import DataSnapshot, QualityScores, QualityWeights, RecordCompleteness
from consolidator.standardization import FormatPolicy, StandardizationSettings
from experiments.common import available, gold_positions, pipeline_settings, save, sources_of

DATASETS = ["chicago_sources", "flights", "beers", "restaurants"]
WEIGHTS = {"МАІ": QualityWeights.default(), "рівні": QualityWeights.equal()}


def main() -> None:
    rows = []
    for dataset in available(DATASETS):
        print(f"[П5] {dataset.title}")
        result, truth = run(dataset)
        rows.extend(_rows(dataset, result, truth))
    save(pd.DataFrame(rows), "e5_quality.csv")


def run(dataset: Dataset) -> tuple[PipelineResult, np.ndarray]:
    dominant = StandardizationSettings(policy=FormatPolicy.DOMINANT)
    result = Pipeline(pipeline_settings(standardization=dominant)).run(sources_of(dataset))
    if dataset.task is not Task.CLEAN:
        return result, components(result.records.size, gold_positions(dataset, result.records))
    entity = dataset.extra.get("layout", {}).get("entity_column")
    if entity is None:
        return result, np.arange(result.records.size)
    entities = dataset.sources["clean"][entity].to_numpy()
    return result, pd.factorize(entities[result.records.keys.astype(int)])[0]


def _rows(dataset: Dataset, result: PipelineResult, truth: np.ndarray) -> list[dict]:
    records = result.records.size
    aligned = [frame.reindex(columns=result.attributes) for frame in result.prepared.frames.values()]
    consolidated = result.consolidated[result.attributes]
    stages = {
        "до обробки": (
            result.quality_before,
            _record_completeness(aligned),
            len(np.unique(truth)) / records,
            records,
            0,
        ),
        "після обробки": (
            result.quality_after,
            _record_completeness([consolidated]),
            entity_uniqueness(result.fusion.entity_ids, truth),
            result.fusion.n_entities,
            result.imputation.count if result.imputation is not None else 0,
        ),
    }
    rows = []
    for stage, (scores, by_records, uniqueness, rows_count, estimated) in stages.items():
        with_gold = scores.replaced("uniqueness", uniqueness)
        record_level = with_gold.replaced("completeness", by_records)
        row = {
            "набір": dataset.name,
            "етап": stage,
            "рядків": rows_count,
            "повнота (клітинки)": scores["completeness"],
            "клітинок, відновлених статистично": estimated,
            "повнота (повні записи)": by_records,
            "узгодженість": scores["consistency"],
            "унікальність (за еталоном)": uniqueness,
            "унікальність (оцінка програми)": scores["uniqueness"],
        }
        row.update(_indices(with_gold, WEIGHTS))
        row["індекс з повнотою за записами, ваги МАІ"] = record_level.index(WEIGHTS["МАІ"])
        rows.append(row)
    return rows


def _indices(scores: QualityScores, weights: Mapping[str, QualityWeights]) -> dict[str, float]:
    return {f"індекс, ваги {label}": scores.index(weight) for label, weight in weights.items()}


def _record_completeness(frames: Sequence[pd.DataFrame]) -> float:
    return RecordCompleteness().measure(DataSnapshot(frames, (), {}, 0, 0))


if __name__ == "__main__":
    main()
