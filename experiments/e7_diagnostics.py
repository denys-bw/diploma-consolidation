from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from consolidator.evaluation import best_threshold, evaluate_positions, gold_labels
from consolidator.linkage import FellegiSunterScorer, LinkageResult, Linker, MetricView, RecordSet
from experiments.common import (
    MAIN_DATASETS,
    SEED,
    available,
    gold_positions,
    linkage_settings,
    linked_pairs,
    prepare,
    save,
    scope_mask,
)

DATASETS = MAIN_DATASETS
FOLDS = 5
MAX_ITERATIONS = 3000


def main() -> None:
    rows = []
    for dataset in available(DATASETS):
        print(f"[діагностика] {dataset.title}")
        records = prepare(dataset).records
        gold = gold_positions(dataset, records)
        settings = linkage_settings()
        trained = Linker(settings).run(records)
        plain = Linker(replace(settings, scorer=FellegiSunterScorer(), blocker=trained.blocker)).run(records)
        scoped = scope_mask(dataset, records, trained.candidates)
        plain_scoped = scope_mask(dataset, records, plain.candidates)
        labels = gold_labels(trained.candidates[scoped], gold, records.size)
        plain_labels = gold_labels(plain.candidates[plain_scoped], gold, records.size)
        features = _features(records, trained)[scoped]
        total = len(gold)
        rows.append(
            {
                "набір": dataset.name,
                "пар-кандидатів": int(scoped.sum()),
                "дублікатів серед кандидатів (PC)": labels.sum() / total if total else 0.0,
                "Фелегі–Зунтер, найкращий поріг": _f1(plain.probabilities[plain_scoped], plain_labels, total),
                "дві моделі, найкращий поріг": _f1(trained.probabilities[scoped], labels, total),
                "логістична регресія з повною розміткою": _f1(
                    _supervised(LogisticRegression(max_iter=MAX_ITERATIONS), features, labels), labels, total
                ),
                "градієнтний бустинг з повною розміткою": _f1(
                    _supervised(HistGradientBoostingClassifier(random_state=SEED), features, labels), labels, total
                ),
                "Фелегі–Зунтер + групування": evaluate_positions(linked_pairs(dataset, plain), gold, records.size).f1,
                "дві моделі + групування": evaluate_positions(linked_pairs(dataset, trained), gold, records.size).f1,
            }
        )
    save(pd.DataFrame(rows), "e7_diagnostics.csv")


def _features(records: RecordSet, result: LinkageResult) -> np.ndarray:
    comparison = result.comparison
    levels = [
        (comparison.levels[:, index] == level).astype(np.float32)
        for index, size in enumerate(comparison.n_levels)
        for level in range(size)
    ]
    view = MetricView.for_records(records, comparison.attributes).compute(result.candidates).features()
    return StandardScaler().fit_transform(np.column_stack([*levels, view]))


def _supervised(model: object, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if labels.sum() < FOLDS:
        return np.zeros(len(labels))
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    return cross_val_predict(model, features, labels, cv=folds, method="predict_proba")[:, 1]


def _f1(scores: np.ndarray, labels: np.ndarray, total: int) -> float:
    return best_threshold(scores, labels, total_positives=total).f1


if __name__ == "__main__":
    main()
