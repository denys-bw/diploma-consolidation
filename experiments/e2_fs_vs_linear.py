from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from recordlinkage import ECMClassifier

from consolidator.evaluation import MatchMetrics, best_threshold, evaluate_positions, gold_labels
from consolidator.linkage import (
    ComparisonMatrix,
    FellegiSunterScorer,
    Linker,
    PairScope,
    comparable_attributes,
)
from consolidator.profiling import ColumnType
from consolidator.statistics import MAJORITY
from experiments.common import (
    CO_TRAINING,
    available,
    gold_positions,
    linkage_settings,
    linked_pairs,
    prepare,
    save,
    scope_mask,
)
from experiments.linear_model import LinearModel

DATASETS = ["restaurants", "chicago_sources", "musicbrainz", "nc_voters_sample"]
ARTICLE_WEIGHTS = {"name": 0.5, "phone": 0.4, "location": 0.1}
ROLE_TYPES = {
    "name": frozenset({ColumnType.TEXT}),
    "phone": frozenset({ColumnType.PHONE}),
    "location": frozenset({ColumnType.CATEGORY, ColumnType.CODE}),
}
ECM_BINARIZE = (0.85, 0.7)
DECISION_PROBABILITY = MAJORITY


def main() -> None:
    rows = []
    for dataset in available(DATASETS):
        print(f"[П2] {dataset.title}")
        records = prepare(dataset).records
        gold = gold_positions(dataset, records)
        attributes = comparable_attributes(records, PairScope.ALL)
        phones = [attribute for attribute in attributes if records.types[attribute] is ColumnType.PHONE]
        categories = [attribute for attribute in attributes if records.types[attribute] is ColumnType.CATEGORY]
        if phones:
            scenarios = {
                "з телефоном": attributes,
                "без телефону": [attribute for attribute in attributes if attribute not in phones],
                "без телефону й категорій": [
                    attribute for attribute in attributes if attribute not in phones and attribute not in categories
                ],
            }
        else:
            scenarios = {"телефону в даних немає": attributes}
        for scenario, compared in scenarios.items():
            trained_settings = replace(linkage_settings(), attributes=tuple(compared))
            trained = Linker(trained_settings).run(records)
            settings = replace(trained_settings, scorer=FellegiSunterScorer())
            result = Linker(settings).run(records)
            plain = Linker(replace(settings, scorer=FellegiSunterScorer(term_frequencies=False))).run(records)
            scoped = scope_mask(dataset, records, result.candidates)
            candidates = result.candidates[scoped]
            labels = gold_labels(candidates, gold, records.size)
            context = {"набір": dataset.name, "сценарій": scenario, "атрибутів": len(compared)}
            accepted = result.probabilities[scoped] >= DECISION_PROBABILITY
            outcomes = {
                "Фелегі–Зунтер, EM без розмітки": (
                    evaluate_positions(candidates[accepted], gold, records.size),
                    DECISION_PROBABILITY,
                ),
                "Фелегі–Зунтер + групування": (
                    evaluate_positions(linked_pairs(dataset, result), gold, records.size),
                    DECISION_PROBABILITY,
                ),
                "Фелегі–Зунтер + групування, без поправки на частоту значень": (
                    evaluate_positions(linked_pairs(dataset, plain), gold, records.size),
                    DECISION_PROBABILITY,
                ),
                CO_TRAINING: (
                    evaluate_positions(linked_pairs(dataset, trained), gold, records.size),
                    DECISION_PROBABILITY,
                ),
            }
            comparison = result.comparison.rows(scoped)
            for label, weights in _linear_models(records.types, compared).items():
                scores = LinearModel(weights).scores(comparison)
                choice = best_threshold(scores, labels, total_positives=len(gold))
                found = candidates[scores >= choice.threshold]
                outcomes[label] = (evaluate_positions(found, gold, records.size), choice.threshold)
            for binarize in ECM_BINARIZE:
                outcomes[f"recordlinkage ECMClassifier, ознаки ≥ {binarize}"] = (
                    evaluate_positions(_ecm(comparison, candidates, binarize), gold, records.size),
                    binarize,
                )
            for model, (metrics, threshold) in outcomes.items():
                rows.append({**context, "модель": model, "поріг": threshold, **_scores(metrics)})
    save(pd.DataFrame(rows), "e2_models.csv")


def _linear_models(types: dict[str, ColumnType], compared: list[str]) -> dict[str, dict[str, float]]:
    roles = {
        role: [attribute for attribute in compared if types[attribute] in kinds] for role, kinds in ROLE_TYPES.items()
    }
    models = {}
    if roles.get("phone"):
        models["лінійна з ручними вагами: 0,5·назва + 0,4·телефон + 0,1·місце"] = _weights(
            roles, ("name", "phone", "location")
        )
    elif roles.get("location"):
        models["лінійна з ручними вагами без телефону: 0,5·назва + 0,1·місце"] = _weights(roles, ("name", "location"))
    models["лінійна з ручними вагами: лише назва"] = _weights(roles, ("name",))
    models["лінійна: рівні ваги всіх атрибутів"] = dict.fromkeys(compared, 1.0)
    return {f"{label} (поріг за розміткою)": weights for label, weights in models.items()}


def _weights(roles: dict[str, list[str]], used: tuple[str, ...]) -> dict[str, float]:
    return {attribute: ARTICLE_WEIGHTS[role] / len(roles[role]) for role in used for attribute in roles.get(role, [])}


def _ecm(comparison: ComparisonMatrix, candidates: np.ndarray, binarize: float) -> np.ndarray:
    present = ~np.isnan(comparison.similarities)
    dense = present.mean(axis=0) > MAJORITY
    index = pd.MultiIndex.from_arrays([candidates[:, 0], candidates[:, 1]])
    features = pd.DataFrame(
        np.nan_to_num(comparison.similarities[:, dense], nan=0.0),
        index=index,
        columns=[attribute for attribute, keep in zip(comparison.attributes, dense) if keep],
    )
    matches = ECMClassifier(binarize=binarize).fit_predict(features)
    if len(matches) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(matches.tolist(), dtype=np.int64)


def _scores(metrics: MatchMetrics) -> dict[str, float]:
    return {
        "знайдено пар": metrics.found,
        "правильних": metrics.true_positives,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "F1": metrics.f1,
    }


if __name__ == "__main__":
    main()
