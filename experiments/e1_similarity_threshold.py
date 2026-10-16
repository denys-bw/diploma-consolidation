from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from benchmarks import Dataset
from consolidator.evaluation import average_precision, best_threshold, evaluate_positions, gold_labels
from consolidator.linkage import (
    COMPARATORS,
    ClericalReview,
    DecisionRule,
    LinkageResult,
    Linker,
    MatchStatus,
    RecordSet,
    candidate_metrics,
)
from consolidator.linkage.comparison import TEXT_METRICS
from experiments.common import (
    MAIN_DATASETS,
    available,
    gold_positions,
    linkage_settings,
    linked_pairs,
    prepare,
    save,
    scope_mask,
)

DATASETS = MAIN_DATASETS
TOLERANCES = (0.10, 0.05, 0.01)


def main() -> None:
    metric_rows, decision_rows, review_rows = [], [], []
    for dataset in available(DATASETS):
        print(f"[П1] {dataset.title}")
        records = prepare(dataset).records
        gold = gold_positions(dataset, records)
        result = Linker(linkage_settings()).run(records)
        scoped = scope_mask(dataset, records, result.candidates)
        labels = gold_labels(result.candidates[scoped], gold, records.size)
        metric_rows.extend(_metric_rows(dataset, records, result, scoped, labels))
        decision_rows.append(_decision_row(dataset, records, result, scoped, gold, labels))
        review_rows.extend(_review_rows(dataset, records, result, gold))
    metrics = pd.DataFrame(metric_rows)
    save(metrics, "e1_metrics.csv")
    save(_summary_by_type(metrics), "e1_metrics_by_type.csv")
    save(_choice_summary(metrics), "e1_metric_choice.csv")
    save(pd.DataFrame(decision_rows), "e1_decision_threshold.csv")
    save(pd.DataFrame(review_rows), "e1_review_zone.csv")


def _metric_rows(
    dataset: Dataset, records: RecordSet, result: LinkageResult, scoped: np.ndarray, labels: np.ndarray
) -> list[dict]:
    candidates = result.candidates[scoped]
    used = dict(zip(result.comparison.attributes, result.comparison.metrics))
    rows = []
    for attribute in result.comparison.attributes:
        column_type = records.types[attribute]
        own = candidate_metrics(column_type)
        tested = list(dict.fromkeys([*own, *(TEXT_METRICS if column_type.is_textual else ())]))
        choice = result.metric_choices.get(attribute)
        for metric in tested:
            comparator = COMPARATORS[metric]().fit(records.values(attribute))
            similarity = comparator.similarity(candidates[:, 0], candidates[:, 1])
            present = ~np.isnan(similarity)
            matches = int(labels[present].sum())
            if matches == 0:
                continue
            best = best_threshold(similarity[present], labels[present])
            rows.append(
                {
                    "набір": dataset.name,
                    "атрибут": attribute,
                    "тип": column_type.value,
                    "метрика": metric,
                    "серед кандидатів програми": metric in own,
                    "обрано програмою": used[attribute] == metric,
                    "оцінка програми без розмітки (AUC)": (choice.scores.get(metric, np.nan) if choice else np.nan),
                    "пар": int(present.sum()),
                    "дублікатів": matches,
                    "AP": average_precision(similarity[present], labels[present]),
                    "найкращий F1": best.f1,
                    "поріг": best.threshold,
                    "precision": best.precision,
                    "recall": best.recall,
                }
            )
    return rows


def _decision_row(
    dataset: Dataset,
    records: RecordSet,
    result: LinkageResult,
    scoped: np.ndarray,
    gold: np.ndarray,
    labels: np.ndarray,
) -> dict:
    probabilities = result.probabilities[scoped]
    candidates = result.candidates[scoped]
    threshold = result.settings.decision.match_probability
    default = evaluate_positions(candidates[probabilities >= threshold], gold, records.size)
    best = best_threshold(probabilities, labels, total_positives=len(gold))
    clustered = evaluate_positions(linked_pairs(dataset, result), gold, records.size)
    accepted = result.accepted[scoped]
    return {
        "набір": dataset.name,
        "пар-кандидатів": len(candidates),
        "дублікатів серед кандидатів": int(labels.sum()),
        "дублікатів усього": len(gold),
        "апріорна частка λ": result.prior,
        "F1 при p ≥ 0,5": default.f1,
        "precision при p ≥ 0,5": default.precision,
        "recall при p ≥ 0,5": default.recall,
        "найкращий поріг p": best.threshold,
        "F1 при найкращому порозі": best.f1,
        "втрата F1 через поріг 0,5": best.f1 - default.f1,
        "AP імовірностей": average_precision(probabilities, labels),
        "F1 після групування": clustered.f1,
        "очікувано хибних збігів (модель)": float((1.0 - result.effective_probabilities[scoped][accepted]).sum()),
        "хибних збігів (еталон)": int((~labels[accepted]).sum()),
    }


def _review_rows(dataset: Dataset, records: RecordSet, result: LinkageResult, gold: np.ndarray) -> list[dict]:
    automatic = evaluate_positions(linked_pairs(dataset, result), gold, records.size)
    truth = {tuple(pair) for pair in np.sort(gold, axis=1).tolist()}
    rows = []
    for tolerance in TOLERANCES:
        rule = DecisionRule(result.settings.decision.match_probability, tolerance, tolerance)
        statuses = rule.statuses(result.effective_probabilities, result.contested)
        possible = np.flatnonzero(statuses == MatchStatus.POSSIBLE)
        review = ClericalReview.of(
            (records.record_id(int(left)), records.record_id(int(right)), (int(left), int(right)) in truth)
            for left, right in result.candidates[possible]
        )
        rerun = Linker(replace(result.settings, decision=rule)).run(records, review)
        checked = evaluate_positions(linked_pairs(dataset, rerun), gold, records.size)
        candidates = len(result.candidates)
        rows.append(
            {
                "набір": dataset.name,
                "допустима частка помилок": tolerance,
                "пар-кандидатів": candidates,
                "пар у зоні ручної перевірки": len(possible),
                "частка кандидатів": len(possible) / candidates if candidates else 0.0,
                "з них дублікатів": sum((int(a), int(b)) in truth for a, b in result.candidates[possible]),
                "F1 без перевірки": automatic.f1,
                "F1 після перевірки": checked.f1,
                "приріст F1": checked.f1 - automatic.f1,
            }
        )
    return rows


def _summary_by_type(metrics: pd.DataFrame) -> pd.DataFrame:
    cases = metrics.groupby(["набір", "атрибут"])
    best = metrics.loc[cases["AP"].idxmax(), ["набір", "атрибут", "метрика"]]
    wins = best.merge(metrics[["набір", "атрибут", "тип"]].drop_duplicates(), on=["набір", "атрибут"])
    wins = wins.groupby(["тип", "метрика"]).size().rename("перемог")
    summary = metrics.groupby(["тип", "метрика"]).agg(
        **{
            "випадків": ("AP", "size"),
            "середня AP": ("AP", "mean"),
            "мінімальна AP": ("AP", "min"),
            "середній найкращий F1": ("найкращий F1", "mean"),
            "середній поріг": ("поріг", "mean"),
            "розкид порогу": ("поріг", "std"),
            "разів обрано програмою": ("обрано програмою", "sum"),
        }
    )
    summary = summary.join(wins).fillna({"перемог": 0}).reset_index()
    summary["перемог"] = summary["перемог"].astype(int)
    return summary.sort_values(["тип", "середня AP"], ascending=[True, False])


def _choice_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, attribute), group in metrics[metrics["серед кандидатів програми"]].groupby(["набір", "атрибут"]):
        if len(group) < 2:
            continue
        chosen = group[group["обрано програмою"]]
        best = group.loc[group["AP"].idxmax()]
        chosen_ap = float(chosen["AP"].iloc[0]) if len(chosen) else np.nan
        rows.append(
            {
                "набір": dataset,
                "атрибут": attribute,
                "тип": best["тип"],
                "обрано програмою": chosen["метрика"].iloc[0] if len(chosen) else "",
                "AP обраної": chosen_ap,
                "найкраща за еталоном": best["метрика"],
                "AP найкращої": float(best["AP"]),
                "втрата AP": float(best["AP"]) - chosen_ap,
                "ранг обраної": int((group["AP"] > chosen_ap).sum()) + 1 if len(chosen) else np.nan,
                "метрик-кандидатів": len(group),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
