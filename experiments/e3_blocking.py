from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from benchmarks import Dataset, NeedsDownload
from benchmarks.multisource import ENTITY, load_nc_voters, nc_voters_table
from consolidator.evaluation import evaluate_blocking, evaluate_positions
from consolidator.linkage import (
    Blocker,
    Comparison,
    ConnectedComponents,
    FullComparison,
    KeyBlocking,
    KeySelector,
    Linker,
    PairScope,
    RecordSet,
    SortedNeighbourhood,
    TokenBlocking,
    random_pairs,
    scale_neighbours,
)
from consolidator.linkage.key_selection import KeyAssessment, KeySuggestion
from consolidator.statistics import MAJORITY
from experiments.common import (
    SEED,
    available,
    gold_positions,
    linkage_settings,
    linked_pairs,
    prepare,
    save,
    scope_mask,
)

DATASETS = ["chicago_sources", "dblp_acm", "musicbrainz", "nc_voters_sample"]
SCOPE = PairScope.ALL
FULL_COMPARISON_LIMIT = 6_000_000
TIMING_SAMPLE = 200_000
SCALING_START = 10_000
STEP_TIME_LIMIT = 1800.0


def main() -> None:
    schemes, keys = [], []
    for dataset in available(DATASETS):
        print(f"[П3] {dataset.title}")
        records = prepare(dataset).records
        gold = gold_positions(dataset, records)
        selector = KeySelector(records, SCOPE)
        suggestion = selector.suggest()
        keys.extend(_key_rows(dataset, selector, suggestion.assessments, gold))
        per_pair = _comparison_seconds_per_pair(records)
        for label, blocker in _schemes(records, suggestion).items():
            schemes.append(_scheme_row(dataset, records, gold, label, blocker, per_pair))
    save(pd.DataFrame(schemes), "e3_blocking.csv")
    save(pd.DataFrame(keys), "e3_key_estimates.csv")
    save(_scaling(), "e3_scaling.csv")


def _schemes(records: RecordSet, suggestion: KeySuggestion) -> dict[str, Blocker | None]:
    affordable = [item for item in suggestion.assessments if 0 < item.candidates <= suggestion.budget]
    single = next((item for item in affordable if len(item.key.parts) == 1), None)
    composite = next((item for item in affordable if len(item.key.parts) > 1), None)
    neighbours = scale_neighbours(records)
    total = records.total_pairs(SCOPE)
    schemes: dict[str, Blocker | None] = {
        "повний перебір": FullComparison(FULL_COMPARISON_LIMIT) if total <= FULL_COMPARISON_LIMIT else None
    }
    if single is not None:
        schemes[f"стандартне, один ключ: {single.key.label}"] = KeyBlocking((single.key,))
    if composite is not None:
        schemes[f"стандартне, складений ключ: {composite.key.label}"] = KeyBlocking((composite.key,))
    if suggestion.keys:
        labels = " | ".join(key.label for key in suggestion.keys)
        schemes[f"кілька ключів (автоматично): {labels}"] = KeyBlocking(suggestion.keys)
        schemes[f"сортоване сусідство, w = {neighbours + 1}: {labels}"] = SortedNeighbourhood(suggestion.keys)
    schemes[f"TF-IDF, k = {neighbours} для кожного запису"] = TokenBlocking(neighbours=neighbours)
    schemes["TF-IDF, k добирається для кожного запису (типово)"] = TokenBlocking()
    return schemes


def _scheme_row(
    dataset: Dataset, records: RecordSet, gold: np.ndarray, label: str, blocker: Blocker | None, per_pair: float
) -> dict:
    total = records.total_pairs(SCOPE)
    row = {"набір": dataset.name, "схема": label, "усіх пар": total}
    if blocker is None:
        return {
            **row,
            "пар-кандидатів": total,
            "RR": 0.0,
            "PC": 1.0,
            "PQ": len(gold) / total,
            "блокування, с": 0.0,
            "уся обробка, с": per_pair * total,
            "F1 пар (p ≥ 0,5)": np.nan,
            "F1 після групування": np.nan,
            "примітка": "час порівняння всіх пар оцінено за вибіркою, обробку не виконано",
        }
    settings = replace(linkage_settings(), blocker=blocker)
    full = isinstance(blocker, FullComparison)
    if full:
        settings = replace(settings, clusterer=ConnectedComponents())
    result = Linker(settings).run(records)
    blocking = evaluate_blocking(result.candidates, gold, records.size, total)
    scoped = scope_mask(dataset, records, result.candidates)
    accepted = result.candidates[scoped & (result.probabilities >= MAJORITY)]
    clustered = np.nan if full else evaluate_positions(linked_pairs(dataset, result), gold, records.size).f1
    return {
        **row,
        "пар-кандидатів": blocking.candidates,
        "RR": blocking.reduction_ratio,
        "PC": blocking.pairs_completeness,
        "PQ": blocking.pairs_quality,
        "блокування, с": result.timings["blocking"],
        "уся обробка, с": sum(result.timings.values()),
        "F1 пар (p ≥ 0,5)": evaluate_positions(accepted, gold, records.size).f1,
        "F1 після групування": clustered,
        "примітка": f"λ = {result.prior:.4f}; групування не виконано" if full else "",
    }


def _key_rows(
    dataset: Dataset, selector: KeySelector, assessments: tuple[KeyAssessment, ...], gold: np.ndarray
) -> list[dict]:
    rows = []
    for item in assessments:
        groups = selector.groups(item.key)
        left, right = groups[gold[:, 0]], groups[gold[:, 1]]
        rows.append(
            {
                "набір": dataset.name,
                "ключ": item.key.label,
                "пар-кандидатів": item.candidates,
                "RR": item.reduction_ratio,
                "PC за оцінкою без розмітки": item.estimated_completeness,
                "PC за еталоном": float(((left >= 0) & (left == right)).mean()),
            }
        )
    return rows


def _comparison_seconds_per_pair(records: RecordSet, scope: PairScope = SCOPE) -> float:
    sample = random_pairs(records, scope, TIMING_SAMPLE, SEED)
    comparison = Comparison.for_records(records, scope)
    started = time.perf_counter()
    comparison.compare(sample)
    return (time.perf_counter() - started) / max(len(sample), 1)


def _scaling() -> pd.DataFrame:
    rows: list[dict] = []
    try:
        population = int(nc_voters_table()[ENTITY].nunique())
    except NeedsDownload as error:
        print(f"[пропущено] nc_voters: {error}")
        return pd.DataFrame(rows)
    entities = min(SCALING_START, population)
    previous = None
    while True:
        print(f"[П3] масштаб: {entities} людей")
        try:
            row = _scaling_row(load_nc_voters(entities))
        except MemoryError:
            rows.append({"людей": entities, "примітка": "не вистачило оперативної пам'яті"})
            break
        seconds = row["обробка з блокуванням, с"]
        growth = seconds / previous if previous else 2.0
        expected = seconds * growth
        rows.append({"людей": entities, **row, "очікуваний час наступного кроку, с": expected, "примітка": ""})
        if entities >= population or expected > STEP_TIME_LIMIT:
            break
        previous = seconds
        entities = min(entities * 2, population)
    return pd.DataFrame(rows)


def _scaling_row(dataset: Dataset) -> dict:
    records = prepare(dataset).records
    gold = gold_positions(dataset, records)
    per_pair = _comparison_seconds_per_pair(records, dataset.scope)
    started = time.perf_counter()
    result = Linker(linkage_settings()).run(records)
    seconds = time.perf_counter() - started
    total = records.total_pairs(dataset.scope)
    blocking = evaluate_blocking(result.candidates, gold, records.size, total)
    return {
        "записів": records.size,
        "усіх пар": total,
        "пар-кандидатів": blocking.candidates,
        "PC": blocking.pairs_completeness,
        "F1": evaluate_positions(linked_pairs(dataset, result), gold, records.size).f1,
        "обробка з блокуванням, с": seconds,
        "порівняння всіх пар (оцінка), с": per_pair * total,
    }


if __name__ == "__main__":
    main()
