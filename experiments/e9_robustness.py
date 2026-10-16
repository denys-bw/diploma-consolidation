from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from benchmarks import Dataset
from consolidator.evaluation import MatchMetrics, evaluate_positions
from consolidator.linkage import Linker
from consolidator.pipeline import PreparedData
from consolidator.schema import SchemaMapping
from consolidator.schema.mapping import COMPOSITE_SEPARATOR, joined_values
from experiments.common import SEED, available, gold_positions, linkage_settings, linked_pairs, prepare, save

DATASETS = ["restaurants", "dblp_acm", "abt_buy", "chicago_sources", "settlements", "musicbrainz"]
SHARED = 0.2
NORMAL = "звичайні дані"
HIDDEN = "назви колонок нічого не кажуть"
FEW_SHARED = f"спільних об'єктів лише {int(SHARED * 100)} %"
HIDDEN_NAME = "колонка {}"
JOINED = "в одному джерелі дві колонки записано одним полем"

Renames = dict[str, dict[str, str]]


def main() -> None:
    rows = []
    for dataset in available(DATASETS):
        print(f"[стійкість] {dataset.title}")
        normal, normal_prepared = _run(dataset)
        rows.append(_row(dataset, NORMAL, normal))
        hidden_dataset, renames = hide_column_names(dataset)
        hidden, hidden_prepared = _run(hidden_dataset)
        expected = correspondences(normal_prepared.mapping)
        found = correspondences(hidden_prepared.mapping, renames)
        matched = len(expected & found)
        rows.append(
            _row(
                dataset,
                HIDDEN,
                hidden,
                columns_found=matched / len(expected) if expected else 1.0,
                columns_precision=matched / len(found) if found else 1.0,
            )
        )
        reduced = reduce_overlap(dataset, SHARED)
        if reduced is not None:
            rows.append(_row(reduced, FEW_SHARED, _run(reduced)[0]))
        pair = joinable_columns(dataset, normal_prepared.mapping)
        if pair is not None:
            joined_dataset, joined_name = join_columns(dataset, pair)
            joined, joined_prepared = _run(joined_dataset)
            found = any(
                COMPOSITE_SEPARATOR in ref.column
                for attribute in joined_prepared.mapping.attributes
                if any(ref.column == joined_name for ref in attribute.columns)
                for ref in attribute.columns
            )
            rows.append(_row(joined_dataset, JOINED, joined, composite_found=float(found)))
    save(pd.DataFrame(rows), "e9_robustness.csv")


def hide_column_names(dataset: Dataset) -> tuple[Dataset, Renames]:
    generator = np.random.default_rng(SEED)
    names = list(dataset.sources)
    sources = {names[0]: dataset.sources[names[0]]}
    renames: Renames = {}
    for name in names[1:]:
        frame = dataset.sources[name]
        order = generator.permutation(len(frame.columns))
        mapping = {str(column): HIDDEN_NAME.format(position + 1) for position, column in zip(order, frame.columns)}
        renames[name] = mapping
        sources[name] = frame.rename(columns=mapping)[
            sorted(mapping.values(), key=lambda label: int(label.split()[-1]))
        ]
    return replace(dataset, sources=sources), renames


def reduce_overlap(dataset: Dataset, share: float) -> Dataset | None:
    names = list(dataset.sources)
    if len(names) != 2:
        return None
    second = names[1]
    linked = sorted({key for pair in dataset.gold for source, key in pair if source == second})
    dropped = set(np.random.default_rng(SEED).choice(linked, int(len(linked) * (1 - share)), replace=False).tolist())
    frame = dataset.sources[second]
    sources = {names[0]: dataset.sources[names[0]], second: frame[~frame.index.astype(str).isin(dropped)]}
    gold = frozenset(
        pair for pair in dataset.gold if not any(source == second and key in dropped for source, key in pair)
    )
    return replace(dataset, sources=sources, gold=gold)


def joinable_columns(dataset: Dataset, mapping: SchemaMapping) -> tuple[str, str] | None:
    names = list(dataset.sources)
    if len(names) < 2:
        return None
    last = names[-1]
    shared = {
        attribute.column_in(last)
        for attribute in mapping.attributes
        if attribute.type.is_textual and len(attribute.sources) > 1 and attribute.column_in(last) is not None
    }
    columns = [str(column) for column in dataset.sources[last].columns if str(column) in shared]
    return (columns[0], columns[1]) if len(columns) >= 2 else None


def join_columns(dataset: Dataset, columns: tuple[str, str]) -> tuple[Dataset, str]:
    names = list(dataset.sources)
    source = names[-1]
    frame = dataset.sources[source]
    name = COMPOSITE_SEPARATOR.join(columns).replace(COMPOSITE_SEPARATOR, "_")
    joined = frame.drop(columns=list(columns)).assign(**{name: joined_values(frame, columns)})
    return replace(dataset, sources={**dataset.sources, source: joined}), name


def correspondences(mapping: SchemaMapping, renames: Renames | None = None) -> set[tuple[tuple[str, str], ...]]:
    originals = {
        source: {hidden: column for column, hidden in renamed.items()} for source, renamed in (renames or {}).items()
    }
    pairs = set()
    for attribute in mapping.attributes:
        columns = [(ref.source, originals.get(ref.source, {}).get(ref.column, ref.column)) for ref in attribute.columns]
        pairs.update(
            tuple(sorted((columns[i], columns[j]))) for i in range(len(columns)) for j in range(i + 1, len(columns))
        )
    return pairs


def _run(dataset: Dataset) -> tuple[MatchMetrics, PreparedData]:
    prepared = prepare(dataset)
    records = prepared.records
    result = Linker(linkage_settings()).run(records)
    return evaluate_positions(linked_pairs(dataset, result), gold_positions(dataset, records), records.size), prepared


def _row(dataset: Dataset, condition: str, metrics: MatchMetrics, **columns: float) -> dict[str, object]:
    return {
        "набір": dataset.name,
        "умова": condition,
        "записів": dataset.n_records,
        "пар дублікатів в еталоні": len(dataset.gold),
        "precision": metrics.precision,
        "recall": metrics.recall,
        "F1": metrics.f1,
        "відповідностей колонок знайдено": columns.get("columns_found", np.nan),
        "точність зіставлення колонок": columns.get("columns_precision", np.nan),
        "складене поле розпізнано": columns.get("composite_found", np.nan),
    }


if __name__ == "__main__":
    main()
