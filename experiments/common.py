from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks import Dataset, NeedsDownload, Task, load_chicago_sources, load_dataset
from benchmarks.dataset import ROOT
from consolidator.io import Source
from consolidator.linkage import LinkageResult, LinkageSettings, PairScope, RecordSet, cluster_pairs
from consolidator.pipeline import Pipeline, PipelineSettings, PreparedData

RESULTS = ROOT / "results"
SEED = 0
CSV_ENCODING = "utf-8-sig"
CO_TRAINING = "Фелегі–Зунтер і логістична регресія (спільне навчання) + групування"
MAIN_DATASETS = [
    "restaurants",
    "dblp_acm",
    "abt_buy",
    "chicago_sources",
    "settlements",
    "musicbrainz",
    "nc_voters_sample",
]


def load(name: str) -> Dataset:
    return load_chicago_sources() if name == "chicago_sources" else load_dataset(name)


def available(names: list[str]) -> Iterator[Dataset]:
    for name in names:
        try:
            yield load(name)
        except NeedsDownload as error:
            print(f"[пропущено] {name}: {error}")


def sources_of(dataset: Dataset) -> list[Source]:
    if dataset.task is Task.CLEAN:
        return table_sources(dataset)
    return [Source(name, frame, dataset.origin, dataset.file_format) for name, frame in dataset.sources.items()]


def table_sources(dataset: Dataset) -> list[Source]:
    layout = dataset.extra.get("layout", {})
    dirty = dataset.sources["dirty"]
    frame = dirty.drop(columns=[layout["row_column"]]) if "row_column" in layout else dirty
    table = Source(dataset.name, frame, dataset.origin, dataset.file_format)
    return table.split_by(layout["source_column"]) if "source_column" in layout else [table]


def linkage_settings() -> LinkageSettings:
    return LinkageSettings()


def pipeline_settings(**changes: object) -> PipelineSettings:
    return replace(PipelineSettings(), **changes)


def prepare(dataset: Dataset, settings: PipelineSettings | None = None) -> PreparedData:
    return Pipeline(settings or pipeline_settings()).prepare(sources_of(dataset))


def scope_mask(dataset: Dataset, records: RecordSet, pairs: np.ndarray) -> np.ndarray:
    if dataset.scope is PairScope.ALL:
        return np.ones(len(pairs), dtype=bool)
    codes = records.source_codes
    return codes[pairs[:, 0]] != codes[pairs[:, 1]]


def in_scope(dataset: Dataset, records: RecordSet, pairs: np.ndarray) -> np.ndarray:
    return pairs[scope_mask(dataset, records, pairs)]


def linked_pairs(dataset: Dataset, result: LinkageResult) -> np.ndarray:
    return cluster_pairs(result.labels, result.records.source_codes, dataset.scope)


def gold_positions(dataset: Dataset, records: RecordSet) -> np.ndarray:
    return records.pairs_from_ids(dataset.gold)


def save(table: pd.DataFrame, name: str, decimals: int = 4) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    rounded = table.round(decimals)
    rounded.to_csv(path, index=False, encoding=CSV_ENCODING)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_rows", 200):
        print(f"\n{path.relative_to(ROOT).as_posix()}")
        print(rounded.to_string(index=False))
    return path
