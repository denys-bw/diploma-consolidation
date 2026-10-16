from dataclasses import replace

import pandas as pd
import pytest

from benchmarks import REGISTRY, NeedsDownload, Task, check, load_chicago_sources, load_dataset
from benchmarks.multisource import ENTITY, KEY, SOURCE, sample_entities, split_sources


def load(name):
    try:
        return load_chicago_sources() if name == "chicago_sources" else load_dataset(name)
    except NeedsDownload as error:
        pytest.skip(f"потрібно скачати: {error}")


@pytest.mark.parametrize("name", list(REGISTRY))
def test_dataset_matches_official_counts(name):
    assert check(load(name)) == []


def test_changed_files_are_detected_by_checksums():
    dataset = load("restaurants")
    tampered = replace(dataset, checksums=dict.fromkeys(dataset.checksums, "0" * 64))
    problems = check(tampered)
    assert len(problems) == len(dataset.checksums) == 2
    assert all("SHA-256" in problem for problem in problems)


def test_unknown_dataset_is_reported():
    with pytest.raises(KeyError, match="Невідомий набір"):
        load_dataset("missing")


def test_chicago_sources_reproduce_original_values():
    original = load("chicago")
    split = load_chicago_sources()
    assert split.task is Task.DEDUP
    assert check(split) == []
    merged = {key: row for frame in split.sources.values() for key, row in frame.iterrows()}
    table = original.sources["chicago"].drop(columns=["Source"])
    for key, row in table.iterrows():
        values = merged[key]
        for column, value in row.items():
            assert values.get(column, "") == value


def test_entity_sample_keeps_the_requested_share_with_copies():
    entities = pd.Series(["a", "a", "b", "c", "c", "c", "d", "e", "f", "g"])
    chosen = sample_entities(entities, 4, 0.5, seed=0)
    repeated = {"a", "c"}
    assert len(chosen) == 4 and len(set(chosen)) == 4
    assert len(set(chosen) & repeated) == 2
    assert list(sample_entities(entities, 4, 0.5, seed=0)) == list(chosen)
    with pytest.raises(ValueError):
        sample_entities(entities, 4, 1.0, seed=0)


def test_sources_and_gold_are_built_from_entity_labels():
    table = pd.DataFrame(
        {
            SOURCE: ["s1", "s1", "s2", "s2", "s3"],
            KEY: ["0", "1", "0", "1", "0"],
            ENTITY: ["x", "y", "x", "z", "x"],
            "name": ["ann", "bob", "anne", "cid", "ann"],
        }
    )
    sources, gold = split_sources(table)
    assert list(sources) == ["s1", "s2", "s3"]
    assert sources["s2"].loc["0", "name"] == "anne"
    assert ENTITY not in sources["s1"].columns
    assert gold == {(("s1", "0"), ("s2", "0")), (("s1", "0"), ("s3", "0")), (("s2", "0"), ("s3", "0"))}
