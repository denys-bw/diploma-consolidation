from dataclasses import replace

import pytest

from benchmarks import NeedsDownload, load_chicago_sources, load_dataset
from consolidator.linkage import RecordSet
from consolidator.profiling import profile_frame
from consolidator.schema import identity_mapping
from consolidator.standardization import standardize_frames


def prepare(dataset):
    frames = dataset.sources
    types = {source: profile_frame(frame).types for source, frame in frames.items()}
    mapping = identity_mapping(frames, types)
    aligned = mapping.apply(frames)
    tables = standardize_frames(list(aligned.values()), mapping.types, names=list(aligned))
    return RecordSet.from_frames({source: table.frame for source, table in zip(aligned, tables)}, mapping.types)


def load(name):
    try:
        return load_chicago_sources() if name == "chicago_sources" else load_dataset(name)
    except NeedsDownload as error:
        pytest.skip(str(error))


def test_chicago_from_four_formats_gives_the_same_records_as_one_csv():
    original = load("chicago")
    split = load("chicago_sources")
    without_source_column = replace(original, sources={"chicago": original.sources["chicago"].drop(columns=["Source"])})
    single = prepare(without_source_column)
    several = prepare(split)
    single_values = {
        key: tuple(row)
        for key, row in zip(single.keys, single.frame[sorted(single.attributes)].itertuples(index=False))
    }
    several_values = {
        key: tuple(row)
        for key, row in zip(
            several.keys,
            several.frame.reindex(columns=sorted(single.attributes), fill_value="").itertuples(index=False),
        )
    }
    assert single_values == several_values
