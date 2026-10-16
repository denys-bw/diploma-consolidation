import pytest

from benchmarks import NeedsDownload, load_dataset
from consolidator.io import Source
from consolidator.profiling import profile_frame
from consolidator.quality import Consistency, DataSnapshot, QualityWeights, RecordCompleteness, Uniqueness, assess
from consolidator.schema import identity_mapping
from consolidator.standardization import StandardizationSettings, standardize_frames


def load(name):
    try:
        return load_dataset(name)
    except NeedsDownload as error:
        pytest.skip(str(error))


def test_flights_are_split_into_sources():
    dataset = load("flights")
    dirty = dataset.sources["dirty"]
    sources = Source("flights", dirty.drop(columns=["tuple_id"]), "dirty.csv", "CSV").split_by("src")
    assert len(sources) == 38


def test_restaurants_reproduce_published_quality_index():
    dataset = load("restaurants")
    frames = dataset.sources
    types = {name: profile_frame(frame).types for name, frame in frames.items()}
    mapping = identity_mapping(frames, types)
    settings = StandardizationSettings(phone_format="999-999-9999")
    tables = standardize_frames(list(frames.values()), mapping.types, settings)
    records = sum(len(frame) for frame in frames.values())
    snapshot = DataSnapshot(
        frames=list(frames.values()),
        statuses=[table.status for table in tables],
        types=mapping.types,
        records=records,
        entities=records - len(dataset.gold),
    )
    scores = assess(snapshot, [RecordCompleteness(), Uniqueness(), Consistency()])
    assert scores["completeness"] == pytest.approx(0.9988, abs=5e-5)
    assert scores["uniqueness"] == pytest.approx(0.8704, abs=5e-5)
    assert scores["consistency"] == pytest.approx(0.3831, abs=5e-5)
    assert scores.index(QualityWeights.default()) == pytest.approx(0.8525, abs=5e-5)
