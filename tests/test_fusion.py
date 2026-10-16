import numpy as np
import pandas as pd
import pytest

from consolidator.fusion import NO_VALUE, Claims, Fuser, Longest, MajorityVote, SourcePriority, discover_truth
from consolidator.linkage import RecordSet
from consolidator.profiling import ColumnType

TIMES = {"flight": ColumnType.CODE, "departure": ColumnType.TIME}


def departures() -> RecordSet:
    reported = {
        "airline": ["7:10 a.m.", "8:00 a.m.", "9:15 a.m.", "6:30 p.m.", "11:05 a.m.", "1:40 p.m."],
        "tracker": ["7:10 a.m.", "8:00 a.m.", "9:15 a.m.", "6:30 p.m.", "", ""],
        "mirror_a": ["7:16 a.m.", "8:05 a.m.", "9:20 a.m.", "6:47 p.m.", "11:00 a.m.", "1:55 p.m."],
        "mirror_b": ["7:18 a.m.", "8:09 a.m.", "9:25 a.m.", "6:50 p.m.", "11:00 a.m.", "1:55 p.m."],
    }
    flights = [
        "AA-3859-IAH-ORD",
        "AA-1733-ORD-PHX",
        "AA-1640-MIA-MCO",
        "AA-518-MIA-JFK",
        "UA-913-SFO-ORD",
        "UA-207-DEN-ORD",
    ]
    frames = {
        source: pd.DataFrame({"flight": flights, "departure": values}, index=[f"{source}-{n}" for n in range(6)])
        for source, values in reported.items()
    }
    return RecordSet.from_frames(frames, TIMES)


def flight_labels(records: RecordSet) -> np.ndarray:
    return pd.factorize(records.values("flight"))[0]


def test_truth_discovery_trusts_accurate_sources():
    claims = Claims(
        items=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        values=np.array([0, 0, 1, 0, 0, 1, 0, 1, 1]),
        sources=np.array([0, 1, 2, 0, 1, 2, 0, 2, 3]),
        n_sources=4,
    )
    estimate = discover_truth(claims)
    assert estimate.converged
    assert estimate.accuracy[0] > estimate.accuracy[2]
    assert np.all((estimate.accuracy > 0) & (estimate.accuracy < 1))
    totals = pd.Series(estimate.probabilities).groupby([claims.items, claims.values]).first()
    assert totals.groupby(level=0).sum().to_numpy() == pytest.approx(np.ones(3))


def test_claims_must_have_equal_lengths():
    with pytest.raises(ValueError):
        Claims(np.zeros(2, dtype=int), np.zeros(1, dtype=int), np.zeros(2, dtype=int), 1)


def test_source_accuracy_starts_from_agreement_with_the_majority():
    claims = Claims(
        items=np.array([0, 0, 0, 1, 1, 1]),
        values=np.array([0, 0, 1, 0, 0, 1]),
        sources=np.array([0, 1, 2, 0, 1, 2]),
        n_sources=3,
    )
    estimate = discover_truth(claims)
    assert estimate.accuracy[0] == pytest.approx(estimate.accuracy[1])
    assert estimate.accuracy[2] < estimate.accuracy[0]
    empty = discover_truth(Claims(np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), 2))
    assert empty.accuracy.tolist() == [0.5, 0.5]


def test_weighted_vote_beats_majority_of_agreeing_mirrors():
    records = departures()
    labels = flight_labels(records)
    weighted = Fuser().fuse(records, labels)
    majority = Fuser({"departure": MajorityVote()}).fuse(records, labels)
    assert weighted.entities["departure"].tolist() == [
        "7:10 a.m.",
        "8:00 a.m.",
        "9:15 a.m.",
        "6:30 p.m.",
        "11:05 a.m.",
        "1:40 p.m.",
    ]
    assert majority.entities["departure"].tolist()[4:] == ["11:00 a.m.", "1:55 p.m."]
    accuracy = weighted.source_accuracy
    assert accuracy["airline"] > accuracy["mirror_a"]
    assert accuracy["tracker"] > accuracy["mirror_b"]


def test_fusion_lineage_and_cell_level_views():
    records = departures()
    result = Fuser().fuse(records, flight_labels(records))
    assert result.n_entities == 6
    assert result.merged_records == records.size - 6
    assert result.entities.loc[0, "sources"] == "airline, tracker, mirror_a, mirror_b"
    assert result.entities.loc[0, "records"] == 4
    assert result.provenance.loc[4, "departure"] == "airline"
    frame = result.record_frame()
    assert frame.loc[frame["key"] == "tracker-4", "departure"].item() == "11:05 a.m."
    conflicts = result.conflicting_cells()
    assert conflicts["departure"].sum() == 12
    assert not conflicts["flight"].any()
    filled = result.filled_cells()
    assert filled["departure"].sum() == 2
    disputed = result.conflicts[result.conflicts["attribute"] == "departure"]
    assert len(disputed) == 6


def test_median_keeps_a_reported_value():
    frames = {
        source: pd.DataFrame({"price": [value]}, index=["1"])
        for source, value in {"a": "10", "b": "40", "c": "12", "d": "13"}.items()
    }
    records = RecordSet.from_frames(frames, {"price": ColumnType.NUMBER})
    result = Fuser().fuse(records, np.zeros(4, dtype=int))
    assert result.entities.loc[0, "price"] == "12"
    assert result.provenance.loc[0, "price"] == "c"


def test_other_resolutions_and_missing_values():
    frames = {
        "first": pd.DataFrame({"title": ["Kyiv", ""]}, index=["1", "2"]),
        "second": pd.DataFrame({"title": ["Kyiv city centre", ""]}, index=["1", "2"]),
    }
    records = RecordSet.from_frames(frames, {"title": ColumnType.TEXT})
    labels = np.array([0, 1, 0, 1])
    longest = Fuser({"title": Longest()}).fuse(records, labels)
    first = Fuser({"title": SourcePriority()}, source_priority=["second", "first"]).fuse(records, labels)
    assert longest.entities["title"].tolist() == ["Kyiv city centre", ""]
    assert first.entities["title"].tolist() == ["Kyiv city centre", ""]
    assert longest.chosen["title"][1] == NO_VALUE
    assert longest.provenance.loc[1, "title"] == ""


def test_fusion_checks_label_count():
    with pytest.raises(ValueError):
        Fuser().fuse(departures(), np.zeros(3, dtype=int))
