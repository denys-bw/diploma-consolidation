import numpy as np
import pandas as pd
import pytest

from consolidator.imputation import AdaptiveImputer, KnnImputer, MeanImputer, cross_validated_errors, impute
from consolidator.profiling import ColumnType
from consolidator.standardization import FormatPolicy, StandardizationSettings

TYPES = {
    "scheduled": ColumnType.TIME,
    "actual": ColumnType.TIME,
    "gate": ColumnType.CODE,
}
DOMINANT = StandardizationSettings(policy=FormatPolicy.DOMINANT)


def flights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scheduled": ["7:10 a.m.", "7:10 a.m.", "7:10 a.m.", "6:30 p.m.", "6:30 p.m.", "6:30 p.m.", "9:40 a.m."],
            "actual": ["7:16 a.m.", "7:16 a.m.", "", "6:47 p.m.", "6:47 p.m.", "", ""],
            "gate": ["B12", "B12", "", "C3", "C3", "C3", ""],
        }
    )


def test_knn_takes_values_of_similar_records():
    frame = flights()
    result = impute(frame, TYPES, KnnImputer(neighbours=2), DOMINANT)
    assert result.frame.loc[2, "actual"] == "7:16 a.m."
    assert result.frame.loc[5, "actual"] == "6:47 p.m."
    assert result.frame.loc[2, "gate"] == ""
    assert result.imputed["actual"].tolist() == [False, False, True, False, False, True, True]
    assert result.count == 3
    assert result.counts()["actual"] == 3


def test_mean_ignores_neighbours():
    result = impute(flights(), TYPES, MeanImputer(), DOMINANT)
    assert result.frame.loc[2, "actual"] == result.frame.loc[5, "actual"] == "1:02 p.m."


def test_only_applicable_cells_are_filled():
    frame = flights()
    applicable = pd.DataFrame(True, index=frame.index, columns=frame.columns)
    applicable.loc[6, "actual"] = False
    result = impute(frame, TYPES, KnnImputer(neighbours=2), DOMINANT, applicable)
    assert result.frame.loc[6, "actual"] == ""
    assert not result.imputed.loc[6, "actual"]


def test_knn_needs_other_measurements_and_enough_observations():
    lonely = pd.DataFrame({"actual": ["7:16 a.m.", "", "6:47 p.m."]})
    assert impute(lonely, {"actual": ColumnType.TIME}, KnnImputer(neighbours=1)).count == 0
    sparse = pd.DataFrame({"price": ["10", "", ""], "year": ["1999", "2001", "2003"]})
    types = {"price": ColumnType.NUMBER, "year": ColumnType.YEAR}
    assert impute(sparse, types, KnnImputer(neighbours=2)).count == 0
    assert impute(sparse, types, MeanImputer()).frame["price"].tolist() == ["10", "10", "10"]


@pytest.mark.parametrize("imputer", [MeanImputer(), KnnImputer(5), AdaptiveImputer()])
def test_frames_without_measurements_are_returned_unchanged(imputer):
    frame = pd.DataFrame({"name": ["a", ""]})
    result = impute(frame, {"name": ColumnType.TEXT}, imputer)
    assert result.frame.equals(frame)
    assert result.count == 0


def correlated(rows: int, seed: int = 0) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    base = generator.uniform(0, 100, rows)
    frame = pd.DataFrame({"a": base, "b": base * 2 + generator.normal(0, 1, rows), "c": generator.uniform(0, 1, rows)})
    return frame.round(3).astype(str)


def test_adaptive_imputer_prefers_neighbours_when_attributes_are_related():
    frame = correlated(200)
    numbers = frame.astype(float)
    numbers.iloc[::7, 0] = np.nan
    chosen = AdaptiveImputer().resolve(numbers)
    assert isinstance(chosen, KnnImputer)
    errors = cross_validated_errors(numbers, AdaptiveImputer.candidates(numbers))
    assert errors[chosen.label] == min(errors.values())
    assert errors[chosen.label] < errors[MeanImputer().label]


def test_adaptive_imputer_falls_back_to_mean_for_a_single_measurement():
    numbers = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0]})
    assert isinstance(AdaptiveImputer().resolve(numbers), MeanImputer)


def test_imputation_reports_the_chosen_method():
    frame = correlated(120)
    frame.iloc[::5, 1] = ""
    types = dict.fromkeys(frame.columns, ColumnType.NUMBER)
    result = impute(frame, types, AdaptiveImputer())
    assert result.imputer is not None and result.imputer.key in ("knn", "mean")
    assert result.mark.startswith("imputed:")
    assert result.count == int((frame["b"] == "").sum())
