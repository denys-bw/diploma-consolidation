import numpy as np
import pandas as pd
import pytest

from consolidator.profiling import (
    OUTLIER_DETECTORS,
    ColumnType,
    IsolationForestDetector,
    TukeyFences,
    detect_outliers,
    group_deviations,
    infer_column_type,
    numeric_values,
    profile_frame,
    quantitative_frame,
    summarize,
)


def series(values, repeat=1):
    return pd.Series(list(values) * repeat, dtype=object)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["310/246-1501", "818-762-1221", "213/555-1234"], ColumnType.PHONE),
        (["386-5286", "285-9902", "534-6361", "265-3830"], ColumnType.PHONE),
        (["+380 44 123 4567", "+380 67 765 4321"], ColumnType.PHONE),
        (["a@b.com", "info@example.org", "x.y@z.ua"], ColumnType.EMAIL),
        (["www.cps.edu", "http://www.chicagosinai.org", "example.com/about"], ColumnType.URL),
        (["10/01/15", "04/30/15", "08/01/17"], ColumnType.DATE),
        (["2011-12-01 07:10", "2011-12-02 08:15"], ColumnType.DATETIME),
        (["7:10 a.m.", "5:56aDec 1", "12:00noon"], ColumnType.TIME),
        (["1999", "2001", "1987"], ColumnType.YEAR),
        (["12.0 oz", "16.0 oz", "12 OZ."], ColumnType.MEASURE),
        (["1,5 кг", "2 кг", "750 г"], ColumnType.MEASURE),
        (["23.80", "2195", "16"], ColumnType.NUMBER),
        (["60640", "60623", "60640", "60619"], ColumnType.CODE),
        (["Yes", "No", "Yes"], ColumnType.BOOLEAN),
        (["a012320c", "ae0cec0e", "a012320c"], ColumnType.CODE),
    ],
)
def test_column_types_are_inferred_from_values(values, expected):
    assert infer_column_type(series(values, 10)).type is expected


def test_identifier_needs_unique_values():
    values = series([f"id{index:04d}" for index in range(200)])
    assert infer_column_type(values).type is ColumnType.IDENTIFIER


def test_categories_have_few_distinct_values():
    categories = series(["Child Care", "Head Start", "Pre-K"] * 30)
    names = series([f"Restaurant number {index}" for index in range(100)])
    descriptions = series([" ".join(["word"] * 20 + [str(index)]) for index in range(100)])
    assert infer_column_type(categories).type is ColumnType.CATEGORY
    assert infer_column_type(names).type is ColumnType.TEXT
    assert infer_column_type(descriptions).type is ColumnType.TEXT


def test_digit_codes_and_identifiers():
    assert infer_column_type(series(["60640", "60623", "60640", "60619"], 5)).type is ColumnType.CODE
    assert infer_column_type(series(["007", "012", "450"])).type is ColumnType.IDENTIFIER
    assert infer_column_type(series([str(value) for value in range(100, 160)])).type is ColumnType.IDENTIFIER
    assert infer_column_type(series(["45", "60", "70", "8", "120"])).type is ColumnType.NUMBER


def test_years_lie_in_the_gregorian_calendar_up_to_now():
    assert infer_column_type(series(["1999", "2001", "1987"])).type is ColumnType.YEAR
    assert infer_column_type(series(["1200", "1300", "1400"])).type is not ColumnType.YEAR
    assert infer_column_type(series(["8000", "9000", "8500"])).type is not ColumnType.YEAR


def test_unknown_unit_words_must_form_a_small_vocabulary():
    measured = series(["12 apples", "3 apples", "5 apples", "7 pears"] * 10)
    addresses = series(
        [f"{number} {street}" for number, street in zip(range(40), [f"Street{index}" for index in range(40)])]
    )
    assert infer_column_type(measured).type is ColumnType.MEASURE
    assert infer_column_type(addresses).type is not ColumnType.MEASURE


def test_column_name_does_not_change_the_type():
    frame = pd.DataFrame({"Email Address": ["10/01/15", "04/30/15"] * 10, "Website": ["Paula Cottone", "Dan V"] * 10})
    renamed = frame.set_axis(["a", "b"], axis=1)
    assert list(profile_frame(frame).types.values()) == list(profile_frame(renamed).types.values())
    assert profile_frame(frame).types["Email Address"] is ColumnType.DATE
    assert profile_frame(frame).types["Website"] is not ColumnType.URL


def test_empty_column():
    assert infer_column_type(series(["", "N/A", " "])).type is ColumnType.EMPTY


def test_profile_counts_missing_and_disguised_missing():
    frame = pd.DataFrame({"ibu": ["N/A", "45", "", "60", "70"], "name": ["a", "b", "c", "d", "e"]})
    profile = profile_frame(frame, "beers")
    ibu = profile.column("ibu")
    assert ibu.missing == 2
    assert ibu.disguised_missing == 1
    assert ibu.completeness == pytest.approx(0.6)
    assert ibu.type is ColumnType.NUMBER
    assert ibu.numeric.median == 60
    assert profile.completeness == pytest.approx(0.8)


def test_patterns_show_inconsistent_formats():
    frame = pd.DataFrame({"phone": ["310/246-1501", "310/472-1211", "310-246-1501"]})
    column = profile_frame(frame).column("phone")
    assert column.top_patterns[0][0] == "999/999-9999"
    assert column.dominant_pattern_share == pytest.approx(2 / 3)


def test_tukey_fences():
    values = numeric_values(pd.Series(["10", "11", "12", "13", "14", "100"]), ColumnType.NUMBER)
    summary = summarize(values)
    assert summary.q1 == pytest.approx(11.25)
    assert TukeyFences().detect(values).tolist() == [False] * 5 + [True]


def test_temporal_values_become_numbers():
    minutes = numeric_values(pd.Series(["7:10 a.m.", "7:10 p.m.", "bad"]), ColumnType.TIME)
    assert minutes[:2].tolist() == [430.0, 1150.0]
    assert pd.isna(minutes[2])


def test_profile_counts_outliers_by_each_detector():
    frame = pd.DataFrame({"abv": ["0.05", "0.055", "0.06", "0.05", "0.058", "0.052", "0.057", "0.054", "0.9"]})
    column = profile_frame(frame).column("abv")
    assert column.outliers["iqr"] == 1
    assert column.outliers["isolation_forest"] >= 1
    assert column.numeric.upper_fence < 0.9
    assert profile_frame(pd.DataFrame({"name": ["a", "b"]})).column("name").outliers == {}


def test_quantitative_frame_keeps_only_measurable_columns():
    frame = pd.DataFrame({"dep": ["7:10 a.m.", ""], "name": ["a", "b"], "year": ["1999", "2001"]})
    types = {"dep": ColumnType.TIME, "name": ColumnType.TEXT, "year": ColumnType.YEAR}
    numbers = quantitative_frame(frame, types)
    assert numbers.columns.tolist() == ["dep", "year"]
    assert numbers.loc[0, "dep"] == 430.0
    assert np.isnan(numbers.loc[1, "dep"])


def test_global_detectors_find_extreme_values():
    values = pd.DataFrame({"price": [10.0, 11.0, 12.0, 11.5, 10.5, 11.2, 10.8, 12.1, 11.7, 95.0]})
    for detector in OUTLIER_DETECTORS:
        flags = detect_outliers(values, detector)
        assert flags["price"].iloc[-1]
        assert flags["price"].sum() <= 2


def test_group_deviations_reveal_conflicts_inside_entities():
    minutes = pd.DataFrame({"departure": [430.0, 430.0, 430.0, 436.0, 1150.0, 1150.0, 1150.0, 1170.0]})
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    deviations = group_deviations(minutes, groups)
    assert deviations["departure"].tolist() == [0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 20.0]
    assert not detect_outliers(minutes, TukeyFences())["departure"].any()
    contextual = detect_outliers(minutes, TukeyFences(), groups)
    assert contextual["departure"].tolist() == [False, False, False, True, False, False, False, True]


def test_isolation_forest_needs_varied_values():
    single = pd.DataFrame({"x": [7.0]})
    constant = pd.DataFrame({"x": [3.0] * 20})
    assert not detect_outliers(single, IsolationForestDetector())["x"].any()
    assert not detect_outliers(constant, IsolationForestDetector())["x"].any()
