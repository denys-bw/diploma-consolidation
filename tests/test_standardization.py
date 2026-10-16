import pandas as pd
import pytest

from consolidator.profiling import ColumnType, numeric_values
from consolidator.standardization import (
    CellStatus,
    FormatPolicy,
    StandardizationSettings,
    standardize_frame,
    standardize_frames,
    standardizer_for,
)

DOMINANT = StandardizationSettings(policy=FormatPolicy.DOMINANT)


def standardize(column_type, values, settings=None):
    return standardizer_for(column_type, settings).fit_transform(pd.Series(values, dtype=object))


def test_numbers_lose_units_and_spurious_percent_signs():
    result = standardize(ColumnType.MEASURE, ["12.0 oz", "16.0 OZ.", "12.0 ounce", "12.0 oz. Alumi-Tek", "N/A"])
    assert result.values.tolist() == ["12", "16", "12", "12", ""]
    abv = standardize(ColumnType.NUMBER, ["0.05", "0.066", "0.052000000000000005%", "0.07"])
    assert abv.values.tolist() == ["0.05", "0.066", "0.052", "0.07"]


def test_true_percentages_are_rescaled():
    result = standardize(ColumnType.NUMBER, ["0.05", "0.06", "0.07", "5%", "6%"])
    assert result.values.tolist() == ["0.05", "0.06", "0.07", "0.05", "0.06"]


def test_units_of_the_same_dimension_are_converted():
    result = standardize(ColumnType.MEASURE, ["1 l", "2 l", "500 ml", "1.5 l"])
    assert result.values.tolist() == ["1", "2", "0.5", "1.5"]


def test_dates_become_iso_or_dominant_layout():
    values = ["10/01/15", "04/30/15", "2015-05-01"]
    assert standardize(ColumnType.DATE, values).values.tolist() == ["2015-10-01", "2015-04-30", "2015-05-01"]
    assert standardize(ColumnType.DATE, values, DOMINANT).values.tolist() == ["10/01/15", "04/30/15", "05/01/15"]


def test_times_follow_dominant_layout_of_the_column():
    values = ["7:10 a.m.", "7:58 p.m.", "5:56aDec 1", "12:00noon", "", "Not Available"]
    result = standardize(ColumnType.TIME, values, DOMINANT)
    assert result.values.tolist() == ["7:10 a.m.", "7:58 p.m.", "5:56 a.m.", "12:00 p.m.", "", ""]
    standard = standardize(ColumnType.TIME, values)
    assert standard.values.tolist()[:4] == ["07:10", "19:58", "05:56", "12:00"]


def test_phones_become_e164_for_the_configured_region():
    values = ["067 123 45 67", "(044) 235-12-34", "+380501234567"]
    assert standardize(ColumnType.PHONE, values).values.tolist() == ["+380671234567", "+380442351234", "+380501234567"]
    us = standardize(ColumnType.PHONE, ["310/246-1501", "3865286"], StandardizationSettings(phone_region="US"))
    assert us.values.tolist() == ["+13102461501", "3865286"]
    assert us.status.tolist() == [CellStatus.CHANGED, CellStatus.INVALID]


def test_phones_keep_dominant_national_style():
    values = ["310/246-1501", "818/762-1221", "310-472-1211"]
    settings = StandardizationSettings(policy=FormatPolicy.DOMINANT, phone_region="US")
    assert standardize(ColumnType.PHONE, values, settings).values.tolist() == [
        "310/246-1501",
        "818/762-1221",
        "310/472-1211",
    ]


def test_phones_follow_an_explicit_reference_format():
    values = ["310/246-1501", "818/762-1221", "310-472-1211"]
    settings = StandardizationSettings(phone_region="US", phone_format="999-999-9999")
    result = standardize(ColumnType.PHONE, values, settings)
    assert result.values.tolist() == ["310-246-1501", "818-762-1221", "310-472-1211"]
    assert result.status.tolist() == [CellStatus.CHANGED, CellStatus.CHANGED, CellStatus.UNCHANGED]


def test_ukrainian_phones_fit_national_and_international_templates():
    values = ["067 123 45 67", "+380 50 123 45 67", "(044) 235-12-34"]
    national = standardize(ColumnType.PHONE, values, StandardizationSettings(phone_format="(999) 999-99-99"))
    assert national.values.tolist() == ["(067) 123-45-67", "(050) 123-45-67", "(044) 235-12-34"]
    international = standardize(ColumnType.PHONE, values, StandardizationSettings(phone_format="+999 99 999 99 99"))
    assert international.values.tolist() == ["+380 67 123 45 67", "+380 50 123 45 67", "+380 44 235 12 34"]


def test_local_numbers_stay_consistent_in_dominant_style():
    settings = StandardizationSettings(policy=FormatPolicy.DOMINANT, phone_region="US")
    result = standardize(ColumnType.PHONE, ["2859902", "285-9902", "5346361"], settings)
    assert result.values.tolist() == ["2859902", "2859902", "5346361"]
    assert result.status.tolist() == [CellStatus.UNCHANGED, CellStatus.CHANGED, CellStatus.UNCHANGED]


def test_datetime_keeps_the_dominant_clock_separator():
    values = ["2021-03-11T10:15", "2020-02-29T23:59:30", "2021-03-12 08:00"]
    assert standardize(ColumnType.DATETIME, values, DOMINANT).values.tolist() == [
        "2021-03-11T10:15",
        "2020-02-29T23:59:30",
        "2021-03-12T08:00",
    ]


@pytest.mark.parametrize(
    ("column_type", "values", "settings"),
    [
        (ColumnType.TIME, ["7:10 a.m.", "11:45 p.m.", "12:05 a.m."], DOMINANT),
        (ColumnType.TIME, ["07:10", "23:45"], None),
        (ColumnType.NUMBER, ["0.05", "0.055", "12"], None),
        (ColumnType.YEAR, ["1999", "2004"], None),
        (ColumnType.DATE, ["03.11.2021", "29.02.2020"], DOMINANT),
        (ColumnType.DATETIME, ["2021-03-11T10:15", "2020-02-29T23:59:30"], None),
    ],
)
def test_rendering_numbers_restores_the_column_format(column_type, values, settings):
    standardizer = standardizer_for(column_type, settings).fit(pd.Series(values, dtype=object))
    numbers = numeric_values(pd.Series(values, dtype=object), column_type)
    assert [standardizer.render(number) for number in numbers] == values


def test_rendering_keeps_column_precision():
    standardizer = standardizer_for(ColumnType.NUMBER).fit(pd.Series(["0.05", "0.055"], dtype=object))
    assert standardizer.render(0.05678) == "0.057"
    assert standardizer_for(ColumnType.TEXT).render(1.0) is None
    assert standardizer_for(ColumnType.TIME).render(float("nan")) is None


def test_categories_take_the_most_frequent_spelling():
    values = ["Chicago Public Schools", "CHICAGO PUBLIC SCHOOLS", "Chicago Public Schools", "YMCA"]
    assert standardize(ColumnType.CATEGORY, values).values.tolist() == ["Chicago Public Schools"] * 3 + ["YMCA"]


def test_booleans():
    assert standardize(ColumnType.BOOLEAN, ["Yes", "no", "так"]).values.tolist() == ["true", "false", "true"]
    assert standardize(ColumnType.BOOLEAN, ["Yes", "No", "yes"], DOMINANT).values.tolist() == ["Yes", "No", "Yes"]


def test_text_cleaning_repairs_encoding_and_spaces():
    result = standardize(ColumnType.TEXT, [" Chicago Commons\nAssociation", "SjÃ¶strÃ¶m", "ok"])
    assert result.values.tolist() == ["Chicago Commons Association", "Sjöström", "ok"]
    assert result.status.tolist() == [CellStatus.CHANGED, CellStatus.CHANGED, CellStatus.UNCHANGED]


def test_frame_summary_counts_statuses():
    frame = pd.DataFrame({"phone": ["067 123 45 67", "", "abc 12"], "name": ["a", "b", "c"]})
    table = standardize_frame(frame, {"phone": ColumnType.PHONE, "name": ColumnType.TEXT})
    summary = table.summary().set_index("column")
    assert summary.loc["phone"].to_dict() == {"missing": 1, "unchanged": 0, "changed": 1, "invalid": 1, "ambiguous": 0}
    assert summary.loc["name", "unchanged"] == 3


def test_several_sources_share_column_parameters():
    first = pd.DataFrame({"date": ["04/30/15"]})
    second = pd.DataFrame({"date": ["05/01/15"]})
    tables = standardize_frames([first, second], {"date": ColumnType.DATE})
    assert tables[1].frame.loc[0, "date"] == "2015-05-01"
    assert tables[0].parameters["date"]["dayfirst"] == "False"


@pytest.mark.parametrize("column_type", list(ColumnType))
def test_every_type_has_a_standardizer(column_type):
    result = standardize(column_type, ["value", ""])
    assert len(result.values) == 2


def test_dates_without_evidence_of_order_stay_as_they_are():
    result = standardize(ColumnType.DATE, ["01/02/2003", "03/04/2005"])
    assert result.values.tolist() == ["01/02/2003", "03/04/2005"]
    assert result.status.tolist() == [CellStatus.AMBIGUOUS, CellStatus.AMBIGUOUS]
    resolved = standardize(ColumnType.DATE, ["01/02/2003", "25/04/2005"])
    assert resolved.values.tolist() == ["2003-02-01", "2005-04-25"]


def test_each_source_keeps_its_own_date_order():
    american = pd.Series(["12/31/2020", "01/02/2021"])
    european = pd.Series(["31/12/2020", "01/02/2021"])
    standardizer = standardizer_for(ColumnType.DATE).fit({"us": american, "eu": european})
    assert standardizer.transform(american, "us").values.tolist() == ["2020-12-31", "2021-01-02"]
    assert standardizer.transform(european, "eu").values.tolist() == ["2020-12-31", "2021-02-01"]


def test_phone_region_is_taken_from_international_numbers_of_any_source():
    local = pd.Series(["(312) 555-0100", "773 555 0101"])
    international = pd.Series(["+1 312 555 0102"])
    standardizer = standardizer_for(ColumnType.PHONE).fit({"local": local, "global": international})
    assert standardizer.source_regions["local"] == "US"
    assert standardizer.transform(local, "local").values.tolist() == ["+13125550100", "+17735550101"]
    unknown = standardizer_for(ColumnType.PHONE).fit({"only": local})
    assert unknown.transform(local, "only").values.tolist() == ["3125550100", "7735550101"]
