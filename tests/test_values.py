from datetime import date, time

import pandas as pd
import pytest

from consolidator.values.dates import format_date, format_time, infer_dayfirst, parse_date, parse_time, parse_year
from consolidator.values.missing import is_missing, missing_mask
from consolidator.values.numbers import format_number, infer_decimal_comma, parse_number
from consolidator.values.patterns import value_shape
from consolidator.values.phones import has_phone_formatting, infer_region, looks_like_phone, parse_phone
from consolidator.values.text import clean_text, comparison_key, repair_mojibake
from consolidator.values.web import looks_like_email, looks_like_url, normalize_email, normalize_url


@pytest.mark.parametrize("value", ["", "  ", "N/A", "n/a", "NULL", "None", "-", "—", "Not Available", "немає", "н/д"])
def test_missing_markers(value):
    assert is_missing(value)


@pytest.mark.parametrize("value", ["0", "NA", "Ні", "no"])
def test_meaningful_values_are_not_missing(value):
    assert not is_missing(value)


def test_missing_mask_handles_nan_and_text():
    mask = missing_mask(pd.Series(["a", None, "N/A", " x "]))
    assert mask.tolist() == [False, True, True, False]


def test_value_shapes_reveal_formats():
    assert value_shape("310/246-1501") == "999/999-9999"
    assert value_shape("Portland OR") == "Aa A"
    assert value_shape("12.0 oz") == value_shape("16.0 oz") != value_shape("12 OZ.")


@pytest.mark.parametrize(
    ("text", "value", "unit"),
    [
        ("12.0 oz", 12.0, "oz"),
        ("12.0 OZ.", 12.0, "oz"),
        ("12.0 ounce", 12.0, "oz"),
        ("0.05%", 0.05, "%"),
        ("$399.00", 399.0, "$"),
        ("1 234,56 грн", 1234.56, "грн"),
        ("12,500", 12500.0, ""),
        ("12,5", 12.5, ""),
        ("1.234.567", 1234567.0, ""),
        ("-3.5", -3.5, ""),
        ("1.69E2", 169.0, ""),
    ],
)
def test_numbers_with_units_currencies_and_separators(text, value, unit):
    parsed = parse_number(text)
    assert parsed is not None
    assert parsed.value == pytest.approx(value)
    assert parsed.unit == unit


@pytest.mark.parametrize("text", ["310-246-1501", "12 Main St", "N/A", "7:10", "30, 22", "2011-12-01"])
def test_non_numbers_are_rejected(text):
    assert parse_number(text) is None


def test_lenient_parsing_takes_leading_number():
    assert parse_number("12.0 oz. Alumi-Tek", lenient=True).value == 12.0
    assert parse_number("12.0 oz. Alumi-Tek") is None


def test_decimal_comma_is_inferred_from_column():
    assert infer_decimal_comma(["12,5", "3,75", "1 200,00"]) is True
    assert infer_decimal_comma(["1,200", "12.5", "3.75"]) is False
    assert parse_number("1.234", decimal_comma=True).value == 1234.0
    assert parse_number("1,300", decimal_comma=False).value == 1300.0
    assert parse_number("1 300,25", decimal_comma=False).value == 1300.25


def test_numbers_are_formatted_without_float_noise():
    assert format_number(12.0) == "12"
    assert format_number(0.052000000000000005) == "0.052"
    assert format_number(1e-05) == "0.00001"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1984-01-01", date(1984, 1, 1)),
        ("12.03.2020", date(2020, 3, 12)),
        ("13/01/2020", date(2020, 1, 13)),
        ("12 березня 2020 р.", date(2020, 3, 12)),
        ("Dec 1, 2011", date(2011, 12, 1)),
    ],
)
def test_dates_in_different_formats(text, expected):
    assert parse_date(text).value == expected


def test_day_order_is_inferred_from_unambiguous_values():
    assert infer_dayfirst(["10/01/15", "04/30/15", "08/01/17"]) is False
    assert infer_dayfirst(["10.01.2015", "30.04.2015"]) is True
    assert parse_date("10/01/15", dayfirst=False).value == date(2015, 10, 1)


def test_dates_can_be_rendered_in_dominant_layout():
    layout = parse_date("04/30/15", dayfirst=False).layout
    assert format_date(date(2016, 2, 1), layout) == "02/01/16"
    assert format_date(date(2016, 2, 1)) == "2016-02-01"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("7:10 a.m.", time(7, 10)),
        ("5:56aDec 1", time(5, 56)),
        ("7:04pDec 1", time(19, 4)),
        ("12:00noon", time(12, 0)),
        ("Thu  19:33", time(19, 33)),
        ("11/30 7:35 p.m.", time(19, 35)),
        ("12:15 a.m.", time(0, 15)),
        ("10:30 PM", time(22, 30)),
    ],
)
def test_times_from_flights_sources(text, expected):
    assert parse_time(text).value == expected


def test_time_without_clock_is_rejected():
    assert parse_time("Dec 02 Contact Airline") is None


def test_time_layout_reproduces_source_style():
    layout = parse_time("7:10 a.m.").layout
    assert format_time(time(19, 5), layout) == "7:05 p.m."
    assert format_time(time(19, 5)) == "19:05"


def test_years():
    assert parse_year("1999") == 1999
    assert parse_year("1999.0") == 1999
    assert parse_year("99") is None


@pytest.mark.parametrize(
    ("text", "region", "e164"),
    [
        ("067 123 45 67", "UA", "+380671234567"),
        ("(044) 235-12-34", "UA", "+380442351234"),
        ("+38 (050) 123-45-67", None, "+380501234567"),
        ("380631234567", "UA", "+380631234567"),
        ("310/246-1501", "US", "+13102461501"),
    ],
)
def test_phones_are_converted_to_e164(text, region, e164):
    assert parse_phone(text, region).e164 == e164


def test_local_numbers_without_area_code_stay_incomplete():
    parsed = parse_phone("3865286", "US")
    assert parsed.e164 is None
    assert parsed.national == "3865286"


def test_phone_shape_detection():
    assert looks_like_phone("(847)\n245-6715")
    assert not looks_like_phone("8581-4703-5433-004 1234")
    assert not looks_like_phone("12")


def test_ordinary_number_notation_is_not_phone_formatting():
    for number in ("48.12345", "-33.86785", "1 234 567", "1.234.567", "0.123456"):
        assert not has_phone_formatting(number)
    for phone in ("555.123.4567", "310 246 1501", "067 123 456", "(555) 123-4567", "+380 67 123 45 67"):
        assert has_phone_formatting(phone)


def test_region_comes_only_from_international_numbers():
    assert infer_region(["+1 312 555 0100", "+1 773 555 0101", "312 555 0102"]) == "US"
    assert infer_region(["0044 20 7946 0958", "020 7946 0959"]) == "GB"
    assert infer_region(["067 123 45 67", "044 235 12 34"]) is None
    assert infer_region(["+1 312 555 0100", "+380 44 235 1234"]) is None


def test_email_and_url_detection():
    assert looks_like_email("phernandez@ymcachicago.org")
    assert normalize_email("mailto:Info@Example.COM") == "info@example.com"
    assert looks_like_url("www.cps.edu")
    assert looks_like_url("http://www.chicagosinai.org")
    assert not looks_like_url("Yes")
    assert normalize_url("HTTPS://www.Example.org/about/") == "example.org/about"


def test_mojibake_is_repaired():
    assert repair_mojibake("SjÃ¶strÃ¶m") == "Sjöström"
    assert repair_mojibake("РџСЂРёРІС–С‚") == "Привіт"
    assert repair_mojibake("Martha‚Äö√Ñ√¥s") == "Martha’s"
    assert repair_mojibake("São Paulo") == "São Paulo"
    assert repair_mojibake("Привіт") == "Привіт"


def test_clean_text_and_comparison_key():
    assert clean_text("  Chicago Commons\nAssociation  ") == "Chicago Commons Association"
    assert comparison_key("Café «Олеся»") == 'cafe "olesia"'
    assert comparison_key("Arnie Morton’s  of\nChicago") == "arnie morton's of chicago"
