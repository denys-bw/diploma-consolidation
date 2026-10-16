from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, time

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
    "січ": 1,
    "січень": 1,
    "січня": 1,
    "лют": 2,
    "лютий": 2,
    "лютого": 2,
    "бер": 3,
    "березень": 3,
    "березня": 3,
    "кві": 4,
    "квіт": 4,
    "квітень": 4,
    "квітня": 4,
    "тра": 5,
    "трав": 5,
    "травень": 5,
    "травня": 5,
    "чер": 6,
    "черв": 6,
    "червень": 6,
    "червня": 6,
    "лип": 7,
    "липень": 7,
    "липня": 7,
    "сер": 8,
    "серп": 8,
    "серпень": 8,
    "серпня": 8,
    "вер": 9,
    "вересень": 9,
    "вересня": 9,
    "жов": 10,
    "жовт": 10,
    "жовтень": 10,
    "жовтня": 10,
    "лис": 11,
    "листоп": 11,
    "листопад": 11,
    "листопада": 11,
    "гру": 12,
    "груд": 12,
    "грудень": 12,
    "грудня": 12,
}
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
YEAR_SUFFIX = re.compile(r"\s*(р\.|року|рік|г\.)\s*$", re.IGNORECASE)
CLOCK = r"(?:(?P<clock_separator>[T\s]+)(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"
ISO_DATE = re.compile(r"^(?P<year>\d{4})(?P<sep>[-/.])(?P<month>\d{1,2})(?P=sep)(?P<day>\d{1,2})" + CLOCK + r"$")
NUMERIC_DATE = re.compile(r"^(?P<a>\d{1,2})(?P<sep>[-/.])(?P<b>\d{1,2})(?P=sep)(?P<year>\d{4}|\d{2})" + CLOCK + r"$")
DAY_MONTH_NAME = re.compile(r"^(?P<day>\d{1,2})\s+(?P<month>[^\W\d_]+)\.?,?\s+(?P<year>\d{4})" + CLOCK + r"$")
MONTH_NAME_DAY = re.compile(
    r"^(?P<month>[^\W\d_]+)\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<year>\d{4})" + CLOCK + r"$"
)
TIME = re.compile(
    r"(?<!\d)(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?(?P<gap>\s*)"
    r"(?P<meridiem>[ap]\.\s?m\.?|[ap]m\b|[ap](?-i:(?![a-z]))|noon|midnight)?",
    re.IGNORECASE,
)
YEAR = re.compile(r"^(?P<year>\d{4})(?:\.0)?$")
FIRST_GREGORIAN_YEAR = 1583
LAST_FOUR_DIGIT_YEAR = 9999
TWO_DIGIT_PIVOT = 69


@dataclass(frozen=True)
class DateLayout:
    order: str
    separator: str
    padded: bool
    year_digits: int
    month_name: bool


@dataclass(frozen=True)
class ParsedDate:
    value: date
    clock: time | None
    layout: DateLayout
    ambiguous: bool
    clock_separator: str = ""


@dataclass(frozen=True)
class TimeLayout:
    twelve_hour: bool
    meridiem: tuple[str, str]
    padded: bool
    seconds: bool


@dataclass(frozen=True)
class ParsedTime:
    value: time
    layout: TimeLayout
    standalone: bool


def parse_date(text: str, dayfirst: bool | None = None) -> ParsedDate | None:
    text = YEAR_SUFFIX.sub("", text.strip())
    match = ISO_DATE.match(text)
    if match:
        layout = DateLayout("ymd", match["sep"], _padded(match["month"], match["day"]), 4, False)
        return _build(int(match["year"]), int(match["month"]), int(match["day"]), match, layout, False)
    match = NUMERIC_DATE.match(text)
    if match:
        return _numeric(match, dayfirst)
    for pattern, order in ((DAY_MONTH_NAME, "dmy"), (MONTH_NAME_DAY, "mdy")):
        match = pattern.match(text)
        if match and match["month"].casefold() in MONTHS:
            layout = DateLayout(order, " ", match["day"].startswith("0"), 4, True)
            return _build(
                int(match["year"]), MONTHS[match["month"].casefold()], int(match["day"]), match, layout, False
            )
    return None


def date_evidence(text: str) -> bool | None:
    match = NUMERIC_DATE.match(YEAR_SUFFIX.sub("", text.strip()))
    if not match:
        return None
    first, second = int(match["a"]), int(match["b"])
    if first > 12 >= second:
        return True
    if second > 12 >= first:
        return False
    return None


def infer_dayfirst(values: Iterable[str]) -> bool | None:
    votes = Counter(evidence for value in values if (evidence := date_evidence(value)) is not None)
    if votes[True] == votes[False]:
        return None
    return votes[True] > votes[False]


def parse_year(text: str) -> int | None:
    match = YEAR.match(text.strip())
    return int(match["year"]) if match else None


def parse_time(text: str) -> ParsedTime | None:
    match = TIME.search(text)
    if match is None:
        return None
    hour, minute = int(match["hour"]), int(match["minute"])
    second = int(match["second"]) if match["second"] else 0
    meridiem = match["meridiem"] or ""
    folded = meridiem.casefold().replace(".", "").replace(" ", "")
    if minute > 59 or second > 59:
        return None
    if folded in ("noon", "midnight"):
        hour = 12 if folded == "noon" else 0
    elif folded:
        if hour > 12:
            return None
        if folded.startswith("p") and hour != 12:
            hour += 12
        elif folded.startswith("a") and hour == 12:
            hour = 0
    elif hour > 23:
        return None
    gap = " " if match["gap"] else ""
    layout = TimeLayout(
        bool(folded), _meridiem_style(meridiem, gap), match["hour"].startswith("0"), bool(match["second"])
    )
    standalone = not (text[: match.start()].strip() or text[match.end() :].strip())
    return ParsedTime(time(hour, minute, second), layout, standalone)


def format_date(value: date, layout: DateLayout | None = None) -> str:
    if layout is None:
        return value.isoformat()
    if layout.month_name:
        month = MONTH_NAMES[value.month - 1]
        day = f"{value.day:02d}" if layout.padded else str(value.day)
        return f"{day} {month} {value.year}" if layout.order == "dmy" else f"{month} {day}, {value.year}"
    year = f"{value.year:04d}" if layout.year_digits == 4 else f"{value.year % 100:02d}"
    month = f"{value.month:02d}" if layout.padded else str(value.month)
    day = f"{value.day:02d}" if layout.padded else str(value.day)
    parts = {"ymd": (year, month, day), "dmy": (day, month, year), "mdy": (month, day, year)}[layout.order]
    return layout.separator.join(parts)


def format_time(value: time, layout: TimeLayout | None = None) -> str:
    if layout is None or not layout.twelve_hour:
        hour = f"{value.hour:02d}" if layout is None or layout.padded else str(value.hour)
        text = f"{hour}:{value.minute:02d}"
        return f"{text}:{value.second:02d}" if layout is not None and layout.seconds else text
    hour = value.hour % 12 or 12
    text = f"{hour:02d}" if layout.padded else str(hour)
    text = f"{text}:{value.minute:02d}" + (f":{value.second:02d}" if layout.seconds else "")
    return text + (layout.meridiem[0] if value.hour < 12 else layout.meridiem[1])


def format_datetime(
    value: date, clock: time | None, layout: DateLayout | None = None, separator: str | None = None
) -> str:
    text = format_date(value, layout)
    if clock is None:
        return text
    separator = separator or ("T" if layout is None else " ")
    return f"{text}{separator}{clock.hour:02d}:{clock.minute:02d}" + (f":{clock.second:02d}" if clock.second else "")


def _numeric(match: re.Match[str], dayfirst: bool | None) -> ParsedDate | None:
    first, second = int(match["a"]), int(match["b"])
    year = int(match["year"])
    digits = len(match["year"])
    if digits == 2:
        year += 2000 if year < TWO_DIGIT_PIVOT else 1900
    padded = _padded(match["a"], match["b"])
    orders = ("mdy", "dmy") if dayfirst is False else ("dmy", "mdy")
    valid = [order for order in orders if _valid(year, *_month_day(order, first, second))]
    if not valid:
        return None
    order = valid[0]
    month, day = _month_day(order, first, second)
    ambiguous = len(valid) == 2 and first != second and dayfirst is None
    layout = DateLayout(order, match["sep"], padded, digits, False)
    return _build(year, month, day, match, layout, ambiguous)


def _month_day(order: str, first: int, second: int) -> tuple[int, int]:
    return (second, first) if order == "dmy" else (first, second)


def _build(
    year: int, month: int, day: int, match: re.Match[str], layout: DateLayout, ambiguous: bool
) -> ParsedDate | None:
    if not _valid(year, month, day):
        return None
    clock, separator = None, ""
    if match["hour"] is not None:
        hour, minute = int(match["hour"]), int(match["minute"])
        second = int(match["second"]) if match["second"] else 0
        if hour > 23 or minute > 59 or second > 59:
            return None
        clock = time(hour, minute, second)
        separator = "T" if "T" in match["clock_separator"] else " "
    return ParsedDate(date(year, month, day), clock, layout, ambiguous, separator)


def _valid(year: int, month: int, day: int) -> bool:
    try:
        date(year, month, day)
    except ValueError:
        return False
    return FIRST_GREGORIAN_YEAR <= year <= LAST_FOUR_DIGIT_YEAR


def _padded(*parts: str) -> bool:
    return any(len(part) == 2 and part.startswith("0") for part in parts)


def _meridiem_style(meridiem: str, gap: str) -> tuple[str, str]:
    folded = meridiem.casefold()
    if not folded:
        return ("", "")
    if folded in ("noon", "midnight"):
        return (" a.m.", " p.m.")
    if "." in folded:
        am, pm = "a.m.", "p.m."
    elif folded in ("a", "p"):
        am, pm = "a", "p"
    else:
        am, pm = "am", "pm"
    if meridiem[0].isupper():
        am, pm = am.upper(), pm.upper()
    return (gap + am, gap + pm)
