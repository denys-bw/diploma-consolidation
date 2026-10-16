from __future__ import annotations

import math
import re
from datetime import datetime

from flask import Flask

from consolidator.profiling import ColumnType
from consolidator.reporting.labels import TYPE_LABELS

SPACE = " "
GROUPING_FROM = 10_000
INTEGER_TEXT = re.compile(r"^-?\d+$")
DECIMAL_TEXT = re.compile(r"^-?\d+\.\d+$")
SCIENTIFIC_TEXT = re.compile(r"^-?\d+(\.\d+)?[eE][-+]?\d+$")
NUMERIC_TEXT = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
RAW_COLUMNS = frozenset(
    {
        "Сутність",
        "Найчастіше значення",
        "Переважний шаблон",
        "Мінімум",
        "Медіана",
        "Максимум",
        "Нижня межа IQR",
        "Верхня межа IQR",
        "Обране значення",
        "Рівень подібності",
    }
)
PERCENT_COLUMNS = frozenset(
    {
        "Повнота",
        "Частка значень цього типу",
        "Частка шаблону",
        "Впевненість",
        "Оцінена точність",
    }
)
SIZE_UNITS = ("Б", "КБ", "МБ", "ГБ")


def number(value: object, digits: int | None = None) -> str:
    if value is None or isinstance(value, bool):
        return "" if value is None else ("так" if value else "ні")
    if isinstance(value, str):
        return _number_text(value.strip())
    if isinstance(value, int):
        return _grouped(str(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return _number_text(f"{value:.{4 if digits is None else digits}f}")
    return str(value)


def percent(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value * 100:.{digits}f}".replace(".", ",") + SPACE + "%"


def points(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return ""
    sign = "+" if value > 0 else ("−" if value < 0 else "")
    return f"{sign}{abs(value) * 100:.{digits}f}".replace(".", ",") + SPACE + "п." + SPACE + "п."


def cell(value: object, column: str) -> str:
    if column in RAW_COLUMNS:
        return "" if value is None else str(value)
    if column in PERCENT_COLUMNS:
        fraction = _fraction(value)
        if fraction is not None:
            return percent(fraction)
    return number(value)


def alignment(value: object, column: str = "") -> str:
    if isinstance(value, bool) or column in RAW_COLUMNS:
        return ""
    if isinstance(value, (int, float)) or (isinstance(value, str) and NUMERIC_TEXT.match(value.strip())):
        return "num"
    return ""


def sentence(text: object) -> str:
    text = "" if text is None else str(text)
    return text[:1].upper() + text[1:]


def type_label(value: ColumnType | str) -> str:
    return TYPE_LABELS[ColumnType(value)]


def timestamp(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone().strftime("%d.%m.%Y %H:%M")


def duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return ""
    if seconds < 10:
        return f"{seconds:.1f}".replace(".", ",") + SPACE + "с"
    if seconds < 60:
        return f"{seconds:.0f}" + SPACE + "с"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}{SPACE}хв {rest:02d}{SPACE}с"


def file_size(size: int) -> str:
    value = float(size)
    for unit in SIZE_UNITS:
        if value < 1024 or unit == SIZE_UNITS[-1]:
            text = f"{value:.0f}" if unit == SIZE_UNITS[0] else f"{value:.1f}".replace(".", ",")
            return text + SPACE + unit
        value /= 1024
    return str(size)


def register_filters(app: Flask) -> None:
    app.jinja_env.filters.update(
        number=number,
        percent=percent,
        points=points,
        cell=cell,
        align=alignment,
        sentence=sentence,
        type_label=type_label,
        timestamp=timestamp,
        duration=duration,
        file_size=file_size,
    )


def _fraction(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    text = str(value).strip()
    return float(text) if NUMERIC_TEXT.match(text) else None


def _number_text(text: str) -> str:
    if INTEGER_TEXT.match(text):
        return _grouped(text)
    if DECIMAL_TEXT.match(text):
        whole, fraction = text.split(".")
        return f"{_grouped(whole)},{fraction}"
    if SCIENTIFIC_TEXT.match(text):
        return f"{float(text):.2e}".replace(".", ",")
    return text


def _grouped(whole: str) -> str:
    sign, digits = ("-", whole[1:]) if whole.startswith("-") else ("", whole)
    if int(digits) < GROUPING_FROM:
        return whole
    groups = []
    while digits:
        groups.insert(0, digits[-3:])
        digits = digits[:-3]
    return sign + SPACE.join(groups)
