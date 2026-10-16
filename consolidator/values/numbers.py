from __future__ import annotations

import math
import re
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from consolidator.values.units import CURRENCY_SYMBOLS, UNITS

CURRENCY_CLASS = re.escape("".join(sorted(CURRENCY_SYMBOLS)))
SPACES = re.compile(r"[\s  ']")
NUMBER = re.compile(
    rf"^(?P<sign>[-+−])?\s*(?P<prefix>[{CURRENCY_CLASS}]|[A-Z]{{3}}\s)?\s*(?P<sign2>[-+−])?"
    r"(?P<number>\d[\d\s  '.,]*\d|\d)(?P<exponent>[eE][-+]?\d{1,3})?(?P<suffix>.*)$",
    re.DOTALL,
)
LEADING_NUMBER = re.compile(
    rf"^(?P<sign>[-+−])?\s*(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>[%‰{CURRENCY_CLASS}]|[^\W\d_]+\.?)?(?![\w])",
)
UNIT_WORD = re.compile(r"^[^\W\d_]+\.?$")
CURRENCY_CODE = re.compile(r"^[A-Z]{3}$")
LIST_SEPARATOR = re.compile(r"[.,]\s")
THOUSANDS_COMMA = re.compile(r"^\d{1,3}(,\d{3})+$")
THOUSANDS_DOT = re.compile(r"^\d{1,3}(\.\d{3}){2,}$")
SINGLE_THOUSANDS_DOT = re.compile(r"^\d{1,3}\.\d{3}$")
SIGNIFICANT_DIGITS = sys.float_info.dig
MINUS_SIGNS = ("-", "−")


@dataclass(frozen=True)
class ParsedNumber:
    value: float
    unit: str
    decimals: int


def parse_number(text: str, decimal_comma: bool | None = None, lenient: bool = False) -> ParsedNumber | None:
    parsed = _parse_strict(text.strip(), decimal_comma)
    if parsed is None and lenient:
        parsed = _parse_leading(text.strip())
    return parsed


def unit_label(token: str) -> str | None:
    text = " ".join(token.split())
    if not text:
        return ""
    if text in CURRENCY_SYMBOLS:
        return text
    unit = UNITS.find(text)
    if unit is not None:
        return unit.symbol
    if CURRENCY_CODE.match(text):
        return text
    if UNIT_WORD.match(text):
        word = text.rstrip(".")
        return word.casefold() if len(word) > 1 else word
    return None


def is_known_unit(unit: str) -> bool:
    return unit == "" or unit in CURRENCY_SYMBOLS or UNITS.find(unit) is not None or bool(CURRENCY_CODE.match(unit))


def decimal_comma_evidence(text: str) -> bool | None:
    match = NUMBER.match(text.strip())
    if match is None:
        return None
    core = SPACES.sub("", match.group("number"))
    if "," in core and "." in core:
        return core.rfind(",") > core.rfind(".")
    if "," in core:
        return not THOUSANDS_COMMA.match(core)
    if "." in core:
        return bool(THOUSANDS_DOT.match(core))
    return None


def infer_decimal_comma(values: Iterable[str]) -> bool:
    votes = Counter(evidence for value in values if (evidence := decimal_comma_evidence(value)) is not None)
    return votes[True] > votes[False]


def format_number(value: float, decimals: int | None = None) -> str:
    if math.isnan(value) or math.isinf(value):
        return ""
    value = round(value, decimals) if decimals is not None else float(f"{value:.{SIGNIFICANT_DIGITS}g}")
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return np.format_float_positional(value, trim="-")


def _parse_strict(text: str, decimal_comma: bool | None) -> ParsedNumber | None:
    match = NUMBER.match(text)
    if match is None or LIST_SEPARATOR.search(match.group("number")):
        return None
    unit = _unit(match.group("prefix"), match.group("suffix"))
    if unit is None:
        return None
    core = SPACES.sub("", match.group("number"))
    exponent = match.group("exponent")
    if exponent:
        if "," in core or core.count(".") > 1:
            return None
        parsed: tuple[float, int] | None = (float(core + exponent), _decimals(core))
    else:
        parsed = _numeric_core(core, decimal_comma)
    if parsed is None:
        return None
    value, decimals = parsed
    if (match.group("sign") or match.group("sign2") or "") in MINUS_SIGNS:
        value = -value
    return ParsedNumber(value, unit, decimals)


def _parse_leading(text: str) -> ParsedNumber | None:
    match = LEADING_NUMBER.match(text)
    if match is None:
        return None
    core = match.group("number").replace(",", ".")
    value = float(core)
    if (match.group("sign") or "") in MINUS_SIGNS:
        value = -value
    return ParsedNumber(value, unit_label(match.group("unit") or "") or "", _decimals(core))


def _unit(prefix: str | None, suffix: str) -> str | None:
    prefix = (prefix or "").strip()
    suffix = " ".join(suffix.split())
    if prefix:
        if suffix:
            return None
        return prefix
    return unit_label(suffix)


def _numeric_core(core: str, decimal_comma: bool | None) -> tuple[float, int] | None:
    if "," in core and "." in core:
        decimal = "," if core.rfind(",") > core.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        if core.count(decimal) > 1:
            return None
        core = core.replace(thousands, "").replace(decimal, ".")
    elif "," in core:
        grouped = THOUSANDS_COMMA.match(core)
        if core.count(",") == 1 and (decimal_comma or not grouped):
            core = core.replace(",", ".")
        elif grouped:
            core = core.replace(",", "")
        else:
            return None
    elif core.count(".") > 1:
        if not THOUSANDS_DOT.match(core):
            return None
        core = core.replace(".", "")
    elif decimal_comma and SINGLE_THOUSANDS_DOT.match(core):
        core = core.replace(".", "")
    try:
        value = float(core)
    except ValueError:
        return None
    return value, _decimals(core)


def _decimals(core: str) -> int:
    return len(core.split(".", 1)[1]) if "." in core else 0
