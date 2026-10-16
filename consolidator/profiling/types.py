from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import Enum

import numpy as np
import pandas as pd

from consolidator.statistics import is_majority
from consolidator.values.counts import frequencies
from consolidator.values.dates import FIRST_GREGORIAN_YEAR, parse_date, parse_time, parse_year
from consolidator.values.missing import present_values
from consolidator.values.numbers import is_known_unit, parse_number
from consolidator.values.phones import has_phone_formatting, looks_like_phone
from consolidator.values.web import looks_like_email, looks_like_url


class ColumnType(str, Enum):
    EMPTY = "empty"
    IDENTIFIER = "identifier"
    CODE = "code"
    BOOLEAN = "boolean"
    CATEGORY = "category"
    NUMBER = "number"
    MEASURE = "measure"
    YEAR = "year"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    PHONE = "phone"
    EMAIL = "email"
    URL = "url"
    TEXT = "text"

    @property
    def is_quantitative(self) -> bool:
        return self in QUANTITATIVE_TYPES

    @property
    def is_textual(self) -> bool:
        return self in TEXTUAL_TYPES


QUANTITATIVE_TYPES = frozenset(
    {ColumnType.NUMBER, ColumnType.MEASURE, ColumnType.YEAR, ColumnType.DATE, ColumnType.DATETIME, ColumnType.TIME}
)
TEXTUAL_TYPES = frozenset({ColumnType.TEXT, ColumnType.CATEGORY})


@dataclass(frozen=True)
class TypeInference:
    type: ColumnType
    share: float
    present: int
    distinct: int


SAMPLE_DISTINCT = 3000
SAMPLE_SEED = 0
BOOLEAN_VALUES = frozenset({"true", "false", "yes", "no", "y", "n", "t", "f", "1", "0", "так", "ні", "+"})
TOKEN = re.compile(r"^(?=.*\d)[\w\-/.:#]{2,}$")


class ValueProbe:
    def __init__(self, counts: pd.Series, present: int) -> None:
        self.distinct = len(counts)
        self.present = present
        self.all_texts = counts.index.to_numpy(dtype=object)
        sample = (
            counts.sample(n=SAMPLE_DISTINCT, random_state=SAMPLE_SEED) if self.distinct > SAMPLE_DISTINCT else counts
        )
        self.texts = sample.index.to_numpy(dtype=object)
        self.weights = sample.to_numpy(dtype=float)
        self.total = float(self.weights.sum())

    @property
    def unique(self) -> bool:
        return self.distinct == self.present

    def share(self, predicate: Callable[[str], bool]) -> float:
        return float(self.weights[self.hits(predicate)].sum() / self.total)

    def hits(self, predicate: Callable[[str], bool]) -> np.ndarray:
        return np.fromiter((bool(predicate(text)) for text in self.texts), dtype=bool, count=len(self.texts))


class TypeRule(ABC):
    @abstractmethod
    def apply(self, probe: ValueProbe) -> tuple[ColumnType, float] | None:
        raise NotImplementedError


class BooleanRule(TypeRule):
    def apply(self, probe: ValueProbe) -> tuple[ColumnType, float] | None:
        if probe.distinct <= 2 and all(text.casefold() in BOOLEAN_VALUES for text in probe.all_texts):
            return ColumnType.BOOLEAN, 1.0
        return None


@dataclass(frozen=True)
class MajorityRule(TypeRule):
    column_type: ColumnType
    predicate: Callable[[str], bool]

    def apply(self, probe: ValueProbe) -> tuple[ColumnType, float] | None:
        share = probe.share(self.predicate)
        return (self.column_type, share) if is_majority(share) else None


class DateRule(TypeRule):
    def apply(self, probe: ValueProbe) -> tuple[ColumnType, float] | None:
        share = probe.share(_is_date)
        if not is_majority(share):
            return None
        clock = probe.share(_has_clock)
        return (ColumnType.DATETIME if is_majority(clock / share) else ColumnType.DATE), share


class PhoneRule(TypeRule):
    def apply(self, probe: ValueProbe) -> tuple[ColumnType, float] | None:
        share = probe.share(looks_like_phone)
        if not is_majority(share):
            return None
        formatted = probe.share(lambda text: looks_like_phone(text) and has_phone_formatting(text))
        return (ColumnType.PHONE, share) if is_majority(formatted / share) else None


class NumberRule(TypeRule):
    def apply(self, probe: ValueProbe) -> tuple[ColumnType, float] | None:
        parsed = [parse_number(text) for text in probe.texts]
        numeric = np.array([item is not None for item in parsed], dtype=bool)
        unknown = np.array([item is not None and not is_known_unit(item.unit) for item in parsed], dtype=bool)
        if unknown.any() and not _unit_vocabulary(parsed, unknown, probe.weights):
            numeric &= ~unknown
        share = float(probe.weights[numeric].sum() / probe.total)
        if not is_majority(share):
            return None
        with_unit = np.array([item is not None and item.unit != "" for item in parsed], dtype=bool) & numeric
        if is_majority(probe.weights[with_unit].sum() / probe.weights[numeric].sum()):
            return ColumnType.MEASURE, share
        digits = probe.hits(str.isdigit)
        if is_majority(float(probe.weights[digits].sum() / probe.total)):
            texts = probe.texts[digits]
            if _leading_zeros(texts):
                return (ColumnType.IDENTIFIER if probe.unique else ColumnType.CODE), share
            if not probe.unique and _fixed_width(texts):
                return ColumnType.CODE, share
            if probe.unique and _enumeration(probe.all_texts):
                return ColumnType.IDENTIFIER, share
        return ColumnType.NUMBER, share


class TokenRule(TypeRule):
    def apply(self, probe: ValueProbe) -> tuple[ColumnType, float] | None:
        share = probe.share(lambda text: TOKEN.match(text) is not None)
        if not is_majority(share):
            return None
        return (ColumnType.IDENTIFIER if probe.unique else ColumnType.CODE), share


class CategoryRule(TypeRule):
    def apply(self, probe: ValueProbe) -> tuple[ColumnType, float] | None:
        return (ColumnType.CATEGORY, 1.0) if probe.distinct <= math.sqrt(probe.present) else None


class TypeDetector:
    def __init__(self, rules: tuple[TypeRule, ...]) -> None:
        self.rules = rules

    def detect(self, present: pd.Series) -> TypeInference:
        present = present[present != ""]
        if present.empty:
            return TypeInference(ColumnType.EMPTY, 1.0, 0, 0)
        probe = ValueProbe(frequencies(present), len(present))
        for rule in self.rules:
            decision = rule.apply(probe)
            if decision is not None:
                return TypeInference(decision[0], decision[1], probe.present, probe.distinct)
        return TypeInference(ColumnType.TEXT, 1.0, probe.present, probe.distinct)


def _is_year(text: str) -> bool:
    year = parse_year(text)
    return year is not None and FIRST_GREGORIAN_YEAR <= year <= date.today().year


def _is_date(text: str) -> bool:
    return parse_date(text) is not None


def _has_clock(text: str) -> bool:
    parsed = parse_date(text)
    return parsed is not None and parsed.clock is not None


def _is_time(text: str) -> bool:
    return parse_time(text) is not None


def _unit_vocabulary(parsed: list, unknown: np.ndarray, weights: np.ndarray) -> bool:
    units = {parsed[index].unit for index in np.flatnonzero(unknown)}
    return len(units) <= math.sqrt(float(weights[unknown].sum()))


def _leading_zeros(texts: np.ndarray) -> bool:
    return any(len(text) > 1 and text.startswith("0") for text in texts)


def _fixed_width(texts: np.ndarray) -> bool:
    widths = {len(text) for text in texts}
    return len(widths) == 1 and next(iter(widths)) > 1


def _enumeration(texts: np.ndarray) -> bool:
    if len(texts) < 2 or not all(str(text).isdigit() for text in texts):
        return False
    numbers = np.array([int(text) for text in texts], dtype=np.int64)
    return int(numbers.max() - numbers.min()) + 1 == len(numbers)


DETECTOR = TypeDetector(
    (
        BooleanRule(),
        MajorityRule(ColumnType.EMAIL, looks_like_email),
        MajorityRule(ColumnType.URL, looks_like_url),
        MajorityRule(ColumnType.YEAR, _is_year),
        DateRule(),
        MajorityRule(ColumnType.TIME, _is_time),
        PhoneRule(),
        NumberRule(),
        TokenRule(),
        CategoryRule(),
    )
)


def infer_column_type(values: pd.Series) -> TypeInference:
    return infer_present_type(present_values(values).str.strip())


def infer_present_type(present: pd.Series) -> TypeInference:
    return DETECTOR.detect(present)
