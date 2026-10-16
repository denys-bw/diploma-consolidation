from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd

from consolidator.profiling.types import ColumnType
from consolidator.standardization.settings import CellStatus, FormatPolicy, StandardizationSettings
from consolidator.values.counts import frequencies
from consolidator.values.dates import (
    DateLayout,
    TimeLayout,
    format_date,
    format_datetime,
    format_time,
    infer_dayfirst,
    parse_date,
    parse_time,
    parse_year,
)
from consolidator.values.missing import missing_mask
from consolidator.values.numbers import decimal_comma_evidence, format_number, infer_decimal_comma, parse_number
from consolidator.values.patterns import value_shape
from consolidator.values.phones import (
    NON_DIGITS,
    ParsedPhone,
    infer_region,
    international_regions,
    looks_like_phone,
    parse_phone,
)
from consolidator.values.text import clean_text, comparison_key
from consolidator.values.transliteration import transliterate
from consolidator.values.units import UNITS
from consolidator.values.web import looks_like_url, normalize_email

TRUE_VALUES = frozenset({"true", "yes", "y", "t", "1", "так", "+"})
FALSE_VALUES = frozenset({"false", "no", "n", "f", "0", "ні"})
PERCENT = "%"
PERCENT_SCALES = (1.0, 0.01)
SECONDS_PER_DAY = 86_400
MINUTES_PER_DAY = 1_440

Parts = Mapping[str | None, pd.Series]


@dataclass(frozen=True, eq=False)
class StandardizedColumn:
    values: pd.Series
    status: pd.Series

    def count(self, status: CellStatus) -> int:
        return int((self.status == status).sum())


class ColumnStandardizer(ABC):
    def __init__(self, settings: StandardizationSettings | None = None) -> None:
        self.settings = settings or StandardizationSettings()

    @property
    def dominant(self) -> bool:
        return self.settings.policy is FormatPolicy.DOMINANT

    def fit(self, values: pd.Series | Parts) -> ColumnStandardizer:
        parts = {None: values} if isinstance(values, pd.Series) else dict(values)
        counts = {source: _present_counts(series) for source, series in parts.items()}
        for source, source_counts in counts.items():
            self._fit_source(source, source_counts)
        self._fit(counts)
        return self

    def transform(self, values: pd.Series, source: str | None = None) -> StandardizedColumn:
        text = values.fillna("").astype(str)
        missing = missing_mask(text)
        results = {raw: self._convert_cell(raw, source) for raw in pd.unique(text[~missing])}
        output = text.map({raw: result[0] for raw, result in results.items()}).astype(object)
        status = text.map({raw: int(result[1]) for raw, result in results.items()}).astype(object)
        output[missing] = ""
        status[missing] = int(CellStatus.MISSING)
        return StandardizedColumn(output, status.astype(np.int8))

    def fit_transform(self, values: pd.Series) -> StandardizedColumn:
        return self.fit(values).transform(values)

    @property
    def parameters(self) -> dict[str, str]:
        return {}

    def _fit_source(self, source: str | None, counts: pd.Series) -> None:
        return None

    def _fit(self, counts: Mapping[str | None, pd.Series]) -> None:
        return None

    @abstractmethod
    def convert(self, value: str, source: str | None = None) -> str | None:
        raise NotImplementedError

    def fallback(self, value: str) -> str:
        return clean_text(value)

    def render(self, number: float) -> str | None:
        return None

    def _clean(self, value: str) -> str:
        cleaned = clean_text(value)
        return transliterate(cleaned) if self.settings.transliterate else cleaned

    def ambiguous(self, value: str, source: str | None = None) -> bool:
        return False

    def _convert_cell(self, raw: str, source: str | None) -> tuple[str, CellStatus]:
        if self.ambiguous(raw, source):
            return clean_text(raw), CellStatus.AMBIGUOUS
        converted = self.convert(raw, source)
        if converted is None or converted == "":
            return self.fallback(raw), CellStatus.INVALID
        return converted, CellStatus.UNCHANGED if converted == raw else CellStatus.CHANGED


class TextStandardizer(ColumnStandardizer):
    def convert(self, value: str, source: str | None = None) -> str | None:
        return self._clean(value)


class CodeStandardizer(ColumnStandardizer):
    def convert(self, value: str, source: str | None = None) -> str | None:
        return " ".join(value.split())


class CategoryStandardizer(ColumnStandardizer):
    def __init__(self, settings: StandardizationSettings | None = None) -> None:
        super().__init__(settings)
        self.spellings: dict[str, str] = {}

    def _fit(self, counts: Mapping[str | None, pd.Series]) -> None:
        votes: dict[str, Counter[str]] = {}
        for raw, count in _combined(counts).items():
            cleaned = self._clean(str(raw))
            votes.setdefault(comparison_key(cleaned), Counter())[cleaned] += int(count)
        self.spellings = {
            key: min(counter.items(), key=lambda item: (-item[1], item[0]))[0] for key, counter in votes.items()
        }

    def convert(self, value: str, source: str | None = None) -> str | None:
        cleaned = self._clean(value)
        return self.spellings.get(comparison_key(cleaned), cleaned)


class BooleanStandardizer(ColumnStandardizer):
    def __init__(self, settings: StandardizationSettings | None = None) -> None:
        super().__init__(settings)
        self.labels = ("true", "false")

    def _fit(self, counts: Mapping[str | None, pd.Series]) -> None:
        if not self.dominant:
            return
        true_votes: Counter[str] = Counter()
        false_votes: Counter[str] = Counter()
        for raw, count in _combined(counts).items():
            stripped = str(raw).strip()
            if stripped.casefold() in TRUE_VALUES:
                true_votes[stripped] += int(count)
            elif stripped.casefold() in FALSE_VALUES:
                false_votes[stripped] += int(count)
        self.labels = (
            true_votes.most_common(1)[0][0] if true_votes else "true",
            false_votes.most_common(1)[0][0] if false_votes else "false",
        )

    def convert(self, value: str, source: str | None = None) -> str | None:
        folded = value.strip().casefold()
        if folded in TRUE_VALUES:
            return self.labels[0]
        if folded in FALSE_VALUES:
            return self.labels[1]
        return None


class NumberStandardizer(ColumnStandardizer):
    def __init__(self, settings: StandardizationSettings | None = None) -> None:
        super().__init__(settings)
        self.decimal_comma = False
        self.source_decimal_comma: dict[str | None, bool] = {}
        self.unit = ""
        self.percent_scale = 1.0
        self.decimals = 0

    @property
    def parameters(self) -> dict[str, str]:
        return {
            "unit": self.unit,
            "decimal_comma": str(self.decimal_comma),
            "percent_scale": str(self.percent_scale),
            "decimals": str(self.decimals),
        }

    def _fit_source(self, source: str | None, counts: pd.Series) -> None:
        if any(decimal_comma_evidence(str(raw)) is not None for raw in counts.index):
            self.source_decimal_comma[source] = infer_decimal_comma(str(raw) for raw in counts.index)

    def _fit(self, counts: Mapping[str | None, pd.Series]) -> None:
        combined = _combined(counts)
        self.decimal_comma = infer_decimal_comma(str(raw) for raw in combined.index)
        units: Counter[str] = Counter()
        percent: list[float] = []
        plain: list[float] = []
        for source, source_counts in counts.items():
            for raw, count in source_counts.items():
                parsed = parse_number(str(raw), self._comma(source), lenient=True)
                if parsed is None:
                    continue
                if parsed.unit:
                    units[parsed.unit] += int(count)
                target = percent if parsed.unit == PERCENT else plain if not parsed.unit else None
                if target is not None and parsed.value != 0:
                    target.extend([abs(parsed.value)] * int(count))
        others = Counter({unit: count for unit, count in units.items() if unit != PERCENT})
        self.unit = others.most_common(1)[0][0] if others else (PERCENT if units and not plain else "")
        self.percent_scale = _percent_scale(percent, plain) if self.unit != PERCENT else 1.0
        converted = (
            self.convert(str(raw), source) for source, source_counts in counts.items() for raw in source_counts.index
        )
        self.decimals = max((len(text.partition(".")[2]) for text in converted if text), default=0)

    def render(self, number: float) -> str | None:
        return format_number(number, self.decimals) if math.isfinite(number) else None

    def convert(self, value: str, source: str | None = None) -> str | None:
        parsed = parse_number(value, self._comma(source), lenient=True)
        if parsed is None:
            return None
        number = parsed.value
        if parsed.unit == PERCENT and self.unit != PERCENT:
            number *= self.percent_scale
        elif parsed.unit and self.unit and parsed.unit != self.unit:
            converted = _convert_unit(number, parsed.unit, self.unit)
            if converted is None:
                return None
            number = converted
        return format_number(number)

    def _comma(self, source: str | None) -> bool:
        return self.source_decimal_comma.get(source, self.decimal_comma)


class YearStandardizer(ColumnStandardizer):
    def convert(self, value: str, source: str | None = None) -> str | None:
        year = parse_year(value)
        if year is None:
            parsed = parse_date(value)
            year = parsed.value.year if parsed is not None else None
        return f"{year:04d}" if year is not None else None

    def render(self, number: float) -> str | None:
        return f"{round(number):04d}" if math.isfinite(number) and number >= 0 else None


class DateStandardizer(ColumnStandardizer):
    def __init__(self, settings: StandardizationSettings | None = None) -> None:
        super().__init__(settings)
        self.dayfirst: bool | None = None
        self.source_dayfirst: dict[str | None, bool | None] = {}
        self.layout: DateLayout | None = None

    @property
    def parameters(self) -> dict[str, str]:
        described = {source: value for source, value in self.source_dayfirst.items() if source is not None}
        return {
            "dayfirst": "" if self.dayfirst is None else str(self.dayfirst),
            "sources": ", ".join(f"{source}: {'' if value is None else value}" for source, value in described.items()),
            "layout": "" if self.layout is None else repr(self.layout),
        }

    def _fit_source(self, source: str | None, counts: pd.Series) -> None:
        self.source_dayfirst[source] = infer_dayfirst(str(raw) for raw in counts.index)

    def _fit(self, counts: Mapping[str | None, pd.Series]) -> None:
        self.dayfirst = infer_dayfirst(str(raw) for raw in _combined(counts).index)
        if not self.dominant:
            return
        layouts: Counter[DateLayout] = Counter()
        for source, source_counts in counts.items():
            for raw, count in source_counts.items():
                parsed = parse_date(str(raw), self._dayfirst(source))
                if parsed is not None and not parsed.ambiguous:
                    layouts[parsed.layout] += int(count)
        self.layout = layouts.most_common(1)[0][0] if layouts else None

    def ambiguous(self, value: str, source: str | None = None) -> bool:
        parsed = parse_date(value, self._dayfirst(source))
        return parsed is not None and parsed.ambiguous

    def convert(self, value: str, source: str | None = None) -> str | None:
        parsed = parse_date(value, self._dayfirst(source))
        return format_date(parsed.value, self.layout) if parsed is not None else None

    def render(self, number: float) -> str | None:
        day = _day(round(number)) if math.isfinite(number) else None
        return format_date(day, self.layout) if day is not None else None

    def _dayfirst(self, source: str | None) -> bool | None:
        own = self.source_dayfirst.get(source)
        return self.dayfirst if own is None else own


class DateTimeStandardizer(DateStandardizer):
    def __init__(self, settings: StandardizationSettings | None = None) -> None:
        super().__init__(settings)
        self.separator: str | None = None

    @property
    def parameters(self) -> dict[str, str]:
        return {**super().parameters, "separator": self.separator or ""}

    def _fit(self, counts: Mapping[str | None, pd.Series]) -> None:
        super()._fit(counts)
        if not self.dominant:
            return
        separators: Counter[str] = Counter()
        for source, source_counts in counts.items():
            for raw, count in source_counts.items():
                parsed = parse_date(str(raw), self._dayfirst(source))
                if parsed is not None and not parsed.ambiguous and parsed.clock_separator:
                    separators[parsed.clock_separator] += int(count)
        self.separator = separators.most_common(1)[0][0] if separators else None

    def convert(self, value: str, source: str | None = None) -> str | None:
        parsed = parse_date(value, self._dayfirst(source))
        if parsed is None:
            return None
        return format_datetime(parsed.value, parsed.clock, self.layout, self.separator)

    def render(self, number: float) -> str | None:
        if not math.isfinite(number):
            return None
        days, seconds = divmod(round(number * SECONDS_PER_DAY), SECONDS_PER_DAY)
        day = _day(days)
        return format_datetime(day, _clock(seconds), self.layout, self.separator) if day is not None else None


class TimeStandardizer(ColumnStandardizer):
    def __init__(self, settings: StandardizationSettings | None = None) -> None:
        super().__init__(settings)
        self.layout: TimeLayout | None = None

    @property
    def parameters(self) -> dict[str, str]:
        return {"layout": "" if self.layout is None else repr(self.layout)}

    def _fit(self, counts: Mapping[str | None, pd.Series]) -> None:
        if not self.dominant:
            return
        layouts: Counter[TimeLayout] = Counter()
        for raw, count in _combined(counts).items():
            parsed = parse_time(str(raw))
            if parsed is not None and parsed.standalone:
                layouts[parsed.layout] += int(count)
        self.layout = layouts.most_common(1)[0][0] if layouts else None

    def convert(self, value: str, source: str | None = None) -> str | None:
        parsed = parse_time(value)
        return format_time(parsed.value, self.layout) if parsed is not None else None

    def render(self, number: float) -> str | None:
        if not math.isfinite(number):
            return None
        if self.layout is not None and self.layout.seconds:
            return format_time(_clock(round(number * 60) % SECONDS_PER_DAY), self.layout)
        return format_time(_clock(round(number) % MINUTES_PER_DAY * 60), self.layout)


class PhoneStandardizer(ColumnStandardizer):
    def __init__(self, settings: StandardizationSettings | None = None) -> None:
        super().__init__(settings)
        self.region = self.settings.phone_region
        self.source_regions: dict[str | None, str | None] = {}
        self.template = self.settings.phone_format or ""

    @property
    def parameters(self) -> dict[str, str]:
        regions = {source: region for source, region in self.source_regions.items() if source is not None}
        described = ", ".join(f"{source}: {region or '—'}" for source, region in regions.items())
        return {"region": self.region or "", "regions": described, "template": self.template}

    def _fit(self, counts: Mapping[str | None, pd.Series]) -> None:
        if self.settings.phone_region is None:
            combined = [str(raw) for raw in _combined(counts).index]
            candidates = international_regions(combined)
            self.region = infer_region(combined, candidates)
            for source, source_counts in counts.items():
                self.source_regions[source] = infer_region((str(raw) for raw in source_counts.index), candidates)
        if self.dominant and not self.settings.phone_format:
            shapes: Counter[str] = Counter()
            for source, source_counts in counts.items():
                for raw, count in source_counts.items():
                    if parse_phone(str(raw), self._region(source)) is not None:
                        shapes[value_shape(" ".join(str(raw).split()))] += int(count)
            self.template = shapes.most_common(1)[0][0] if shapes else ""

    def convert(self, value: str, source: str | None = None) -> str | None:
        if not looks_like_phone(value):
            return None
        region = self._region(source)
        parsed = parse_phone(value, region)
        if parsed is None or (parsed.e164 is None and region is not None and not self.dominant):
            return None
        digits = self._template_digits(parsed)
        if digits is not None:
            symbols = iter(digits)
            return "".join(next(symbols) if symbol == "9" else symbol for symbol in self.template)
        return parsed.e164 or parsed.digits

    def fallback(self, value: str) -> str:
        digits = NON_DIGITS.sub("", value)
        return digits or clean_text(value)

    def _region(self, source: str | None) -> str | None:
        if self.settings.phone_region is not None:
            return self.settings.phone_region
        return self.source_regions.get(source) or self.region

    def _template_digits(self, parsed: ParsedPhone) -> str | None:
        slots = self.template.count("9")
        if not slots:
            return None
        if parsed.e164 is not None:
            international = parsed.e164.removeprefix("+")
            for digits in (parsed.national, parsed.domestic, international):
                if len(digits) == slots:
                    return digits
            return None
        return parsed.digits if len(parsed.digits) == slots else None


class EmailStandardizer(ColumnStandardizer):
    def convert(self, value: str, source: str | None = None) -> str | None:
        return normalize_email(value)


class UrlStandardizer(ColumnStandardizer):
    def convert(self, value: str, source: str | None = None) -> str | None:
        stripped = value.strip()
        if not looks_like_url(stripped):
            return None
        scheme, separator, rest = stripped.partition("://")
        if not separator:
            scheme, rest = "", stripped
        host, slash, path = rest.partition("/")
        prefix = f"{scheme.lower()}://" if scheme else ""
        return f"{prefix}{host.lower()}{slash}{path}".rstrip("/")


STANDARDIZERS: dict[ColumnType, type[ColumnStandardizer]] = {
    ColumnType.EMPTY: TextStandardizer,
    ColumnType.TEXT: TextStandardizer,
    ColumnType.CATEGORY: CategoryStandardizer,
    ColumnType.IDENTIFIER: CodeStandardizer,
    ColumnType.CODE: CodeStandardizer,
    ColumnType.BOOLEAN: BooleanStandardizer,
    ColumnType.NUMBER: NumberStandardizer,
    ColumnType.MEASURE: NumberStandardizer,
    ColumnType.YEAR: YearStandardizer,
    ColumnType.DATE: DateStandardizer,
    ColumnType.DATETIME: DateTimeStandardizer,
    ColumnType.TIME: TimeStandardizer,
    ColumnType.PHONE: PhoneStandardizer,
    ColumnType.EMAIL: EmailStandardizer,
    ColumnType.URL: UrlStandardizer,
}


def standardizer_for(column_type: ColumnType, settings: StandardizationSettings | None = None) -> ColumnStandardizer:
    return STANDARDIZERS[column_type](settings)


def _present_counts(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str)
    return frequencies(text[~missing_mask(text)])


def _combined(counts: Mapping[str | None, pd.Series]) -> pd.Series:
    present = [series for series in counts.values() if len(series)]
    if not present:
        return pd.Series(dtype=np.int64)
    return pd.concat(present).groupby(level=0, sort=False).sum().sort_values(ascending=False, kind="stable")


def _percent_scale(percent: list[float], plain: list[float]) -> float:
    if not percent or not plain:
        return 1.0
    target = math.log(float(np.median(plain)))
    level = math.log(float(np.median(percent)))
    return min(PERCENT_SCALES, key=lambda scale: abs(level + math.log(scale) - target))


def _day(ordinal: int) -> date | None:
    return date.fromordinal(ordinal) if 1 <= ordinal <= date.max.toordinal() else None


def _clock(seconds: int) -> time:
    return time(seconds // 3600, seconds // 60 % 60, seconds % 60)


def _convert_unit(value: float, source: str, target: str) -> float | None:
    source_unit, target_unit = UNITS.find(source), UNITS.find(target)
    if source_unit is None or target_unit is None:
        return None
    return UNITS.convert(value, source_unit, target_unit)
