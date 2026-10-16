from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

LATIN_PREFIXES = {
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "h": 1e2,
    "da": 1e1,
    "d": 1e-1,
    "c": 1e-2,
    "m": 1e-3,
    "µ": 1e-6,
    "n": 1e-9,
}
PREFIX_VARIANTS = {"µ": ("μ", "u")}
CYRILLIC_PREFIXES = {"T": "Т", "G": "Г", "M": "М", "k": "к", "d": "д", "c": "с", "m": "м", "µ": "мк", "n": "н"}
CURRENCY_SYMBOLS = frozenset(
    character for character in map(chr, range(0x10000)) if unicodedata.category(character) == "Sc"
)


@dataclass(frozen=True)
class Unit:
    symbol: str
    dimension: str
    factor: float
    offset: float = 0.0

    def to_base(self, value: float) -> float:
        return value * self.factor + self.offset

    def from_base(self, value: float) -> float:
        return (value - self.offset) / self.factor


@dataclass(frozen=True)
class BaseUnit:
    symbol: str
    dimension: str
    factor: float
    prefixes: tuple[str, ...]
    variants: tuple[str, ...] = ()
    cyrillic: str = ""
    names: tuple[str, ...] = ()


SI_UNITS = (
    BaseUnit("m", "length", 1.0, ("k", "h", "d", "c", "m", "µ", "n"), (), "м", ("metre", "meter", "metres", "meters")),
    BaseUnit("g", "mass", 1.0, ("k", "m", "µ"), (), "г", ("gram", "grams", "gramme", "grammes")),
    BaseUnit("t", "mass", 1e6, ("k", "M"), (), "т", ("tonne", "tonnes")),
    BaseUnit("l", "volume", 1.0, ("h", "da", "d", "c", "m", "µ"), ("L",), "л", ("litre", "liter", "litres", "liters")),
    BaseUnit("s", "time", 1.0, ("m", "µ", "n"), (), "с", ("sec", "second", "seconds")),
    BaseUnit("Hz", "frequency", 1.0, ("k", "M", "G", "T"), (), "Гц"),
    BaseUnit("W", "power", 1.0, ("m", "k", "M", "G", "T"), (), "Вт", ("watt", "watts")),
    BaseUnit("Wh", "energy", 3600.0, ("k", "M", "G", "T"), (), "Вт·год"),
    BaseUnit("J", "energy", 1.0, ("k", "M", "G"), (), "Дж"),
    BaseUnit("V", "voltage", 1.0, ("m", "k", "M"), (), "В", ("volt", "volts")),
    BaseUnit("A", "current", 1.0, ("m", "µ", "k"), (), "А", ("amp", "amps", "ampere", "amperes")),
    BaseUnit("Ah", "charge", 3600.0, ("m",), (), "А·год"),
    BaseUnit("Pa", "pressure", 1.0, ("h", "k", "M", "G"), (), "Па"),
    BaseUnit("bar", "pressure", 1e5, ("m",), (), "бар"),
    BaseUnit("B", "data", 1.0, ("k", "M", "G", "T"), (), "Б", ("byte", "bytes")),
    BaseUnit("bit", "data", 0.125, ("k", "M", "G", "T"), (), "біт", ("bits",)),
)
OTHER_UNITS = (
    ("min", "time", 60.0, ("хв", "minute", "minutes", "mins")),
    ("h", "time", 3600.0, ("год", "hr", "hrs", "hour", "hours")),
    ("d", "time", 86400.0, ("доба", "діб", "day", "days")),
    ("in", "length", 0.0254, ("inch", "inches", "″", '"', "дюйм", "дюймів")),
    ("ft", "length", 0.3048, ("foot", "feet", "′")),
    ("yd", "length", 0.9144, ("yard", "yards")),
    ("mi", "length", 1609.344, ("mile", "miles", "миля", "миль")),
    ("oz", "mass", 28.349523125, ("ounce", "ounces", "унція", "унцій")),
    ("lb", "mass", 453.59237, ("lbs", "pound", "pounds", "фунт", "фунтів")),
    ("fl oz", "volume", 0.0295735295625, ("fl. oz.", "fl.oz.", "fl oz.", "fl. oz", "fl.oz", "floz", "fluid ounce")),
    ("gal", "volume", 3.785411784, ("gallon", "gallons", "галон")),
    ("qt", "volume", 0.946352946, ("quart", "quarts")),
    ("pt", "volume", 0.473176473, ("pint", "pints")),
    ("%", "ratio", 0.01, ("percent", "per cent", "pct", "відс", "відс.", "відсотків", "відсотка")),
    ("‰", "ratio", 0.001, ("permille", "per mille")),
    ("pcs", "count", 1.0, ("pc", "pcs.", "piece", "pieces", "шт", "шт.")),
)
TEMPERATURES = (
    ("°C", 1.0, 273.15, ("℃", "degC")),
    ("°F", 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0, ("℉", "degF")),
)
POWER_DIMENSIONS = {2: "area", 3: "volume"}
POWER_SIGNS = {2: ("2", "²"), 3: ("3", "³")}
POWER_PREFIXES = ("", "k", "d", "c", "m")
CUBIC_METRE_IN_LITRES = 1000.0


class UnitRegistry:
    def __init__(self, entries: Iterable[tuple[Unit, Iterable[str]]]) -> None:
        exact: dict[str, Unit] = {}
        folded: dict[str, set[Unit]] = {}
        for unit, aliases in entries:
            for alias in (unit.symbol, *aliases):
                exact.setdefault(alias, unit)
                folded.setdefault(alias.casefold(), set()).add(unit)
        self._exact: Mapping[str, Unit] = exact
        self._folded: Mapping[str, Unit] = {alias: units.pop() for alias, units in folded.items() if len(units) == 1}

    def find(self, token: str) -> Unit | None:
        text = " ".join(token.split())
        if text in self._exact:
            return self._exact[text]
        return self._folded.get(text.casefold()) if len(text) > 1 else None

    def convert(self, value: float, source: Unit, target: Unit) -> float | None:
        if source.dimension != target.dimension:
            return None
        return target.from_base(source.to_base(value))


def standard_units() -> UnitRegistry:
    return UnitRegistry(_entries())


def _entries() -> Iterator[tuple[Unit, tuple[str, ...]]]:
    for base in SI_UNITS:
        yield _si_unit(base, "")
        for prefix in base.prefixes:
            yield _si_unit(base, prefix)
    for symbol, dimension, factor, aliases in OTHER_UNITS:
        yield Unit(symbol, dimension, factor), aliases
    for symbol, factor, offset, aliases in TEMPERATURES:
        yield Unit(symbol, "temperature", factor, offset), aliases
    for prefix in POWER_PREFIXES:
        for power, dimension in POWER_DIMENSIONS.items():
            scale = LATIN_PREFIXES.get(prefix, 1.0) ** power
            factor = scale * CUBIC_METRE_IN_LITRES if dimension == "volume" else scale
            latin = tuple(f"{prefix}m{sign}" for sign in POWER_SIGNS[power])
            cyrillic = tuple(f"{CYRILLIC_PREFIXES.get(prefix, '')}м{sign}" for sign in POWER_SIGNS[power])
            yield Unit(latin[1], dimension, factor), latin + cyrillic


def _si_unit(base: BaseUnit, prefix: str) -> tuple[Unit, tuple[str, ...]]:
    factor = base.factor * LATIN_PREFIXES.get(prefix, 1.0)
    unit = Unit(prefix + base.symbol, base.dimension, factor)
    spellings = (prefix, *PREFIX_VARIANTS.get(prefix, ()))
    aliases = [spelling + symbol for spelling in spellings for symbol in (base.symbol, *base.variants)]
    if base.cyrillic:
        cyrillic_prefix = CYRILLIC_PREFIXES.get(prefix, None if prefix else "")
        if cyrillic_prefix is not None:
            aliases.append(cyrillic_prefix + base.cyrillic)
    if not prefix:
        aliases.extend(base.names)
    return unit, tuple(aliases)


UNITS = standard_units()
