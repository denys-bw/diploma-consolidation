from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import jellyfish
import numpy as np
import pandas as pd

from consolidator.linkage.records import RecordSet
from consolidator.values.dates import parse_date, parse_year
from consolidator.values.text import WORD, comparison_key

DIGITS = re.compile(r"\D")
PART_SEPARATOR = "\x1f"


class KeyMethod(str, Enum):
    EXACT = "exact"
    PREFIX = "prefix"
    FIRST_TOKEN = "first_token"
    SOUNDEX = "soundex"
    SORTED_TOKENS = "sorted_tokens"
    YEAR = "year"
    DIGITS = "digits"


@dataclass(frozen=True)
class KeyPart:
    attribute: str
    method: KeyMethod = KeyMethod.EXACT
    length: int = 0

    @property
    def label(self) -> str:
        if self.method is KeyMethod.EXACT:
            return self.attribute
        if self.method is KeyMethod.PREFIX:
            return f"{self.attribute}[:{self.length}]"
        if self.method is KeyMethod.DIGITS:
            return f"{self.attribute}[-{self.length}:]" if self.length else f"{self.attribute}.digits"
        return f"{self.method.value}({self.attribute})"

    def values(self, records: RecordSet) -> np.ndarray:
        column = records.frame[self.attribute].astype(str)
        unique = pd.unique(column)
        transform = _transform(self)
        mapping = {value: transform(value) if value else "" for value in unique}
        return column.map(mapping).to_numpy(dtype=object)


@dataclass(frozen=True)
class BlockingKey:
    parts: tuple[KeyPart, ...]

    @classmethod
    def of(cls, attribute: str, method: KeyMethod = KeyMethod.EXACT, length: int = 0) -> BlockingKey:
        return cls((KeyPart(attribute, method, length),))

    def combined(self, other: BlockingKey) -> BlockingKey:
        return BlockingKey(self.parts + other.parts)

    @property
    def label(self) -> str:
        return " + ".join(part.label for part in self.parts)

    def values(self, records: RecordSet) -> np.ndarray:
        columns = [pd.Series(part.values(records), dtype=object) for part in self.parts]
        if len(columns) == 1:
            return columns[0].to_numpy(dtype=object)
        joined = columns[0]
        empty = columns[0] == ""
        for column in columns[1:]:
            joined = joined + PART_SEPARATOR + column
            empty |= column == ""
        joined[empty] = ""
        return joined.to_numpy(dtype=object)

    def groups(self, records: RecordSet) -> np.ndarray:
        values = self.values(records)
        codes, _ = pd.factorize(values)
        codes = codes.astype(np.int64)
        codes[values == ""] = -1
        return codes


def _transform(part: KeyPart) -> Callable[[str], str]:
    method, length = part.method, part.length
    if method is KeyMethod.EXACT:
        return comparison_key
    if method is KeyMethod.PREFIX:
        return lambda value: comparison_key(value).replace(" ", "")[:length]
    if method is KeyMethod.FIRST_TOKEN:
        return lambda value: next(iter(WORD.findall(comparison_key(value))), "")
    if method is KeyMethod.SOUNDEX:
        return lambda value: _soundex(next(iter(WORD.findall(comparison_key(value))), ""))
    if method is KeyMethod.SORTED_TOKENS:
        return lambda value: " ".join(sorted(WORD.findall(comparison_key(value))))
    if method is KeyMethod.YEAR:
        return _year_key
    if method is KeyMethod.DIGITS:
        return lambda value: DIGITS.sub("", value)[-length:] if length else DIGITS.sub("", value)
    raise ValueError(method)


def _soundex(token: str) -> str:
    letters = "".join(character for character in token if "a" <= character <= "z")
    return jellyfish.soundex(letters) if letters else token


def _year_key(value: str) -> str:
    year = parse_year(value)
    if year is None:
        parsed = parse_date(value)
        year = parsed.value.year if parsed is not None else None
    return str(year) if year is not None else ""
