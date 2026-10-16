from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import numpy as np
import pandas as pd

from consolidator.values.transliteration import transliterate

WHITESPACE = re.compile(r"\s+")
WORD = re.compile(r"\w+")
MOJIBAKE_ENCODINGS = ("cp1252", "cp1251", "mac_roman")
MOJIBAKE_ROUNDS = 3
LATIN_LIMIT = "ɐ"
PUNCTUATION_VARIANTS = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "ʼ": "'",
        "`": "'",
        "´": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
    }
)


def clean_text(value: str) -> str:
    return WHITESPACE.sub(" ", repair_mojibake(unicodedata.normalize("NFC", value))).strip()


def comparison_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold().translate(PUNCTUATION_VARIANTS)
    text = strip_latin_diacritics(transliterate(text))
    return WHITESPACE.sub(" ", text).strip()


def comparison_keys(values: Iterable[object]) -> np.ndarray:
    series = pd.Series(list(values) if not isinstance(values, (pd.Series, np.ndarray)) else values, dtype=object)
    series = series.fillna("").astype(str)
    mapping = {value: comparison_key(value) if value.strip() else "" for value in pd.unique(series)}
    return series.map(mapping).to_numpy(dtype=object)


def repair_mojibake(value: str) -> str:
    if value.isascii():
        return value
    current = value
    for _ in range(MOJIBAKE_ROUNDS):
        repaired = _repair_once(current)
        if repaired == current:
            break
        current = repaired
    return current


def strip_latin_diacritics(value: str) -> str:
    if value.isascii():
        return value
    kept: list[str] = []
    latin_base = False
    for character in unicodedata.normalize("NFD", value):
        if unicodedata.combining(character):
            if latin_base:
                continue
        else:
            latin_base = character < LATIN_LIMIT
        kept.append(character)
    return unicodedata.normalize("NFC", "".join(kept))


def _repair_once(value: str) -> str:
    for encoding in MOJIBAKE_ENCODINGS:
        try:
            candidate = value.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if candidate != value:
            return candidate
    return value
