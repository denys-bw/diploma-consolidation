from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from enum import IntEnum

from consolidator.statistics import is_majority
from consolidator.values.text import comparison_key

URI_SEPARATORS = re.compile(r"[/#]")
CAMEL_BOUNDARY = re.compile(r"(?<=[a-zа-яіїєґ0-9])(?=[A-ZА-ЯІЇЄҐ])")
SEPARATORS = re.compile(r"[\s_\-.:'\"]+")
DIGITS = re.compile(r"\d+")


class NameRelation(IntEnum):
    NONE = 0
    CONTAINS = 1
    ABBREVIATES = 2
    WORDS = 3
    SAME = 4


def readable_name(name: str) -> str:
    text = str(name).strip()
    if "://" in text:
        text = URI_SEPARATORS.split(text.rstrip("/#"))[-1]
    text = CAMEL_BOUNDARY.sub(" ", text)
    return " ".join(SEPARATORS.sub(" ", comparison_key(text)).split())


def name_relation(left: str, right: str) -> NameRelation:
    a, b = readable_name(left), readable_name(right)
    if not a or not b:
        return NameRelation.NONE
    if a == b:
        return NameRelation.SAME
    words_a, words_b = a.split(), b.split()
    if sorted(words_a) == sorted(words_b) or "".join(words_a) == "".join(words_b):
        return NameRelation.WORDS
    if len(words_a) == len(words_b) and all(x.startswith(y) or y.startswith(x) for x, y in zip(words_a, words_b)):
        return NameRelation.ABBREVIATES
    if _acronym(words_a, words_b) or _acronym(words_b, words_a):
        return NameRelation.ABBREVIATES
    if set(words_a) < set(words_b) or set(words_b) < set(words_a):
        return NameRelation.CONTAINS
    return NameRelation.NONE


def _acronym(short: list[str], long: list[str]) -> bool:
    return len(short) == 1 and len(long) > 1 and short[0] == "".join(word[0] for word in long)


def positional_names(names: Sequence[str]) -> set[str]:
    stems = {name: DIGITS.sub("", readable_name(name)).strip() for name in names if DIGITS.search(str(name))}
    counts = Counter(stems.values())
    return {name for name, stem in stems.items() if len(names) > 1 and is_majority(counts[stem] / len(names))}
