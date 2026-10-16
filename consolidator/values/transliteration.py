from __future__ import annotations

import re

LETTERS = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "h",
    "ґ": "g",
    "д": "d",
    "е": "e",
    "є": "ie",
    "ж": "zh",
    "з": "z",
    "и": "y",
    "і": "i",
    "ї": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ь": "",
    "ю": "iu",
    "я": "ia",
}
WORD_INITIAL = {"є": "ye", "ї": "yi", "й": "y", "ю": "yu", "я": "ya"}
NON_UKRAINIAN = {"ё": "io", "ы": "y", "э": "e", "ъ": ""}
APOSTROPHES = frozenset("'’ʼ‘`´")
CYRILLIC = re.compile("[Ѐ-ӿ]")


def has_cyrillic(text: str) -> bool:
    return CYRILLIC.search(text) is not None


def transliterate(text: str) -> str:
    if not has_cyrillic(text):
        return text
    result: list[str] = []
    for position, character in enumerate(text):
        lower = character.lower()
        if lower in APOSTROPHES and _is_inside_word(text, position):
            continue
        latin = _latin(text, position, lower)
        if latin is None:
            result.append(character)
        elif character != lower:
            result.append(latin.upper() if _is_upper_word(text, position) else latin.capitalize())
        else:
            result.append(latin)
    return "".join(result)


def _latin(text: str, position: int, lower: str) -> str | None:
    if lower == "г" and position > 0 and text[position - 1].lower() == "з":
        return "gh"
    if lower in WORD_INITIAL and _starts_word(text, position):
        return WORD_INITIAL[lower]
    if lower in LETTERS:
        return LETTERS[lower]
    return NON_UKRAINIAN.get(lower)


def _starts_word(text: str, position: int) -> bool:
    if position == 0:
        return True
    previous = text[position - 1]
    return not previous.isalpha() and previous not in APOSTROPHES


def _is_inside_word(text: str, position: int) -> bool:
    return 0 < position < len(text) - 1 and text[position - 1].isalpha() and text[position + 1].isalpha()


def _is_upper_word(text: str, position: int) -> bool:
    neighbours = [
        text[index] for index in (position - 1, position + 1) if 0 <= index < len(text) and text[index].isalpha()
    ]
    return bool(neighbours) and all(neighbour.isupper() for neighbour in neighbours)
