from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import phonenumbers
from phonenumbers import PhoneMetadata, PhoneNumberFormat, ValidationResult

SUPPORTED_REGIONS = frozenset(phonenumbers.SUPPORTED_REGIONS)
COUNTRY_REGIONS = {
    code: tuple(region for region in regions if region in SUPPORTED_REGIONS)
    for code, regions in phonenumbers.COUNTRY_CODE_TO_REGION_CODE.items()
}
EXTENSION = re.compile(r"\s*(?:ext\.?|x|доб\.?)\s*\d{1,5}$", re.IGNORECASE)
PHONE_CHARACTERS = re.compile(r"^[\d\s()+\-./]+$")
NON_DIGITS = re.compile(r"\D")
PHONE_SEPARATORS = "()+-./ "
NUMBER_NOTATION = re.compile(
    r"^[-−]?(?:\d+(?:[.,]\d+)?|\d{1,3}(?: \d{3})+(?:[.,]\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{1,3}(?:\.\d{3})+(?:,\d+)?)$"
)
INTEGER_PART = re.compile(r"^[-−]?(\d+)")
INTERNATIONAL_PREFIXES = ("+", "00")
E164_MAX_DIGITS = 15
VALIDATION_SAMPLE = 1000
SAMPLE_SEED = 0


def _shortest_international_number() -> int:
    lengths = []
    for region in SUPPORTED_REGIONS:
        metadata = PhoneMetadata.metadata_for_region(region)
        possible = [length for length in metadata.general_desc.possible_length if length > 0]
        if possible:
            lengths.append(len(str(metadata.country_code)) + min(possible))
    return min(lengths)


MIN_DIGITS = _shortest_international_number()
MAX_DIGITS = E164_MAX_DIGITS


@dataclass(frozen=True)
class ParsedPhone:
    digits: str
    national: str
    e164: str | None
    valid: bool
    domestic: str = ""


def looks_like_phone(text: str) -> bool:
    core = EXTENSION.sub("", text.strip())
    if not PHONE_CHARACTERS.match(core):
        return False
    return MIN_DIGITS <= len(NON_DIGITS.sub("", core)) <= MAX_DIGITS


def has_phone_formatting(text: str) -> bool:
    core = text.strip()
    if not any(character in core for character in PHONE_SEPARATORS):
        return False
    return core.startswith(("+", "(")) or not _number_notation(core)


def is_international(text: str) -> bool:
    return text.strip().startswith(INTERNATIONAL_PREFIXES)


def parse_phone(text: str, region: str | None) -> ParsedPhone | None:
    core = _international(" ".join(EXTENSION.sub("", text.strip()).split()))
    digits = NON_DIGITS.sub("", core)
    if not digits:
        return None
    try:
        number = phonenumbers.parse(core, region)
    except phonenumbers.NumberParseException:
        return ParsedPhone(digits, digits, None, False)
    national = str(number.national_number)
    if phonenumbers.is_possible_number_with_reason(number) != ValidationResult.IS_POSSIBLE:
        return ParsedPhone(digits, national, None, False)
    e164 = phonenumbers.format_number(number, PhoneNumberFormat.E164)
    domestic = NON_DIGITS.sub("", phonenumbers.format_number(number, PhoneNumberFormat.NATIONAL))
    return ParsedPhone(digits, national, e164, phonenumbers.is_valid_number(number), domestic)


def international_regions(values: Iterable[str]) -> tuple[str, ...]:
    regions: Counter[str] = Counter()
    for value in values:
        if not is_international(value):
            continue
        try:
            code = phonenumbers.parse(_international(value.strip()), None).country_code
        except phonenumbers.NumberParseException:
            continue
        regions.update(COUNTRY_REGIONS.get(code, ()))
    return tuple(sorted(regions, key=lambda region: (-regions[region], region)))


def infer_region(values: Iterable[str], candidates: Sequence[str] | None = None) -> str | None:
    present = [value.strip() for value in values if value and value.strip()]
    regions = tuple(candidates) if candidates is not None else international_regions(present)
    local = sorted({value for value in present if not is_international(value)})
    if not regions or not local:
        return None
    if len(local) > VALIDATION_SAMPLE:
        chosen = np.random.default_rng(SAMPLE_SEED).choice(len(local), VALIDATION_SAMPLE, replace=False)
        local = [local[index] for index in sorted(chosen)]
    shares = {region: _valid_share(local, region) for region in regions}
    best = max(regions, key=lambda region: shares[region])
    return best if shares[best] > 0.5 else None


def _valid_share(values: Sequence[str], region: str) -> float:
    belonging = 0
    for value in values:
        try:
            number = phonenumbers.parse(_international(value), region)
        except phonenumbers.NumberParseException:
            continue
        belonging += bool(
            phonenumbers.is_valid_number(number) and phonenumbers.region_code_for_number(number) == region
        )
    return belonging / len(values)


def _number_notation(text: str) -> bool:
    if not NUMBER_NOTATION.match(text):
        return False
    integer = INTEGER_PART.match(text)
    return integer is not None and (integer.group(1) == "0" or not integer.group(1).startswith("0"))


def _international(text: str) -> str:
    return "+" + text[2:] if text.startswith("00") else text
