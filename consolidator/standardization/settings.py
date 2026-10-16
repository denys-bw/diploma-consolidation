from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class FormatPolicy(str, Enum):
    STANDARD = "standard"
    DOMINANT = "dominant"


class CellStatus(IntEnum):
    MISSING = 0
    UNCHANGED = 1
    CHANGED = 2
    INVALID = 3
    AMBIGUOUS = 4


@dataclass(frozen=True)
class StandardizationSettings:
    policy: FormatPolicy = FormatPolicy.STANDARD
    phone_region: str | None = None
    phone_format: str | None = None
    transliterate: bool = False
