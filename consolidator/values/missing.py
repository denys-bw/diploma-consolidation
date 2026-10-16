from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_MARKERS = frozenset(
    {
        "",
        "-",
        "--",
        "---",
        "–",
        "—",
        "?",
        "??",
        "n/a",
        "n.a.",
        "n.a",
        "#n/a",
        "nan",
        "null",
        "none",
        "nil",
        "not available",
        "not applicable",
        "unknown",
        "missing",
        "undefined",
        "немає",
        "нема",
        "невідомо",
        "відсутнє",
        "відсутній",
        "н/д",
        "н.д.",
        "не вказано",
        "не визначено",
    }
)


def is_missing(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    return str(value).strip().casefold() in MISSING_MARKERS


def missing_mask(values: pd.Series) -> np.ndarray:
    text = values.fillna("").astype(str)
    counts = text.value_counts(sort=False)
    flags = pd.Series([is_missing(value) for value in counts.index], index=counts.index, dtype=bool)
    return text.map(flags).to_numpy(dtype=bool)


def present_values(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str)
    return text[~missing_mask(text)]
