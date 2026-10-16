from __future__ import annotations

import pandas as pd


def frequencies(values: pd.Series) -> pd.Series:
    return values.value_counts(sort=False).sort_values(ascending=False, kind="stable")
