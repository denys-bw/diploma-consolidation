from __future__ import annotations

import numpy as np
import pandas as pd

from consolidator.linkage.pairs import pairs_within_groups
from consolidator.linkage.records import PairScope
from consolidator.values.counts import frequencies


def gold_pairs(truth: np.ndarray, source_codes: np.ndarray, scope: PairScope) -> np.ndarray:
    labels = pd.Series(np.asarray(truth, dtype=object)).fillna("").astype(str).str.strip()
    groups = pd.factorize(labels)[0].astype(np.int64)
    groups[(labels == "").to_numpy()] = -1
    return pairs_within_groups(groups, np.asarray(source_codes), scope)


def true_entities(truth: np.ndarray) -> int:
    labels = pd.Series(np.asarray(truth, dtype=object)).fillna("").astype(str).str.strip()
    return int(labels[labels != ""].nunique() + (labels == "").sum())


def entity_uniqueness(predicted: np.ndarray, truth: np.ndarray) -> float:
    if len(predicted) == 0:
        return 1.0
    frame = pd.DataFrame({"predicted": predicted, "truth": truth})
    majority = frame.groupby("predicted")["truth"].agg(lambda values: frequencies(values).index[0])
    return majority.nunique() / len(majority)
