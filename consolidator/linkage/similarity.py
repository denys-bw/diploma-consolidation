from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from scipy import sparse

ROW_CHUNK = 200_000


def jaro_winkler(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    return _pairwise(left, right, JaroWinkler.normalized_similarity)


def levenshtein(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    return _pairwise(left, right, Levenshtein.normalized_similarity)


def token_set(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    return _pairwise(left, right, fuzz.token_set_ratio) / 100.0


def exact(left: Sequence[str], right: Sequence[str]) -> np.ndarray:
    return (np.asarray(left, dtype=object) == np.asarray(right, dtype=object)).astype(np.float32)


def cosine_rows(matrix: sparse.csr_matrix, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), ROW_CHUNK):
        stop = start + ROW_CHUNK
        product = matrix[left[start:stop]].multiply(matrix[right[start:stop]])
        result[start:stop] = np.asarray(product.sum(axis=1)).ravel()
    return np.clip(result, 0.0, 1.0)


def jaccard_rows(matrix: sparse.csr_matrix, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    binary = (matrix > 0).astype(np.float32).tocsr()
    sizes = np.asarray(binary.sum(axis=1)).ravel()
    result = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), ROW_CHUNK):
        stop = start + ROW_CHUNK
        a, b = left[start:stop], right[start:stop]
        common = np.asarray(binary[a].multiply(binary[b]).sum(axis=1)).ravel()
        union = sizes[a] + sizes[b] - common
        result[start:stop] = np.divide(common, union, out=np.zeros_like(common), where=union > 0)
    return result


def _pairwise(left: Sequence[str], right: Sequence[str], scorer) -> np.ndarray:
    if len(left) == 0:
        return np.zeros(0, dtype=np.float32)
    return process.cpdist(list(left), list(right), scorer=scorer, dtype=np.float32, workers=-1)
