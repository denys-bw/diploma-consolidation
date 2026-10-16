from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import normalize

TOKEN_PATTERN = r"(?u)\b\w+\b"
QGRAM = 3
HASH_FEATURES = 1 << 20
CHUNK_CELLS = 4_000_000


def tfidf(texts: Sequence[str]) -> sparse.csr_matrix:
    documents = list(texts)
    if not any(documents):
        return sparse.csr_matrix((len(documents), 0), dtype=np.float32)
    try:
        counts = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, dtype=np.float32).fit_transform(
            documents
        )
    except ValueError:
        return sparse.csr_matrix((len(documents), 0), dtype=np.float32)
    counts = counts.tocsr()
    counts.data = 1.0 + np.log(counts.data)
    frequency = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float64)
    idf = np.log(len(documents) / np.maximum(frequency, 1.0)).astype(np.float32)
    weighted = counts @ sparse.diags(idf)
    weighted.eliminate_zeros()
    return normalize(weighted.tocsr(), copy=False)


def qgram_vectors(texts: Sequence[str]) -> sparse.csr_matrix:
    vectorizer = HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(QGRAM, QGRAM),
        lowercase=False,
        alternate_sign=False,
        n_features=HASH_FEATURES,
        norm="l2",
        dtype=np.float32,
    )
    return vectorizer.transform(list(texts)).tocsr()


def row_cosine(left: sparse.csr_matrix, right: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(left.multiply(right).sum(axis=1), dtype=np.float64).ravel()


def nearest(queries: sparse.csr_matrix, targets: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    best = np.full(queries.shape[0], -1, dtype=np.int64)
    scores = np.zeros(queries.shape[0], dtype=np.float64)
    if queries.shape[0] == 0 or targets.shape[0] == 0 or queries.shape[1] == 0:
        return best, scores
    transposed = targets.T.tocsr()
    step = max(1, CHUNK_CELLS // targets.shape[0])
    for start in range(0, queries.shape[0], step):
        similarity = (queries[start : start + step] @ transposed).toarray()
        column = similarity.argmax(axis=1)
        value = similarity[np.arange(len(column)), column]
        best[start : start + step] = np.where(value > 0, column, -1)
        scores[start : start + step] = value
    return best, scores


def mutual_nearest(left: sparse.csr_matrix, right: sparse.csr_matrix) -> np.ndarray:
    forward, _ = nearest(left, right)
    backward, _ = nearest(right, left)
    rows = np.flatnonzero(forward >= 0)
    mutual = rows[backward[forward[rows]] == rows]
    return np.column_stack([mutual, forward[mutual]]).astype(np.int64)
