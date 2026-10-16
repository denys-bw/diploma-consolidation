from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from consolidator.linkage import similarity as metrics
from consolidator.linkage.records import PairScope, RecordSet
from consolidator.profiling.numeric import numeric_values
from consolidator.profiling.types import ColumnType
from consolidator.statistics import sturges_bins
from consolidator.values.phones import MIN_DIGITS, NON_DIGITS
from consolidator.values.text import comparison_keys
from consolidator.values.web import normalize_url

COMPARISON_BATCH = 500_000
CALIBRATION_PAIRS = 20_000
CALIBRATION_SEED = 0
MINUTES_PER_DAY = 1_440.0
TOKEN_PATTERN = r"(?u)\b\w+\b"


@dataclass(frozen=True, eq=False)
class ComparisonMatrix:
    attributes: tuple[str, ...]
    metrics: tuple[str, ...]
    similarities: np.ndarray
    exact: np.ndarray
    frequencies: np.ndarray
    structural: tuple[int, ...]

    @cached_property
    def _scheme(self) -> tuple[np.ndarray, tuple[int, ...]]:
        levels = np.full(self.similarities.shape, -1, dtype=np.int16)
        sizes = []
        for index, fixed in enumerate(self.structural):
            similarity = self.similarities[:, index]
            present = ~np.isnan(similarity)
            if fixed:
                levels[present, index] = np.rint(similarity[present] * (fixed - 1)).astype(np.int16)
                sizes.append(fixed)
                continue
            graded = present & ~self.exact[:, index]
            bins = sturges_bins(int(graded.sum()))
            levels[graded, index] = np.minimum(np.floor(similarity[graded] * bins), bins - 1).astype(np.int16)
            levels[present & self.exact[:, index], index] = bins
            sizes.append(bins + 1)
        return levels, tuple(sizes)

    @property
    def levels(self) -> np.ndarray:
        return self._scheme[0]

    @property
    def n_levels(self) -> tuple[int, ...]:
        return self._scheme[1]

    def __len__(self) -> int:
        return len(self.similarities)

    def column(self, attribute: str) -> np.ndarray:
        return self.similarities[:, self.attributes.index(attribute)]

    def subset(self, attributes: Sequence[str]) -> ComparisonMatrix:
        index = [self.attributes.index(attribute) for attribute in attributes]
        return ComparisonMatrix(
            tuple(attributes),
            tuple(self.metrics[i] for i in index),
            self.similarities[:, index],
            self.exact[:, index],
            self.frequencies[:, index],
            tuple(self.structural[i] for i in index),
        )

    def rows(self, selection: np.ndarray) -> ComparisonMatrix:
        return ComparisonMatrix(
            self.attributes,
            self.metrics,
            self.similarities[selection],
            self.exact[selection],
            self.frequencies[selection],
            self.structural,
        )

    def replaced(self, attribute: str, other: ComparisonMatrix) -> ComparisonMatrix:
        index = self.attributes.index(attribute)
        source = other.attributes.index(attribute)
        similarities, exact, frequencies = self.similarities.copy(), self.exact.copy(), self.frequencies.copy()
        similarities[:, index] = other.similarities[:, source]
        exact[:, index] = other.exact[:, source]
        frequencies[:, index] = other.frequencies[:, source]
        metric_names = list(self.metrics)
        metric_names[index] = other.metrics[source]
        structural = list(self.structural)
        structural[index] = other.structural[source]
        return ComparisonMatrix(
            self.attributes, tuple(metric_names), similarities, exact, frequencies, tuple(structural)
        )

    @classmethod
    def concat(cls, parts: Sequence[ComparisonMatrix]) -> ComparisonMatrix:
        first = parts[0]
        return cls(
            first.attributes,
            first.metrics,
            np.concatenate([part.similarities for part in parts]),
            np.concatenate([part.exact for part in parts]),
            np.concatenate([part.frequencies for part in parts]),
            first.structural,
        )


class Comparator(ABC):
    metric = ""
    fixed_levels = 0

    def __init__(self) -> None:
        self.present = np.zeros(0, dtype=bool)
        self.frequency = np.zeros(0, dtype=np.float64)

    def fit(self, values: np.ndarray) -> Comparator:
        values = np.asarray(values, dtype=object)
        self.present = np.array([bool(str(value).strip()) for value in values], dtype=bool)
        keys = self._fit(values)
        self.frequency = _term_frequencies(keys, self.present)
        return self

    def similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        result = np.full(len(left), np.nan, dtype=np.float32)
        both = self.present[left] & self.present[right]
        if both.any():
            result[both] = self._similarity(left[both], right[both])
        return result

    def exact(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        both = self.present[left] & self.present[right]
        result = np.zeros(len(left), dtype=bool)
        if both.any():
            result[both] = self._exact(left[both], right[both])
        return result

    def frequencies(self, exact: np.ndarray, left: np.ndarray) -> np.ndarray:
        return np.where(exact, self.frequency[left], np.nan).astype(np.float32)

    @abstractmethod
    def _fit(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _exact(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TextComparator(Comparator):
    def __init__(self) -> None:
        super().__init__()
        self.keys = np.zeros(0, dtype=object)

    def _fit(self, values: np.ndarray) -> np.ndarray:
        self.keys = comparison_keys(values)
        self.present &= self.keys != ""
        self._prepare()
        return self.keys

    def _prepare(self) -> None:
        return None

    def _exact(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return self.keys[left] == self.keys[right]


class JaroWinklerComparator(TextComparator):
    metric = "jaro_winkler"

    def _similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return metrics.jaro_winkler(self.keys[left], self.keys[right])


class LevenshteinComparator(TextComparator):
    metric = "levenshtein"

    def _similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return metrics.levenshtein(self.keys[left], self.keys[right])


class TokenSetComparator(TextComparator):
    metric = "token_set"

    def _similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return metrics.token_set(self.keys[left], self.keys[right])


class TfIdfComparator(TextComparator):
    metric = "tfidf"
    analyzer = "word"
    ngram_range = (1, 1)

    def _prepare(self) -> None:
        vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            token_pattern=TOKEN_PATTERN if self.analyzer == "word" else None,
            lowercase=False,
            sublinear_tf=True,
            dtype=np.float32,
        )
        try:
            self.matrix = vectorizer.fit_transform(self.keys).tocsr()
        except ValueError:
            self.matrix = sparse.csr_matrix((len(self.keys), 1), dtype=np.float32)

    def _similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return metrics.cosine_rows(self.matrix, left, right)


class CharTfIdfComparator(TfIdfComparator):
    metric = "tfidf_char"
    analyzer = "char_wb"
    ngram_range = (3, 3)


class JaccardComparator(TfIdfComparator):
    metric = "jaccard"

    def _similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return metrics.jaccard_rows(self.matrix, left, right)


class ExactComparator(TextComparator):
    metric = "exact"
    fixed_levels = 2

    def _similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return self._exact(left, right).astype(np.float32)


class UrlComparator(ExactComparator):
    metric = "url"

    def _fit(self, values: np.ndarray) -> np.ndarray:
        self.keys = np.array(
            [normalize_url(str(value)) or str(value).strip().lower() for value in values], dtype=object
        )
        self.present &= self.keys != ""
        return self.keys


class PhoneComparator(Comparator):
    metric = "phone"
    fixed_levels = 4
    DIFFERENT, ONE_EDIT, SUFFIX, SAME = range(4)

    def __init__(self) -> None:
        super().__init__()
        self.digits = np.zeros(0, dtype=object)
        self.suffix_digits = MIN_DIGITS

    def _fit(self, values: np.ndarray) -> np.ndarray:
        self.digits = np.array([NON_DIGITS.sub("", str(value)) for value in values], dtype=object)
        self.present &= np.array([len(digits) >= MIN_DIGITS for digits in self.digits], dtype=bool)
        records = int(self.present.sum())
        self.suffix_digits = max(MIN_DIGITS, math.floor(math.log10(records)) + 1 if records else MIN_DIGITS)
        return self.digits

    def _similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        a, b = self.digits[left], self.digits[right]
        codes = np.full(len(left), self.DIFFERENT, dtype=np.float32)
        edits = np.fromiter(
            (len(x) == len(y) and Levenshtein.distance(x, y) == 1 for x, y in zip(a, b)), dtype=bool, count=len(a)
        )
        codes[edits] = self.ONE_EDIT
        codes[self._suffix(a, b)] = self.SUFFIX
        codes[a == b] = self.SAME
        return codes / (self.fixed_levels - 1)

    def _exact(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return self.digits[left] == self.digits[right]

    def _suffix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        limit = self.suffix_digits
        return np.fromiter(
            (x != y and min(len(x), len(y)) >= limit and (x.endswith(y) or y.endswith(x)) for x, y in zip(a, b)),
            dtype=bool,
            count=len(a),
        )


class NumericComparator(Comparator):
    metric = "number_difference"
    column_type = ColumnType.NUMBER
    circular: float | None = None

    def __init__(self) -> None:
        super().__init__()
        self.numbers = np.zeros(0, dtype=np.float64)
        self.reference = np.zeros(0, dtype=np.float64)

    def _fit(self, values: np.ndarray) -> np.ndarray:
        numbers = numeric_values(pd.Series(values, dtype=object), self.column_type)
        self.present &= np.isfinite(numbers)
        self.numbers = self._scale(numbers)
        observed = self.numbers[self.present]
        if len(observed) >= 2:
            generator = np.random.default_rng(CALIBRATION_SEED)
            first = generator.integers(0, len(observed), CALIBRATION_PAIRS)
            second = generator.integers(0, len(observed), CALIBRATION_PAIRS)
            distinct = first != second
            self.reference = np.sort(self._difference(observed[first[distinct]], observed[second[distinct]]))
        return np.where(self.present, numbers, np.nan)

    def _similarity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        difference = self._difference(self.numbers[left], self.numbers[right])
        if not len(self.reference):
            return (difference == 0).astype(np.float32)
        closer = np.searchsorted(self.reference, difference, side="right") / len(self.reference)
        return (1.0 - closer).astype(np.float32)

    def _exact(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return self.numbers[left] == self.numbers[right]

    def _scale(self, numbers: np.ndarray) -> np.ndarray:
        return numbers

    def _difference(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        difference = np.abs(left - right)
        if self.circular is not None:
            difference = np.minimum(difference, self.circular - difference)
        return difference


class NumberComparator(NumericComparator):
    def _scale(self, numbers: np.ndarray) -> np.ndarray:
        finite = numbers[np.isfinite(numbers)]
        if len(finite) and (finite > 0).all():
            return np.log(numbers)
        return numbers


class YearComparator(NumericComparator):
    metric = "year_difference"
    column_type = ColumnType.YEAR


class DateComparator(NumericComparator):
    metric = "day_difference"
    column_type = ColumnType.DATE


class TimeComparator(NumericComparator):
    metric = "minute_difference"
    column_type = ColumnType.TIME
    circular = MINUTES_PER_DAY


COMPARATORS: dict[str, Callable[[], Comparator]] = {
    "tfidf_char": CharTfIdfComparator,
    "tfidf": TfIdfComparator,
    "jaro_winkler": JaroWinklerComparator,
    "levenshtein": LevenshteinComparator,
    "token_set": TokenSetComparator,
    "jaccard": JaccardComparator,
    "exact": ExactComparator,
    "phone": PhoneComparator,
    "url": UrlComparator,
    "number_difference": NumberComparator,
    "year_difference": YearComparator,
    "day_difference": DateComparator,
    "minute_difference": TimeComparator,
}
TEXT_METRICS = ("tfidf_char", "tfidf", "jaro_winkler", "levenshtein", "token_set", "jaccard")
CODE_METRICS = ("levenshtein", "exact", "jaro_winkler", "tfidf_char")
CANDIDATE_METRICS: dict[ColumnType, tuple[str, ...]] = {
    ColumnType.TEXT: TEXT_METRICS,
    ColumnType.CATEGORY: TEXT_METRICS,
    ColumnType.CODE: CODE_METRICS,
    ColumnType.IDENTIFIER: CODE_METRICS,
    ColumnType.BOOLEAN: ("exact",),
    ColumnType.EMAIL: ("exact", "levenshtein"),
    ColumnType.URL: ("url",),
    ColumnType.PHONE: ("phone",),
    ColumnType.NUMBER: ("number_difference",),
    ColumnType.MEASURE: ("number_difference",),
    ColumnType.YEAR: ("year_difference",),
    ColumnType.DATE: ("day_difference",),
    ColumnType.DATETIME: ("day_difference",),
    ColumnType.TIME: ("minute_difference",),
}


def candidate_metrics(column_type: ColumnType) -> tuple[str, ...]:
    return CANDIDATE_METRICS.get(column_type, TEXT_METRICS)


def comparator_for(column_type: ColumnType, metric: str | None = None) -> Comparator:
    name = metric if metric in COMPARATORS else candidate_metrics(column_type)[0]
    return COMPARATORS[name]()


class Comparison:
    def __init__(self, comparators: Mapping[str, Comparator]) -> None:
        self.comparators = dict(comparators)

    @classmethod
    def for_records(
        cls,
        records: RecordSet,
        scope: PairScope,
        metrics_by_attribute: Mapping[str, str] | None = None,
        attributes: Sequence[str] | None = None,
    ) -> Comparison:
        chosen = list(attributes) if attributes is not None else comparable_attributes(records, scope)
        overrides = metrics_by_attribute or {}
        comparators = {
            attribute: comparator_for(records.types[attribute], overrides.get(attribute)) for attribute in chosen
        }
        return cls(comparators).fit(records)

    @property
    def attributes(self) -> tuple[str, ...]:
        return tuple(self.comparators)

    @property
    def metrics(self) -> dict[str, str]:
        return {attribute: comparator.metric for attribute, comparator in self.comparators.items()}

    def fit(self, records: RecordSet) -> Comparison:
        for attribute, comparator in self.comparators.items():
            comparator.fit(records.values(attribute))
        return self

    def compare(self, pairs: np.ndarray) -> ComparisonMatrix:
        shape = (len(pairs), len(self.comparators))
        similarities = np.zeros(shape, dtype=np.float32)
        exact = np.zeros(shape, dtype=bool)
        frequencies = np.full(shape, np.nan, dtype=np.float32)
        for start in range(0, len(pairs), COMPARISON_BATCH):
            batch = slice(start, start + COMPARISON_BATCH)
            left, right = pairs[batch, 0], pairs[batch, 1]
            for index, comparator in enumerate(self.comparators.values()):
                similarities[batch, index] = comparator.similarity(left, right)
                exact[batch, index] = comparator.exact(left, right)
                frequencies[batch, index] = comparator.frequencies(exact[batch, index], left)
        return ComparisonMatrix(
            self.attributes,
            tuple(comparator.metric for comparator in self.comparators.values()),
            similarities,
            exact,
            frequencies,
            tuple(comparator.fixed_levels for comparator in self.comparators.values()),
        )


def comparable_attributes(records: RecordSet, scope: PairScope) -> list[str]:
    chosen = []
    codes = records.source_codes
    for attribute in records.attributes:
        column_type = records.types.get(attribute, ColumnType.TEXT)
        if column_type is ColumnType.EMPTY:
            continue
        keys = comparison_keys(records.values(attribute))
        present = keys != ""
        if present.sum() < 2:
            continue
        if scope is PairScope.CROSS and len(np.unique(codes[present])) < 2:
            continue
        if len(records.source_names) > 1 and _constant_per_source(keys, codes, present):
            continue
        if column_type is ColumnType.IDENTIFIER and not _shared_across_sources(keys, codes, present):
            continue
        chosen.append(attribute)
    return chosen


def _term_frequencies(keys: np.ndarray, present: np.ndarray) -> np.ndarray:
    frequency = np.full(len(keys), np.nan, dtype=np.float64)
    if not present.any():
        return frequency
    observed = pd.Series(keys[present])
    shares = observed.map(observed.value_counts(normalize=True)).to_numpy(dtype=np.float64)
    frequency[present] = shares
    return frequency


def _constant_per_source(keys: np.ndarray, codes: np.ndarray, present: np.ndarray) -> bool:
    return all(len(set(keys[present & (codes == code)])) <= 1 for code in np.unique(codes[present]))


def _shared_across_sources(keys: np.ndarray, codes: np.ndarray, present: np.ndarray) -> bool:
    owners: dict[str, int] = {}
    return any(owners.setdefault(key, code) != code for key, code in zip(keys[present], codes[present]))
