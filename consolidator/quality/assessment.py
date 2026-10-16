from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from consolidator.quality.ahp import AhpResult, analyse, pairwise_matrix
from consolidator.quality.criteria import CRITERIA, CRITERION_KEYS, DataSnapshot, QualityCriterion

DEFAULT_JUDGEMENTS = {
    ("uniqueness", "completeness"): 3.0,
    ("completeness", "consistency"): 3.0,
    ("uniqueness", "consistency"): 5.0,
}


class CriterionValues(Mapping[str, float]):
    def __init__(self, values: Mapping[str, float]) -> None:
        self._values = {criterion: float(value) for criterion, value in values.items()}

    def __getitem__(self, criterion: str) -> float:
        return self._values[criterion]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._values!r})"


class QualityWeights(CriterionValues):
    @classmethod
    def from_ahp(cls, result: AhpResult) -> QualityWeights:
        return cls(result.as_dict())

    @classmethod
    def default(cls) -> QualityWeights:
        return cls.from_ahp(default_ahp())

    @classmethod
    def equal(cls, criteria: Sequence[str] = CRITERION_KEYS) -> QualityWeights:
        return cls(dict.fromkeys(criteria, 1.0 / len(criteria)))


class QualityScores(CriterionValues):
    def index(self, weights: Mapping[str, float]) -> float:
        total = sum(weights[criterion] for criterion in self)
        value = sum(weights[criterion] * score for criterion, score in self.items())
        return value / total if total else 0.0

    def replaced(self, criterion: str, value: float) -> QualityScores:
        return QualityScores({**self, criterion: value})


def default_ahp() -> AhpResult:
    return analyse(CRITERION_KEYS, pairwise_matrix(CRITERION_KEYS, DEFAULT_JUDGEMENTS))


def assess(snapshot: DataSnapshot, criteria: Sequence[QualityCriterion] = CRITERIA) -> QualityScores:
    return QualityScores({criterion.key: criterion.measure(snapshot) for criterion in criteria})
