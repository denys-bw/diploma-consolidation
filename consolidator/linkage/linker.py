from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from consolidator.linkage.blocking import Blocker, TokenBlocking
from consolidator.linkage.clustering import (
    NOT_REVIEWED,
    AverageLinkage,
    Clusterer,
    ConnectedComponents,
    OneToOneAssignment,
    ScoredPairs,
    with_review,
)
from consolidator.linkage.comparison import Comparison, ComparisonMatrix
from consolidator.linkage.decision import ClericalReview, DecisionRule, MatchStatus, locate
from consolidator.linkage.features import MetricValues, MetricView
from consolidator.linkage.fellegi_sunter import FellegiSunterModel, OneToOneConstraint
from consolidator.linkage.pairs import encode, union
from consolidator.linkage.policy import ACROSS, DuplicatePolicy, SourceStructure, pair_groups
from consolidator.linkage.records import PairScope, RecordSet
from consolidator.linkage.reliability import EVENTS_PER_PARAMETER, IDENTIFIABLE_ATTRIBUTES, Reliability, assess
from consolidator.linkage.scoring import (
    CoTraining,
    FellegiSunterScorer,
    PairScorer,
    PairScores,
    ScoringTask,
    mutual_best,
)
from consolidator.linkage.selection import MetricChoice, select_metrics


@dataclass(frozen=True)
class LinkageSettings:
    policy: DuplicatePolicy = DuplicatePolicy.AUTO
    blocker: Blocker = field(default_factory=TokenBlocking)
    decision: DecisionRule = field(default_factory=DecisionRule)
    scorer: PairScorer = field(default_factory=CoTraining)
    clusterer: Clusterer | None = None
    attributes: tuple[str, ...] | None = None
    metrics: Mapping[str, str] = field(default_factory=dict)
    select_metrics: bool = True


@dataclass(frozen=True, eq=False)
class LinkageResult:
    records: RecordSet
    settings: LinkageSettings
    structure: SourceStructure
    blocker: Blocker
    clusterer: Clusterer
    candidates: np.ndarray
    comparison: ComparisonMatrix
    metric_choices: dict[str, MetricChoice]
    model: FellegiSunterModel | None
    prior: float
    training_rounds: int
    one_to_one: bool
    corrected: bool
    weights: np.ndarray
    probabilities: np.ndarray
    independent: np.ndarray
    statuses: np.ndarray
    reviewed: np.ndarray
    accepted: np.ndarray
    labels: np.ndarray
    timings: dict[str, float]

    @property
    def scope(self) -> PairScope:
        return self.structure.scope

    @property
    def few_attributes(self) -> bool:
        return len(self.comparison.attributes) < IDENTIFIABLE_ATTRIBUTES

    @property
    def matches(self) -> np.ndarray:
        return self.candidates[self.accepted]

    @property
    def entities(self) -> int:
        return len(np.unique(self.labels))

    @property
    def effective_probabilities(self) -> np.ndarray:
        return with_review(self.probabilities, self.reviewed)

    @property
    def contested(self) -> np.ndarray:
        decision = self.settings.decision
        return decision.accepted(self.probabilities) != decision.accepted(self.independent)

    @property
    def weight_thresholds(self) -> tuple[float, float]:
        return self.settings.decision.weight_thresholds(self.effective_probabilities, self.statuses, self.prior)

    def count(self, status: MatchStatus) -> int:
        return int((self.statuses == status).sum())

    def review_queue(self, extended: bool = False) -> np.ndarray:
        open_pairs = self.reviewed == NOT_REVIEWED
        answered = np.flatnonzero(~open_pairs)
        design = EVENTS_PER_PARAMETER * (len(self.comparison.attributes) + 1) - len(answered)
        if self.few_attributes and design > 0:
            queue = _spread(self.weights, np.flatnonzero(open_pairs), design)
            queue = queue if extended else queue[:design]
            return np.concatenate([answered, queue])
        threshold = self.settings.decision.match_probability
        possible = (self.statuses == MatchStatus.POSSIBLE) & open_pairs
        best = mutual_best(self.candidates, self.probabilities, self.records.source_codes)
        queue = _interleave(
            _alternate(self.probabilities, np.flatnonzero(possible & best), threshold),
            _alternate(self.probabilities, np.flatnonzero(possible & ~best), threshold),
        )
        if not extended:
            return np.concatenate([answered, queue])
        others = np.flatnonzero(open_pairs & ~possible)
        errors = self.settings.decision.errors(self.probabilities[others], self.contested[others])
        ranked = others[np.lexsort((others, -errors))]
        return np.concatenate([answered, queue, ranked])

    def reliability(self) -> Reliability:
        return assess(
            self.effective_probabilities,
            with_review(self.independent, self.reviewed),
            self.statuses,
            self.settings.decision,
            self.training_rounds,
            self.one_to_one,
            len(self.comparison.attributes),
            self.corrected,
        )

    def expected_false_matches(self) -> float:
        return float((1.0 - self.effective_probabilities[self.accepted]).sum())

    def expected_missed_matches(self) -> float:
        return float(self.effective_probabilities[~self.accepted].sum())


class CandidateJudge:
    def __init__(self, records: RecordSet, comparison: Comparison, scope: PairScope) -> None:
        self.records = records
        self.comparison = comparison
        self.scope = scope
        self.codes = np.zeros(0, dtype=np.int64)
        self.parts: list[ComparisonMatrix] = []

    def __call__(self, pairs: np.ndarray) -> np.ndarray:
        if len(pairs) == 0:
            return np.zeros(0)
        vectors = self.vectors(pairs)
        groups = _groups(self.records, pairs, self.scope)
        task = ScoringTask(self.records, pairs, vectors, _initial_prior(self.records, self.scope, len(pairs)), groups)
        return FellegiSunterScorer().unsupervised(task).probabilities

    def vectors(self, pairs: np.ndarray) -> ComparisonMatrix:
        if len(pairs) == 0:
            return self.comparison.compare(pairs)
        codes = encode(pairs, self.records.size)
        known = _positions(self.codes, codes)
        fresh = known < 0
        if fresh.any():
            self.parts.append(self.comparison.compare(pairs[fresh]))
            self.codes = np.concatenate([self.codes, codes[fresh]])
            known = _positions(self.codes, codes)
        return ComparisonMatrix.concat(self.parts).rows(known)


class Linker:
    def __init__(self, settings: LinkageSettings | None = None) -> None:
        self.settings = settings or LinkageSettings()

    def run(self, records: RecordSet, review: ClericalReview | None = None) -> LinkageResult:
        settings = self.settings
        reviewed_pairs, confirmed = (review or ClericalReview()).on(records)
        declared = SourceStructure.declared(settings.policy, len(records.source_names))
        scope = declared.scope if declared is not None else PairScope.ALL
        timings: dict[str, float] = {}
        engine = Comparison.for_records(records, scope, settings.metrics, settings.attributes)
        judge = CandidateJudge(records, engine, scope)
        with _timed(timings, "blocking"):
            blocker = settings.blocker.resolve(records, scope)
            if blocker.adaptive and engine.attributes:
                candidates, _ = blocker.adaptive_candidates(records, scope, judge)
            else:
                candidates = blocker.candidates(records, scope)
            candidates = union(candidates, reviewed_pairs[confirmed])
        with _timed(timings, "comparison"):
            usable = bool(engine.attributes) and len(candidates) > 0
            vectors = judge.vectors(candidates) if usable else engine.compare(candidates)
            values = self._values(records, engine.attributes, candidates) if usable else None
            choices: dict[str, MetricChoice] = {}
            if usable and values is not None and settings.select_metrics:
                vectors, choices = self._select(records, scope, candidates, vectors, values)
        with _timed(timings, "classification"):
            reviewed = np.full(len(candidates), NOT_REVIEWED, dtype=np.int8)
            located = locate(candidates, reviewed_pairs, records.size)
            reviewed[located[located >= 0]] = confirmed[located >= 0]
            structure = declared
            if structure is None:
                structure = self._detect(records, candidates, vectors, reviewed, usable)
                if structure.scope is PairScope.CROSS:
                    keep = pair_groups(records, candidates) == ACROSS
                    candidates, vectors, reviewed = candidates[keep], vectors.rows(keep), reviewed[keep]
                    values = values.rows(keep) if values is not None else None
            scores = self._score(records, candidates, vectors, values, reviewed, structure) if usable else None
            weights = scores.weights if scores else np.zeros(len(candidates))
            probabilities = scores.probabilities if scores else np.zeros(len(candidates))
            independent = scores.independent if scores else probabilities
            decision = settings.decision
            contested = decision.accepted(probabilities) != decision.accepted(independent)
            statuses = decision.statuses(with_review(probabilities, reviewed), contested)
        with _timed(timings, "clustering"):
            clusterer = self._clusterer(structure) if scores else ConnectedComponents()
            scored = ScoredPairs(candidates, weights, probabilities, reviewed, structure.scope)
            clustering = clusterer.cluster(records, scored, decision.match_probability)
        return LinkageResult(
            records=records,
            settings=settings,
            structure=structure,
            blocker=blocker,
            clusterer=clusterer,
            candidates=candidates,
            comparison=vectors,
            metric_choices=choices,
            model=scores.model if scores else None,
            prior=scores.prior if scores else 0.0,
            training_rounds=scores.rounds if scores else 0,
            one_to_one=scores.one_to_one if scores else False,
            corrected=scores.corrected if scores else False,
            weights=weights,
            probabilities=probabilities,
            independent=independent,
            statuses=statuses,
            reviewed=reviewed,
            accepted=clustering.accepted,
            labels=clustering.labels,
            timings=timings,
        )

    def _values(self, records: RecordSet, attributes: tuple[str, ...], candidates: np.ndarray) -> MetricValues | None:
        view = MetricView.for_records(records, attributes)
        if view.empty or not view.affordable(len(candidates)):
            return None
        return view.compute(candidates)

    def _select(
        self,
        records: RecordSet,
        scope: PairScope,
        candidates: np.ndarray,
        vectors: ComparisonMatrix,
        values: MetricValues,
    ) -> tuple[ComparisonMatrix, dict[str, MetricChoice]]:
        groups = _groups(records, candidates, scope)
        task = ScoringTask(records, candidates, vectors, _initial_prior(records, scope, len(candidates)), groups)
        scores = FellegiSunterScorer().unsupervised(task)
        fixed = set(self.settings.metrics)
        choices = select_metrics(vectors, scores.attribute_weights(vectors.levels), scores.priors, values)
        changed = {
            attribute: choice.metric
            for attribute, choice in choices.items()
            if attribute not in fixed and choice.metric != dict(zip(vectors.attributes, vectors.metrics))[attribute]
        }
        if changed:
            replacement = Comparison.for_records(records, scope, changed, list(changed)).compare(candidates)
            for attribute in changed:
                vectors = vectors.replaced(attribute, replacement)
        return vectors, choices

    def _detect(
        self,
        records: RecordSet,
        candidates: np.ndarray,
        vectors: ComparisonMatrix,
        reviewed: np.ndarray,
        usable: bool,
    ) -> SourceStructure:
        groups = pair_groups(records, candidates)
        if not usable:
            return SourceStructure(PairScope.CROSS, True, True)
        prior = _initial_prior(records, PairScope.ALL, len(candidates))
        return SourceStructure.detected(vectors, groups, reviewed, prior)

    def _score(
        self,
        records: RecordSet,
        candidates: np.ndarray,
        vectors: ComparisonMatrix,
        values: MetricValues | None,
        reviewed: np.ndarray,
        structure: SourceStructure,
    ) -> PairScores | None:
        if len(candidates) == 0:
            return None
        constraint = OneToOneConstraint(candidates, records.source_codes, records.size) if structure.clean else None
        task = ScoringTask(
            records=records,
            pairs=candidates,
            comparison=vectors,
            prior=_initial_prior(records, structure.scope, len(candidates)),
            groups=_groups(records, candidates, structure.scope),
            reviewed=reviewed,
            constraint=constraint,
            values=values,
        )
        return self.settings.scorer.score(task)

    def _clusterer(self, structure: SourceStructure) -> Clusterer:
        if self.settings.clusterer is not None:
            return self.settings.clusterer
        return OneToOneAssignment() if structure.clean else AverageLinkage()


def _groups(records: RecordSet, pairs: np.ndarray, scope: PairScope) -> np.ndarray | None:
    if len(records.source_names) > 1 and scope is PairScope.ALL:
        return pair_groups(records, pairs)
    return None


def _positions(codes: np.ndarray, probe: np.ndarray) -> np.ndarray:
    if len(codes) == 0:
        return np.full(len(probe), -1, dtype=np.int64)
    order = np.argsort(codes, kind="stable")
    ordered = codes[order]
    positions = np.minimum(np.searchsorted(ordered, probe), len(ordered) - 1)
    return np.where(ordered[positions] == probe, order[positions], -1)


def _alternate(probabilities: np.ndarray, indices: np.ndarray, threshold: float) -> np.ndarray:
    above = indices[probabilities[indices] >= threshold]
    below = indices[probabilities[indices] < threshold]
    above = above[np.argsort(probabilities[above], kind="stable")]
    below = below[np.argsort(-probabilities[below], kind="stable")]
    merged = np.empty(len(above) + len(below), dtype=np.int64)
    shared = min(len(above), len(below))
    merged[0 : 2 * shared : 2] = above[:shared]
    merged[1 : 2 * shared : 2] = below[:shared]
    merged[2 * shared :] = np.concatenate([above[shared:], below[shared:]])
    return merged


def _interleave(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    shared = min(len(first), len(second))
    merged = np.empty(len(first) + len(second), dtype=np.int64)
    merged[0 : 2 * shared : 2] = first[:shared]
    merged[1 : 2 * shared : 2] = second[:shared]
    merged[2 * shared :] = np.concatenate([first[shared:], second[shared:]])
    return merged


def _spread(weights: np.ndarray, indices: np.ndarray, count: int) -> np.ndarray:
    ranked = indices[np.lexsort((indices, -weights[indices]))]
    if len(ranked) < 3:
        return ranked
    values = weights[ranked]
    design = min(count, len(ranked))
    targets = values[0] - _bit_reversal(design) / max(design, 1) * (values[0] - values[-1])
    taken = np.zeros(len(ranked), dtype=bool)
    chosen = []
    for target in targets:
        position = _nearest_free(values, taken, target)
        taken[position] = True
        chosen.append(position)
    order = _bit_reversal(len(ranked))
    rest = order[~taken[order]]
    return ranked[np.concatenate([np.array(chosen, dtype=np.int64), rest])]


def _bit_reversal(count: int) -> np.ndarray:
    if count <= 1:
        return np.zeros(count, dtype=np.int64)
    bits = int(np.ceil(np.log2(count)))
    positions = np.arange(1 << bits)
    reversed_bits = np.zeros_like(positions)
    for bit in range(bits):
        reversed_bits |= ((positions >> bit) & 1) << (bits - 1 - bit)
    return reversed_bits[reversed_bits < count]


def _nearest_free(descending: np.ndarray, taken: np.ndarray, target: float) -> int:
    start = int(np.clip(np.searchsorted(-descending, -target), 0, len(descending) - 1))
    for step in range(len(descending)):
        for position in (start - step, start + step):
            if 0 <= position < len(descending) and not taken[position]:
                return position
    return start


@contextmanager
def _timed(timings: dict[str, float], stage: str) -> Iterator[None]:
    started = time.perf_counter()
    yield
    timings[stage] = time.perf_counter() - started


def _initial_prior(records: RecordSet, scope: PairScope, candidates: int) -> float:
    if candidates == 0:
        return 0.5
    expected = int(np.sort(records.source_sizes)[:-1].sum()) if scope is PairScope.CROSS else records.size // 2
    return float(np.clip(expected / candidates, 1.0 / candidates, 0.5))
