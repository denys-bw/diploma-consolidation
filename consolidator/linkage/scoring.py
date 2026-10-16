from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

import numpy as np

from consolidator.linkage.clustering import NOT_REVIEWED, OneToOneAssignment
from consolidator.linkage.comparison import ComparisonMatrix
from consolidator.linkage.features import MetricValues, MetricView
from consolidator.linkage.fellegi_sunter import (
    FellegiSunterModel,
    OneToOneConstraint,
    estimate_group_priors,
    fit_fellegi_sunter,
    fit_non_matches,
    frequency_adjustment,
    pair_priors,
    posterior,
    weight_from_probability,
)
from consolidator.linkage.policy import ACROSS, WITHIN, mixed_groups
from consolidator.linkage.records import RecordSet
from consolidator.linkage.reliability import EVENTS_PER_PARAMETER
from consolidator.statistics import MAJORITY, PRIOR_SCALE, BayesianLogistic

ROUND_LIMIT = 50
FIT_SAMPLE = 1_000_000
PROBABILITY_LIMIT = 1e-12
SEED = 0
LN2 = float(np.log(2.0))


@dataclass(frozen=True, eq=False)
class PairScores:
    model: FellegiSunterModel
    weights: np.ndarray
    groups: np.ndarray
    group_priors: tuple[float, ...]
    rounds: int = 0
    one_to_one: bool = False
    opinion: PairScores | None = None
    corrected: bool = False
    inside: FellegiSunterModel | None = None

    @property
    def priors(self) -> np.ndarray:
        return pair_priors(self.groups, self.group_priors)

    @property
    def prior(self) -> float:
        return float(self.priors.mean()) if len(self.groups) else self.model.prior

    @property
    def probabilities(self) -> np.ndarray:
        return posterior(self.weights, self.priors)

    @property
    def independent(self) -> np.ndarray:
        return self.opinion.probabilities if self.opinion is not None else self.probabilities

    def attribute_weights(self, levels: np.ndarray) -> np.ndarray:
        weights = self.model.attribute_weights(levels)
        if self.inside is not None:
            within = self.groups == WITHIN
            weights[within] = self.inside.attribute_weights(levels[within])
        return weights


@dataclass(frozen=True, eq=False)
class ScoringTask:
    records: RecordSet
    pairs: np.ndarray
    comparison: ComparisonMatrix
    prior: float
    groups: np.ndarray | None = None
    closed: frozenset[int] = field(default_factory=frozenset)
    reviewed: np.ndarray | None = None
    constraint: OneToOneConstraint | None = None
    values: MetricValues | None = None

    @property
    def known(self) -> np.ndarray:
        return self.reviewed if self.reviewed is not None else np.full(len(self.pairs), NOT_REVIEWED, dtype=np.int8)

    @property
    def pair_groups(self) -> np.ndarray:
        return self.groups if self.groups is not None else np.zeros(len(self.pairs), dtype=np.int64)

    def metric_values(self) -> MetricValues | None:
        if self.values is not None:
            return self.values
        view = MetricView.for_records(self.records, self.comparison.attributes)
        if view.empty or not view.affordable(len(self.pairs)):
            return None
        return view.compute(self.pairs)


class PairScorer(ABC):
    key = ""

    def score(self, task: ScoringTask) -> PairScores:
        scores = self.unsupervised(task)
        if task.constraint is not None:
            scores = assignment_priors(task, scores)
        return ReviewCorrection().apply(task, scores)

    @abstractmethod
    def unsupervised(self, task: ScoringTask) -> PairScores:
        raise NotImplementedError


class FellegiSunterScorer(PairScorer):
    key = "fellegi_sunter"

    def __init__(self, term_frequencies: bool = True) -> None:
        self.term_frequencies = term_frequencies

    def unsupervised(self, task: ScoringTask) -> PairScores:
        scores = self._fit(task, None)
        constraint = task.constraint
        if constraint is None or not constraint.violated(scores.probabilities):
            return scores
        return replace(self._fit(task, constraint), one_to_one=True)

    def _fit(self, task: ScoringTask, constraint: OneToOneConstraint | None) -> PairScores:
        comparison = task.comparison
        levels = comparison.levels
        groups = task.pair_groups
        mixed = mixed_groups(groups, task.closed)
        fitted = groups == ACROSS if mixed else np.ones(len(groups), dtype=bool)
        model = fit_fellegi_sunter(
            levels[fitted],
            comparison.n_levels,
            comparison.attributes,
            prior=task.prior,
            groups=groups[fitted],
            closed=task.closed,
            constraint=constraint,
        )
        initial = list(model.priors)
        inside = None
        if mixed:
            inside = fit_non_matches(levels[~fitted], comparison.n_levels, comparison.attributes, model.m, task.prior)
            initial[WITHIN] = inside.prior
        layout = PairScores(model, np.zeros(len(groups)), groups, tuple(initial), inside=inside)
        weights = layout.attribute_weights(levels).sum(axis=1)
        if self.term_frequencies:
            weights = weights + frequency_adjustment(comparison.frequencies)
        priors = estimate_group_priors(weights, groups, initial, task.closed)
        return replace(layout, weights=weights, group_priors=priors)


class CoTraining(PairScorer):
    key = "co_training"

    def __init__(self, prior_scale: float = PRIOR_SCALE) -> None:
        self.prior_scale = prior_scale

    def unsupervised(self, task: ScoringTask) -> PairScores:
        base = FellegiSunterScorer().unsupervised(task)
        values = task.metric_values()
        if values is None:
            return base
        features = values.features()
        current = base
        decisions = base.probabilities >= MAJORITY
        for step in range(1, ROUND_LIMIT + 1):
            targets = current.probabilities
            if targets.sum() < 1.0 or (1.0 - targets).sum() < 1.0:
                break
            weights = (base.weights + _log_ratio(features, targets, self.prior_scale)) / 2.0
            priors = estimate_group_priors(weights, base.groups, current.group_priors, task.closed)
            current = replace(base, weights=weights, group_priors=priors, rounds=step, opinion=base)
            updated = current.probabilities >= MAJORITY
            if np.array_equal(updated, decisions):
                break
            decisions = updated
        return current


class ReviewCorrection:
    def apply(self, task: ScoringTask, scores: PairScores) -> PairScores:
        known = task.known
        labelled = np.flatnonzero(known != NOT_REVIEWED)
        if len(labelled) == 0:
            return scores
        targets = known[labelled].astype(np.float64)
        if not 0.0 < targets.sum() < len(targets):
            return scores
        offset = log_odds(scores.probabilities)
        features = review_features(task, offset)
        model = BayesianLogistic().fit(features[labelled], targets, offset=offset[labelled], reference=features)
        logits = model.decision_function(features, offset)
        open_pairs = ~np.isin(scores.groups, list(task.closed))
        weights = scores.weights.copy()
        weights[open_pairs] = logits[open_pairs] / LN2 - weight_from_probability(scores.priors[open_pairs])
        return replace(scores, weights=weights, corrected=True, opinion=scores.opinion or scores)


def assignment_priors(task: ScoringTask, scores: PairScores) -> PairScores:
    order = np.lexsort((-scores.weights, -scores.probabilities))
    _, assigned = OneToOneAssignment().assign(task.records, task.pairs, order)
    groups = assigned.astype(np.int64)
    regrouped = _regrouped(scores, groups)
    chosen = regrouped.probabilities[assigned]
    if min(chosen.sum(), (1.0 - chosen).sum()) < EVENTS_PER_PARAMETER:
        return scores
    opinion = _regrouped(scores.opinion, groups) if scores.opinion is not None else None
    return replace(regrouped, opinion=opinion)


def _regrouped(scores: PairScores, groups: np.ndarray) -> PairScores:
    probabilities = scores.probabilities
    initial = [
        float(probabilities[groups == group].mean()) if (groups == group).any() else MAJORITY for group in (0, 1)
    ]
    return replace(scores, groups=groups, group_priors=estimate_group_priors(scores.weights, groups, initial))


def review_features(task: ScoringTask, offset: np.ndarray) -> np.ndarray:
    comparison = task.comparison
    best = mutual_best(task.pairs, offset, task.records.source_codes).astype(np.float64)
    parts = [offset[:, None], best[:, None], np.nan_to_num(comparison.similarities.astype(np.float64), nan=0.0)]
    parts.append((~np.isnan(comparison.similarities)).astype(np.float64))
    values = task.metric_values()
    if values is not None:
        parts.append(values.features().astype(np.float64))
    return np.column_stack(parts)


def mutual_best(pairs: np.ndarray, scores: np.ndarray, sources: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.zeros(0, dtype=bool)
    width = int(sources.max()) + 1
    index = np.arange(len(pairs))
    keys = np.concatenate(
        [pairs[:, 0] * width + sources[pairs[:, 1]], pairs[:, 1] * width + sources[pairs[:, 0]]]
    ).astype(np.int64)
    owners = np.concatenate([index, index])
    order = np.lexsort((-np.concatenate([scores, scores]), keys))
    first = np.ones(len(order), dtype=bool)
    first[1:] = keys[order][1:] != keys[order][:-1]
    best = np.zeros(len(pairs), dtype=np.int64)
    np.add.at(best, owners[order][first], 1)
    return best == 2


def log_odds(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, PROBABILITY_LIMIT, 1.0 - PROBABILITY_LIMIT)
    return np.log(clipped / (1.0 - clipped))


def _log_ratio(features: np.ndarray, targets: np.ndarray, prior_scale: float) -> np.ndarray:
    rows = np.arange(len(targets))
    if len(rows) > FIT_SAMPLE:
        rows = np.sort(np.random.default_rng(SEED).choice(rows, FIT_SAMPLE, replace=False))
    prevalence = float(np.clip(targets[rows].mean(), PROBABILITY_LIMIT, 1.0 - PROBABILITY_LIMIT))
    baseline = np.full(len(rows), log_odds(np.array([prevalence]))[0])
    model = BayesianLogistic(prior_scale).fit(features[rows], targets[rows], offset=baseline)
    return model.decision_function(features) / LN2


SCORERS: dict[str, type[PairScorer]] = {scorer.key: scorer for scorer in (CoTraining, FellegiSunterScorer)}
