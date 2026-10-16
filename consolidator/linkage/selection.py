from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from consolidator.linkage.comparison import ComparisonMatrix
from consolidator.linkage.features import MetricValues
from consolidator.linkage.fellegi_sunter import frequency_adjustments, posterior
from consolidator.statistics import weighted_auc


@dataclass(frozen=True)
class MetricChoice:
    attribute: str
    metric: str
    scores: dict[str, float]


def select_metrics(
    comparison: ComparisonMatrix,
    weights: np.ndarray,
    priors: np.ndarray,
    values: MetricValues,
) -> dict[str, MetricChoice]:
    current = dict(zip(comparison.attributes, comparison.metrics))
    if len(comparison.attributes) < 2:
        return {attribute: MetricChoice(attribute, metric, {}) for attribute, metric in current.items()}
    contributions = weights + frequency_adjustments(comparison.frequencies)
    total = contributions.sum(axis=1)
    choices = {}
    for index, attribute in enumerate(comparison.attributes):
        candidates = values.metrics(attribute)
        if len(candidates) < 2:
            choices[attribute] = MetricChoice(attribute, current[attribute], {})
            continue
        evidence = posterior(total - contributions[:, index], priors)
        scores = {}
        for metric in candidates:
            similarity = values.column(attribute, metric)
            present = ~np.isnan(similarity)
            scores[metric] = weighted_auc(similarity[present], evidence[present]) if present.any() else 0.5
        best = max(candidates, key=lambda metric: (scores[metric], -candidates.index(metric)))
        choices[attribute] = MetricChoice(attribute, best, scores)
    return choices
