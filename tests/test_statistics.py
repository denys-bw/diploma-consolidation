import numpy as np
import pytest
from scipy import stats

from consolidator.statistics import (
    MAJORITY,
    BayesianLogistic,
    holm,
    is_majority,
    signed_rank_greater,
    sturges_bins,
    weighted_auc,
)


def test_majority_needs_more_than_half():
    assert is_majority(0.51)
    assert not is_majority(MAJORITY)


def test_sturges_rule():
    assert sturges_bins(0) == 1
    assert sturges_bins(1) == 1
    assert sturges_bins(2) == 2
    assert sturges_bins(100) == 8
    assert sturges_bins(1024) == 11


def test_holm_stops_at_the_first_hypothesis_it_cannot_reject():
    p_values = np.array([0.01, 0.04, 0.03, 0.005])
    assert holm(p_values, 0.05).tolist() == [True, False, False, True]
    assert holm(np.array([0.001, 0.002]), 0.05).tolist() == [True, True]
    assert holm(np.array([]), 0.05).tolist() == []


def test_signed_rank_test_is_one_sided():
    generator = np.random.default_rng(0)
    shifted = generator.normal(0.5, 1.0, 60)
    test = signed_rank_greater(shifted)
    assert test.pairs == 60
    assert test.p_value == pytest.approx(stats.wilcoxon(shifted, alternative="greater").pvalue)
    assert test.p_value < 0.01
    assert signed_rank_greater(-shifted).p_value > 0.99


def test_signed_rank_test_ignores_ties_and_missing_values():
    test = signed_rank_greater(np.array([0.0, np.nan, 0.0]))
    assert test.pairs == 0 and test.p_value == 1.0


def test_weighted_auc_matches_the_ordinary_auc_for_hard_labels():
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.8])
    labels = np.array([0, 0, 1, 1, 0])
    pairs = [(p, n) for p in scores[labels == 1] for n in scores[labels == 0]]
    expected = np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs])
    assert weighted_auc(scores, labels) == pytest.approx(expected)


def test_weighted_auc_without_one_class_is_uninformative():
    assert weighted_auc(np.array([0.1, 0.9]), np.array([1.0, 1.0])) == 0.5
    assert weighted_auc(np.array([0.1, 0.9]), np.array([0.0, 0.0])) == 0.5


def test_weighted_auc_uses_soft_labels():
    scores = np.array([0.0, 1.0])
    positive = np.array([0.2, 0.8])
    negative = 1.0 - positive
    wins = positive[1] * negative[0] + 0.5 * (positive * negative).sum()
    assert weighted_auc(scores, positive) == pytest.approx(wins / (positive.sum() * negative.sum()))


def test_bayesian_logistic_learns_the_direction_of_an_effect():
    generator = np.random.default_rng(1)
    features = generator.normal(size=(400, 2))
    targets = (generator.random(400) < 1 / (1 + np.exp(-2 * features[:, 0]))).astype(float)
    model = BayesianLogistic().fit(features, targets)
    assert model.coefficients[0] > 0.5
    assert abs(model.coefficients[1]) < abs(model.coefficients[0])
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (400,) and np.all((probabilities > 0) & (probabilities < 1))


def test_bayesian_logistic_prior_keeps_estimates_finite_under_separation():
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    targets = np.array([0.0, 0.0, 1.0, 1.0])
    model = BayesianLogistic().fit(features, targets)
    assert np.isfinite(model.coefficients).all() and model.coefficients[0] > 0


def test_bayesian_logistic_with_an_offset_and_no_features_only_shifts():
    offset = np.array([2.0, 1.0, 0.0, -1.0])
    model = BayesianLogistic().fit(np.zeros((4, 0)), np.zeros(4), offset=offset)
    assert model.coefficients.shape == (0,)
    assert model.intercept < 0
    shifted = model.decision_function(np.zeros((4, 0)), offset)
    assert np.allclose(shifted - offset, model.intercept)
