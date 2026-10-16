import numpy as np
import pandas as pd
import pytest

from consolidator.evaluation import (
    average_precision,
    best_threshold,
    error_cells,
    evaluate_blocking,
    evaluate_correction,
    evaluate_detection,
    evaluate_positions,
    gold_labels,
    gold_pairs,
    true_entities,
)
from consolidator.linkage import PairScope


def test_metrics_on_positions_ignore_order_inside_pairs():
    found = np.array([[1, 0], [2, 3], [4, 5]])
    gold = np.array([[0, 1], [3, 2], [6, 7], [7, 8]])
    metrics = evaluate_positions(found, gold, size=9)
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (2, 1, 2)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(4 / 7)
    assert metrics.as_dict()["tp"] == 2
    assert gold_labels(found, gold, 9).tolist() == [True, True, False]


def test_gold_pairs_come_from_entity_labels():
    truth = np.array(["a", "a", "", "b", "b", "a"], dtype=object)
    sources = np.array([0, 1, 1, 0, 0, 1])
    assert gold_pairs(truth, sources, PairScope.ALL).tolist() == [[0, 1], [0, 5], [1, 5], [3, 4]]
    assert gold_pairs(truth, sources, PairScope.CROSS).tolist() == [[0, 1], [0, 5]]
    assert true_entities(truth) == 3


def test_blocking_metrics():
    metrics = evaluate_blocking(np.array([[0, 1], [0, 2], [1, 2]]), np.array([[0, 1], [3, 4]]), size=5, total_pairs=10)
    assert metrics.reduction_ratio == pytest.approx(0.7)
    assert metrics.pairs_completeness == pytest.approx(0.5)
    assert metrics.pairs_quality == pytest.approx(1 / 3)


def test_empty_results_do_not_divide_by_zero():
    metrics = evaluate_positions(np.empty((0, 2), dtype=int), np.array([[0, 1]]), size=2)
    assert metrics.precision == 0.0 and metrics.f1 == 0.0


def test_best_threshold_and_average_precision():
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
    labels = np.array([True, True, False, True, False])
    choice = best_threshold(scores, labels)
    assert choice.threshold == pytest.approx(0.3)
    assert choice.f1 == pytest.approx(6 / 7)
    assert 0.0 < average_precision(scores, labels) <= 1.0


def test_cell_errors_detection_and_correction():
    dirty = pd.DataFrame({"a": ["1", "x", "3"], "b": ["N/A", "5", "6"]})
    clean = pd.DataFrame({"a": ["1", "2", "3"], "b": ["", "5", "7"]})
    errors = error_cells(dirty, clean)
    assert errors == {(1, "a"), (0, "b"), (2, "b")}
    detection = evaluate_detection({(1, "a"), (0, "a")}, errors)
    assert (detection.true_positives, detection.false_positives) == (1, 1)
    correction = evaluate_correction({(1, "a"): "2", (0, "b"): "", (2, "b"): "8", (2, "a"): "3"}, dirty, clean)
    assert (correction.true_positives, correction.false_positives, correction.false_negatives) == (2, 1, 1)
