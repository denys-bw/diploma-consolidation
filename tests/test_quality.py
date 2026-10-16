import numpy as np
import pandas as pd
import pytest

from consolidator.profiling import ColumnType
from consolidator.quality import (
    CRITERION_KEYS,
    Completeness,
    Consistency,
    DataSnapshot,
    QualityScores,
    QualityWeights,
    RecordCompleteness,
    Uniqueness,
    analyse,
    assess,
    default_ahp,
    pairwise_matrix,
)
from consolidator.standardization import CellStatus


def snapshot(frames=(), statuses=(), types=None, records=0, entities=0):
    return DataSnapshot(list(frames), list(statuses), types or {}, records, entities)


def test_default_judgements_give_published_weights():
    result = default_ahp()
    assert result.as_dict() == pytest.approx(
        {"completeness": 0.2583, "uniqueness": 0.6370, "consistency": 0.1047}, abs=5e-5
    )
    assert result.lambda_max == pytest.approx(3.0385, abs=5e-5)
    assert result.consistency_index == pytest.approx(0.0193, abs=5e-5)
    assert result.consistency_ratio == pytest.approx(0.0332, abs=5e-5)
    assert result.consistent


def test_pairwise_matrix_is_reciprocal():
    matrix = pairwise_matrix(CRITERION_KEYS, {("uniqueness", "completeness"): 3.0})
    assert matrix[1, 0] == 3.0
    assert matrix[0, 1] == pytest.approx(1 / 3)
    assert np.allclose(matrix * matrix.T, 1.0)


def test_ahp_rejects_invalid_judgements():
    with pytest.raises(ValueError):
        pairwise_matrix(CRITERION_KEYS, {("uniqueness", "completeness"): 0.0})
    with pytest.raises(ValueError):
        analyse(CRITERION_KEYS, np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    with pytest.raises(ValueError):
        analyse(CRITERION_KEYS, np.ones((2, 2)))


def test_inconsistent_judgements_are_detected():
    judgements = {
        ("completeness", "uniqueness"): 9.0,
        ("uniqueness", "consistency"): 9.0,
        ("consistency", "completeness"): 9.0,
    }
    assert not analyse(CRITERION_KEYS, pairwise_matrix(CRITERION_KEYS, judgements)).consistent


def test_quality_index_is_weighted_by_ahp():
    values = {"completeness": 863 / 864, "uniqueness": 752 / 864, "consistency": 331 / 864}
    scores = QualityScores(values)
    assert scores.index(QualityWeights.default()) == pytest.approx(0.8525, abs=5e-5)
    assert scores.index(QualityWeights.equal()) == pytest.approx(np.mean(list(values.values())))
    assert scores.replaced("uniqueness", 1.0)["uniqueness"] == 1.0
    assert dict(scores) == pytest.approx(values)


def test_completeness_counts_cells_or_whole_records():
    frame = pd.DataFrame({"name": ["a", "", "c", "n/a"], "phone": ["1", "2", "", "4"]})
    assert Completeness().measure(snapshot([frame])) == pytest.approx(5 / 8)
    assert RecordCompleteness().measure(snapshot([frame])) == pytest.approx(1 / 4)
    assert Completeness().measure(snapshot()) == 1.0


def test_uniqueness_is_share_of_distinct_entities():
    assert Uniqueness().measure(snapshot(records=864, entities=752)) == pytest.approx(0.8704, abs=5e-5)
    assert Uniqueness().measure(snapshot()) == 1.0


def test_consistency_uses_only_formatted_types():
    status = pd.DataFrame(
        {
            "phone": [CellStatus.UNCHANGED, CellStatus.CHANGED, CellStatus.MISSING, CellStatus.INVALID],
            "name": [CellStatus.CHANGED] * 4,
        }
    )
    types = {"phone": ColumnType.PHONE, "name": ColumnType.TEXT}
    assert Consistency().measure(snapshot(statuses=[status], types=types)) == pytest.approx(1 / 3)


def test_assessment_measures_every_criterion():
    frame = pd.DataFrame({"phone": ["1", ""]})
    status = pd.DataFrame({"phone": [CellStatus.UNCHANGED, CellStatus.MISSING]})
    scores = assess(snapshot([frame], [status], {"phone": ColumnType.PHONE}, records=2, entities=1))
    assert dict(scores) == {"completeness": 0.5, "uniqueness": 0.5, "consistency": 1.0}
    record_level = assess(snapshot([frame], records=2, entities=2), [RecordCompleteness(), Uniqueness()])
    assert dict(record_level) == {"completeness": 0.5, "uniqueness": 1.0}
