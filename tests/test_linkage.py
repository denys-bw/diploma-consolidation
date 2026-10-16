import numpy as np
import pandas as pd
import pytest

from consolidator.evaluation import evaluate_positions
from consolidator.fusion import Fuser, Longest
from consolidator.linkage import (
    AutomaticKeys,
    AverageLinkage,
    BlockingKey,
    ClericalReview,
    Comparison,
    ConnectedComponents,
    CoTraining,
    DecisionRule,
    DuplicatePolicy,
    FellegiSunterScorer,
    FullComparison,
    KeyBlocking,
    KeyMethod,
    KeySelector,
    LinkageSettings,
    Linker,
    MatchStatus,
    MetricView,
    NumericNeighbourhood,
    OneToOneAssignment,
    OneToOneConstraint,
    PairScope,
    RecordSet,
    ReviewCorrection,
    ScoredPairs,
    ScoringTask,
    SortedNeighbourhood,
    SourceStructure,
    TokenBlocking,
    cluster_pairs,
    comparable_attributes,
    components,
    estimate_prior,
    fit_fellegi_sunter,
    frequency_adjustment,
    internal_duplicates_evidence,
    pair_groups,
    random_pairs,
    scale_neighbours,
)
from consolidator.linkage import features as feature_module
from consolidator.linkage.comparison import NumberComparator, PhoneComparator, TimeComparator, YearComparator
from consolidator.linkage.fellegi_sunter import fit_non_matches
from consolidator.linkage.pairs import canonical_pairs, count_within_groups, pairs_within_groups
from consolidator.linkage.reliability import EVENTS_PER_PARAMETER, FALSE_MATCHES, FEW_ATTRIBUTES
from consolidator.linkage.scoring import PairScores, assignment_priors, mutual_best
from consolidator.linkage.selection import select_metrics
from consolidator.profiling import ColumnType, profile_frame
from consolidator.statistics import sturges_bins
from tests.synthetic import linked_sources

TYPES = {"name": ColumnType.TEXT, "city": ColumnType.CATEGORY, "phone": ColumnType.PHONE, "year": ColumnType.YEAR}
CROSS_FULL = LinkageSettings(policy=DuplicatePolicy.ACROSS_CLEAN, blocker=FullComparison())


def restaurants() -> RecordSet:
    left = pd.DataFrame(
        {
            "name": ["arnie morton's of chicago", "art's delicatessen", "hotel bel-air", "cafe bizou", "campanile"],
            "city": ["los angeles", "studio city", "bel air", "sherman oaks", "los angeles"],
            "phone": ["+13102461501", "+18187621221", "+13104721211", "+18187888155", "+12139381447"],
            "year": ["1990", "1985", "1946", "1993", "1989"],
        },
        index=["1", "3", "5", "7", "9"],
    )
    right = pd.DataFrame(
        {
            "name": ["arnie morton's of chicago", "art's deli", "bel-air hotel", "cafe bizou", "chinois on main"],
            "city": ["los angeles", "studio city", "bel air", "sherman oaks", "santa monica"],
            "phone": ["+13102461501", "+18187621221", "+13104721211", "+18187888155", "+13103929025"],
            "year": ["1990", "1985", "1947", "", "1983"],
        },
        index=["2", "4", "6", "8", "10"],
    )
    return RecordSet.from_frames({"fodors": left, "zagats": right}, TYPES)


def synthetic(count: int = 300, shared: float = 0.5, seed: int = 0, internal: float = 0.0):
    frames, gold = linked_sources(count, shared, seed, internal)
    types = profile_frame(pd.concat(list(frames.values()))).types
    return RecordSet.from_frames(frames, types), gold


def f1(records: RecordSet, result, gold, scope: PairScope) -> float:
    found = cluster_pairs(result.labels, records.source_codes, scope)
    return evaluate_positions(found, records.pairs_from_ids(gold), records.size).f1


def scored(pairs, probabilities, weights=None, reviewed=None):
    pairs = np.asarray(pairs)
    probabilities = np.asarray(probabilities, dtype=float)
    weights = probabilities if weights is None else np.asarray(weights, dtype=float)
    reviewed = np.full(len(pairs), -1, dtype=np.int8) if reviewed is None else np.asarray(reviewed, dtype=np.int8)
    return ScoredPairs(pairs, weights, probabilities, reviewed)


def records_of(size, sources=None):
    codes = np.zeros(size, dtype=int) if sources is None else np.asarray(sources)
    frames = {
        f"s{code}": pd.DataFrame({"x": ["v"] * int((codes == code).sum())}) for code in sorted(set(codes.tolist()))
    }
    records = RecordSet.from_frames(frames, {"x": ColumnType.TEXT})
    order = np.argsort(codes, kind="stable")
    assert np.array_equal(records.source_codes, codes[order])
    return records


def test_record_set_keeps_ids_and_sources():
    records = restaurants()
    assert records.size == 10
    assert records.record_id(0) == ("fodors", "1")
    assert records.record_id(5) == ("zagats", "2")
    assert records.total_pairs(PairScope.CROSS) == 25
    assert records.total_pairs(PairScope.ALL) == 45
    pairs = records.pairs_from_ids([(("zagats", "2"), ("fodors", "1"))])
    assert pairs.tolist() == [[0, 5]]
    assert records.pair_ids(pairs) == {(("fodors", "1"), ("zagats", "2"))}


def test_pairs_within_groups_respects_scope():
    groups = np.array([0, 0, 0, 1, 1, -1])
    sources = np.array([0, 1, 1, 0, 0, 1])
    assert pairs_within_groups(groups, sources, PairScope.ALL).tolist() == [[0, 1], [0, 2], [1, 2], [3, 4]]
    assert pairs_within_groups(groups, sources, PairScope.CROSS).tolist() == [[0, 1], [0, 2]]
    assert count_within_groups(groups, sources, PairScope.ALL) == 4
    assert count_within_groups(groups, sources, PairScope.CROSS) == 2


def test_canonical_pairs_are_sorted_and_unique():
    assert canonical_pairs(np.array([[3, 1], [1, 3], [2, 2], [0, 4]])).tolist() == [[0, 4], [1, 3]]


def test_random_pairs_stay_in_scope():
    records = restaurants()
    sample = random_pairs(records, PairScope.CROSS, 20, seed=1)
    assert len(sample) == 20
    assert (records.source_codes[sample[:, 0]] != records.source_codes[sample[:, 1]]).all()


def test_full_comparison_and_key_blocking():
    records = restaurants()
    assert len(FullComparison().candidates(records, PairScope.CROSS)) == 25
    with pytest.raises(ValueError, match="Без блокування"):
        FullComparison(max_pairs=10).candidates(records, PairScope.CROSS)
    city = KeyBlocking((BlockingKey.of("city"),))
    assert city.candidates(records, PairScope.CROSS).tolist() == [[0, 5], [1, 6], [2, 7], [3, 8], [4, 5]]
    composite = BlockingKey.of("city").combined(BlockingKey.of("name", KeyMethod.PREFIX, 1))
    assert composite.label == "city + name[:1]"
    assert KeyBlocking((composite,)).candidates(records, PairScope.CROSS).tolist() == [[0, 5], [1, 6], [3, 8]]


def test_sorted_neighbourhood_window_follows_data_size():
    records = restaurants()
    key = (BlockingKey.of("phone", KeyMethod.DIGITS),)
    pairs = SortedNeighbourhood(key, window=2).candidates(records, PairScope.CROSS)
    assert [0, 5] in pairs.tolist()
    assert len(pairs) <= records.size
    automatic = SortedNeighbourhood(key).candidates(records, PairScope.CROSS)
    assert len(automatic) >= len(pairs)


def test_token_blocking_uses_logarithmic_neighbourhood():
    records = restaurants()
    assert scale_neighbours(records) == 4
    pairs = TokenBlocking(neighbours=1).candidates(records, PairScope.CROSS)
    assert {(0, 5), (1, 6), (3, 8)} <= {tuple(pair) for pair in pairs.tolist()}
    assert TokenBlocking().adaptive and not TokenBlocking(neighbours=3).adaptive


def test_neighbourhood_grows_only_where_matches_continue():
    names = [f"kalomira store {index}" for index in range(12)] + [f"unique{index} {index * 7}" for index in range(52)]
    frame = pd.DataFrame({"name": names})
    records = RecordSet.from_frames({"only": frame}, {"name": ColumnType.TEXT})
    cluster = set(range(12))

    def judge(pairs):
        return np.array([float(a in cluster and b in cluster) for a, b in pairs])

    fixed = TokenBlocking(neighbours=scale_neighbours(records)).candidates(records, PairScope.ALL)
    adaptive, _ = TokenBlocking().adaptive_candidates(records, PairScope.ALL, judge)
    inside = {(a, b) for a in cluster for b in cluster if a < b}
    assert not inside <= {tuple(pair) for pair in fixed.tolist()}
    assert inside <= {tuple(pair) for pair in adaptive.tolist()}
    assert len(adaptive) < records.total_pairs(PairScope.ALL)


def test_key_selection_covers_mutual_nearest_neighbours_within_budget():
    records, _ = synthetic(200)
    selector = KeySelector(records, PairScope.CROSS)
    suggestion = selector.suggest()
    assert suggestion.keys
    assert suggestion.candidates <= suggestion.budget == records.size * scale_neighbours(records)
    assert suggestion.estimated_completeness > 0.5
    standard = AutomaticKeys().resolve(records, PairScope.CROSS)
    assert standard == KeyBlocking(suggestion.keys)
    empty = RecordSet.from_frames(
        {"a": pd.DataFrame({"x": [""]}), "b": pd.DataFrame({"x": [""]})}, {"x": ColumnType.TEXT}
    )
    with pytest.raises(ValueError, match="ключі"):
        AutomaticKeys().resolve(empty, PairScope.CROSS)


def test_phone_levels_are_structural():
    comparator = PhoneComparator().fit(
        np.array(["3865286", "+13123865286", "+13123865287", "+15550001111"], dtype=object)
    )
    left, right = np.array([0, 1, 1, 0]), np.array([1, 2, 1, 3])
    codes = comparator.similarity(left, right) * (comparator.fixed_levels - 1)
    assert codes.round().tolist() == [PhoneComparator.SUFFIX, PhoneComparator.ONE_EDIT, PhoneComparator.SAME, 0]
    assert comparator.exact(left, right).tolist() == [False, False, True, False]


def test_numeric_similarity_is_calibrated_by_the_spread_of_values():
    small = NumberComparator().fit(np.array([str(value) for value in range(100, 200)], dtype=object))
    near = small.similarity(np.array([0]), np.array([5]))[0]
    far = small.similarity(np.array([0]), np.array([99]))[0]
    assert near > far
    assert 0.0 <= far < 0.1
    years = YearComparator().fit(np.array(["1990", "1985", "1946", "1947", ""], dtype=object))
    similarity = years.similarity(np.array([2, 0, 3]), np.array([3, 1, 4]))
    assert similarity[0] > similarity[1]
    assert np.isnan(similarity[2])


def test_time_of_day_difference_wraps_around_midnight():
    times = np.array(["23:55", "00:05", "12:00"] + [f"{hour:02d}:00" for hour in range(24)], dtype=object)
    comparator = TimeComparator().fit(times)
    similarity = comparator.similarity(np.array([0, 0]), np.array([1, 2]))
    assert similarity[0] > similarity[1]


def test_levels_follow_sturges_rule_with_a_separate_exact_level():
    records, _ = synthetic(200)
    pairs = FullComparison().candidates(records, PairScope.CROSS)
    matrix = Comparison.for_records(records, PairScope.CROSS).compare(pairs)
    index = matrix.attributes.index("name")
    graded = ~np.isnan(matrix.similarities[:, index]) & ~matrix.exact[:, index]
    assert matrix.n_levels[index] == sturges_bins(int(graded.sum())) + 1
    assert (matrix.levels[matrix.exact[:, index], index] == matrix.n_levels[index] - 1).all()
    phone = matrix.attributes.index("phone")
    assert matrix.n_levels[phone] == PhoneComparator.fixed_levels


def test_agreement_on_a_rare_value_weighs_more_than_on_a_common_one():
    names = np.array(["smith"] * 50 + ["zielinski", "zielinski"] + [f"name {i}" for i in range(48)], dtype=object)
    records = RecordSet.from_frames({"s": pd.DataFrame({"name": names})}, {"name": ColumnType.TEXT})
    matrix = Comparison.for_records(records, PairScope.ALL).compare(np.array([[0, 1], [50, 51]]))
    adjustment = frequency_adjustment(matrix.frequencies)
    assert adjustment[1] > adjustment[0]
    assert frequency_adjustment(np.full((2, 1), np.nan)).tolist() == [0.0, 0.0]


def test_comparable_attributes_skip_source_constant_columns():
    left = pd.DataFrame({"name": ["a", "b"], "origin": ["web", "web"]})
    right = pd.DataFrame({"name": ["a", "c"], "origin": ["crm", "crm"]})
    records = RecordSet.from_frames({"l": left, "r": right}, {"name": ColumnType.TEXT, "origin": ColumnType.CATEGORY})
    assert comparable_attributes(records, PairScope.CROSS) == ["name"]


def test_comparison_in_batches_matches_single_pass(monkeypatch):
    records = restaurants()
    pairs = FullComparison().candidates(records, PairScope.CROSS)
    whole = Comparison.for_records(records, PairScope.CROSS).compare(pairs)
    monkeypatch.setattr("consolidator.linkage.comparison.COMPARISON_BATCH", 7)
    batched = Comparison.for_records(records, PairScope.CROSS).compare(pairs)
    assert np.array_equal(whole.levels, batched.levels)
    assert np.allclose(whole.similarities, batched.similarities, equal_nan=True)


def test_em_recovers_known_mixture():
    rng = np.random.default_rng(0)
    matches = rng.choice(3, size=(200, 3), p=[0.05, 0.15, 0.8])
    non_matches = rng.choice(3, size=(1800, 3), p=[0.85, 0.1, 0.05])
    levels = np.vstack([matches, non_matches]).astype(np.int8)
    model = fit_fellegi_sunter(levels, (3, 3, 3), ("a", "b", "c"), prior=0.5)
    assert model.converged
    assert model.prior == pytest.approx(0.1, abs=0.03)
    for m, u in zip(model.m, model.u):
        assert m[2] > 0.6 and u[0] > 0.7
    weights = model.match_weights(levels)
    assert weights[:200].mean() > 0 > weights[200:].mean()
    assert model.parameters_frame()["weight"].tolist()[2] == pytest.approx(model.weights[0][2])


def test_em_ignores_missing_comparisons():
    levels = np.array([[2, -1], [2, 1], [0, 0], [0, -1]] * 50, dtype=np.int8)
    model = fit_fellegi_sunter(levels, (3, 2), ("a", "b"), prior=0.25)
    weights = model.match_weights(np.array([[2, -1]], dtype=np.int8))
    assert weights[0] == pytest.approx(model.weights[0][2])


def test_monotone_weights():
    rng = np.random.default_rng(1)
    levels = np.vstack(
        [rng.choice(4, size=(300, 2), p=[0.1, 0.4, 0.1, 0.4]), rng.choice(4, size=(700, 2), p=[0.7, 0.1, 0.15, 0.05])]
    )
    model = fit_fellegi_sunter(levels.astype(np.int8), (4, 4), ("a", "b"), prior=0.3)
    for weights in model.weights:
        assert np.all(np.diff(weights) >= -1e-9)


def test_pairs_within_a_source_get_their_own_non_duplicate_profile():
    rng = np.random.default_rng(4)
    across = np.repeat([[1, 1], [0, 0]], [100, 900], axis=0)
    formatted = np.column_stack([np.zeros(980, dtype=int), rng.random(980) < 0.6])
    within = np.vstack([np.ones((20, 2), dtype=int), formatted])
    model = fit_fellegi_sunter(across.astype(np.int8), (2, 2), ("a", "b"), prior=0.1)
    inside = fit_non_matches(within.astype(np.int8), (2, 2), ("a", "b"), model.m, 0.1)
    assert inside.m == model.m
    assert inside.weights[1][1] < model.weights[1][1] - 2.0
    assert inside.prior == pytest.approx(0.02, abs=0.01)


def test_prior_reestimation_for_assigned_pairs():
    assert estimate_prior(np.array([10.0] * 9 + [-10.0])) == pytest.approx(0.9, abs=0.01)


def test_pairs_chosen_by_one_to_one_assignment_get_their_own_match_share():
    size = 80
    records = records_of(2 * size, [0] * size + [1] * size)
    left = np.arange(size)
    pairs = np.vstack([np.column_stack([left, size + left]), np.column_stack([left, size + (left + 1) % size])])
    weights = np.concatenate([np.repeat([8.0, 0.0, -6.0], [40, 20, 20]), np.full(size, -8.0)])
    groups = np.zeros(len(pairs), dtype=np.int64)
    pooled = PairScores(None, weights, groups, (estimate_prior(weights),))
    task = ScoringTask(records, pairs, None, pooled.prior)
    regrouped = assignment_priors(task, pooled)
    assert regrouped.groups.tolist() == [1] * size + [0] * size
    assert (pooled.probabilities[40:60] < 0.5).all()
    assert (regrouped.probabilities[40:60] > 0.5).all()
    assert (regrouped.probabilities[60:] < 0.5).all()
    few = PairScores(None, weights[:5], groups[:5], pooled.group_priors)
    assert assignment_priors(ScoringTask(records, pairs[:5], None, pooled.prior), few) is few


def test_mutual_best_marks_pairs_that_are_each_others_strongest_candidate():
    pairs = np.array([[0, 2], [0, 3], [1, 2], [1, 3], [0, 4]])
    best = mutual_best(pairs, np.array([0.9, 0.6, 0.7, 0.4, 0.5]), np.array([0, 0, 1, 1, 2]))
    assert best.tolist() == [True, False, False, False, True]


def test_one_to_one_constraint_caps_expected_partners():
    pairs = np.array([[0, 2], [0, 3], [1, 2]])
    constraint = OneToOneConstraint(pairs, np.array([0, 0, 1, 1]), 4)
    adjusted = constraint.apply(np.array([0.9, 0.9, 0.9]))
    assert adjusted[0] + adjusted[1] <= 1.0 + 1e-6
    assert adjusted[0] + adjusted[2] <= 1.0 + 1e-6
    assert constraint.bound == 2.0
    assert constraint.violated(np.array([0.9, 0.9, 0.9]))


def test_decision_zones_follow_tolerated_error_rates():
    rule = DecisionRule(false_match_rate=0.05, missed_match_rate=0.05)
    probabilities = np.array([0.999] * 40 + [0.6] * 10 + [0.4] * 10 + [0.001] * 400)
    statuses = rule.statuses(probabilities)
    assert (statuses[:40] == MatchStatus.MATCH).all()
    assert (statuses[-400:] == MatchStatus.NON_MATCH).all()
    assert (statuses[40:50] == MatchStatus.MATCH).sum() == 5
    assert (statuses[40:60] == MatchStatus.POSSIBLE).sum() >= 5
    certain = DecisionRule().statuses(np.array([0.999, 0.998, 0.001, 0.002]))
    assert MatchStatus.POSSIBLE not in certain.tolist()
    contested = np.zeros(len(probabilities), dtype=bool)
    contested[0] = True
    disputed = DecisionRule(false_match_rate=0.0).statuses(probabilities, contested)
    assert disputed[0] == MatchStatus.POSSIBLE
    upper, lower = rule.probability_bounds(probabilities, statuses)
    assert lower < 0.5 <= upper
    with pytest.raises(ValueError):
        DecisionRule(match_probability=1.0)


def test_connected_components_and_average_linkage():
    pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]])
    probabilities = np.array([0.95, 0.9, 0.95, 0.1, 0.1])
    records = records_of(4)
    chained = ConnectedComponents().cluster(records, scored(pairs, probabilities), 0.5)
    assert len(set(chained.labels.tolist())) == 1
    assert chained.accepted.tolist() == [True, True, True, False, False]
    average = AverageLinkage().cluster(records, scored(pairs, probabilities), 0.5)
    labels = average.labels
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
    assert average.accepted.tolist() == [True, False, True, False, False]
    assert len(set(components(4, pairs[probabilities >= 0.5]).tolist())) == 1


def test_average_linkage_ignores_pairs_between_unrelated_groups():
    pairs = np.array([[0, 1], [2, 3], [1, 2], [0, 3], [4, 5]])
    probabilities = np.array([0.9, 0.8, 0.1, 0.2, 0.4])
    labels = AverageLinkage().labels(np.zeros(6, dtype=np.int64), pairs, probabilities, 0.5)
    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert labels[0] != labels[2] and labels[4] != labels[5]


def test_average_linkage_counts_pairs_outside_candidates_as_non_matches():
    sources = np.zeros(4, dtype=np.int64)
    chain = np.array([[0, 1], [2, 3], [1, 2], [0, 3]])
    labels = AverageLinkage().labels(sources, chain, np.array([0.9, 0.8, 0.7, 0.6]), 0.5)
    assert len(set(labels.tolist())) == 2
    complete = np.array([[0, 1], [2, 3], [1, 2], [0, 3], [0, 2], [1, 3]])
    labels = AverageLinkage().labels(sources, complete, np.array([0.9, 0.8, 0.7, 0.6, 0.7, 0.6]), 0.5)
    assert len(set(labels.tolist())) == 1


def test_average_linkage_counts_only_pairs_in_scope():
    sources = np.array([0, 0, 1])
    pairs = np.array([[0, 2], [1, 2]])
    probabilities = np.array([0.9, 0.8])
    across = AverageLinkage().labels(sources, pairs, probabilities, 0.5, PairScope.CROSS)
    assert len(set(across.tolist())) == 1
    everywhere = AverageLinkage().labels(sources, pairs, probabilities, 0.5, PairScope.ALL)
    assert everywhere[0] == everywhere[2] and everywhere[1] != everywhere[2]


def test_one_to_one_assignment_allows_one_record_per_source():
    records = records_of(4, [0, 0, 1, 1])
    pairs = np.array([[0, 2], [1, 2], [0, 3]])
    result = OneToOneAssignment().cluster(records, scored(pairs, [0.9, 0.8, 0.6], [5.0, 4.0, 1.0]), 0.5)
    assert result.accepted.tolist() == [True, False, False]
    labels = result.labels
    assert labels[0] == labels[2] and labels[1] != labels[2] and labels[3] != labels[0]
    reviewed = OneToOneAssignment().cluster(records, scored(pairs, [0.9, 0.8, 0.6], [5.0, 4.0, 1.0], [0, 1, -1]), 0.5)
    assert reviewed.accepted.tolist() == [False, True, True]


def test_linker_end_to_end_on_small_sources():
    records = restaurants()
    result = Linker(CROSS_FULL).run(records)
    found = records.pair_ids(result.matches)
    assert (("fodors", "1"), ("zagats", "2")) in found
    assert (("fodors", "9"), ("zagats", "10")) not in found
    assert set(result.timings) == {"blocking", "comparison", "classification", "clustering"}
    assert result.entities == records.size - len(result.matches)
    assert result.blocker == FullComparison()
    assert sum(result.count(status) for status in MatchStatus) == len(result.candidates)
    assert 0.0 <= result.expected_false_matches() <= len(result.matches)


def test_default_linker_links_synthetic_sources():
    records, gold = synthetic(400, shared=0.5, seed=3)
    result = Linker().run(records)
    assert result.structure.decided and result.structure.clean
    assert f1(records, result, gold, PairScope.CROSS) > 0.95
    assert result.metric_choices
    assert all(
        choice.metric in feature_module.candidate_metrics(records.types[choice.attribute])
        for choice in result.metric_choices.values()
    )


def test_clerical_review_overrides_the_model():
    records = restaurants()
    automatic = Linker(CROSS_FULL).run(records)
    assert (("fodors", "5"), ("zagats", "6")) in records.pair_ids(automatic.matches)
    review = ClericalReview.of(
        [
            (("zagats", "6"), ("fodors", "5"), False),
            (("fodors", "9"), ("zagats", "10"), True),
            (("fodors", "404"), ("zagats", "2"), True),
        ]
    )
    reviewed = Linker(CROSS_FULL).run(records, review)
    found = records.pair_ids(reviewed.matches)
    assert (("fodors", "5"), ("zagats", "6")) not in found
    assert (("fodors", "9"), ("zagats", "10")) in found
    assert sorted(reviewed.reviewed[reviewed.reviewed >= 0].tolist()) == [0, 1]
    assert len(review) == 3


def test_linker_without_comparable_attributes_keeps_records_apart():
    frames = {
        "a": pd.DataFrame({"note": ["", ""]}, index=["1", "2"]),
        "b": pd.DataFrame({"note": ["", ""]}, index=["1", "2"]),
    }
    records = RecordSet.from_frames(frames, {"note": ColumnType.TEXT})
    result = Linker().run(records)
    assert result.model is None
    assert result.labels.tolist() == [0, 1, 2, 3]
    assert result.entities == 4
    assert len(result.matches) == 0


def test_fusion_keeps_lineage_of_linked_records():
    records = restaurants()
    labels = np.arange(records.size)
    labels[5] = 0
    labels[6] = 1
    result = Fuser({"name": Longest()}).fuse(records, labels)
    assert len(result.entities) == 8
    first = result.entities.loc[0]
    assert first["sources"] == "fodors, zagats"
    assert first["records"] == 2
    second = result.entities.loc[1]
    assert second["name"] == "art's delicatessen"
    assert set(result.conflicts["attribute"]) == {"name"}
    assert result.merged_records == 2


def test_metric_view_covers_every_attribute():
    records = restaurants()
    pairs = FullComparison().candidates(records, PairScope.CROSS)
    view = MetricView.for_records(records, ["name", "phone", "year"])
    values = view.compute(pairs)
    assert {name.split(": ")[0] for name in values.feature_names} == {"name", "phone", "year"}
    assert values.features().shape == (len(pairs), view.width)
    assert len(values.metrics("name")) > 1 and values.metrics("phone") == ["phone"]


def test_metric_selection_prefers_the_metric_that_separates_likely_duplicates():
    generator = np.random.default_rng(5)
    words = [f"w{index}" for index in range(300)]
    first = [" ".join(generator.choice(words, 3, replace=False)) for _ in range(150)]
    second = [" ".join(reversed(name.split())) for name in first]
    numbers = [str(value) for value in generator.integers(10_000, 99_999, 150)]
    cities = [str(value) for value in generator.choice(["north", "south", "east", "west"], 150)]
    left = pd.DataFrame({"name": first, "code": numbers, "city": cities})
    right = pd.DataFrame({"name": second, "code": numbers, "city": cities})
    types = {"name": ColumnType.TEXT, "code": ColumnType.CODE, "city": ColumnType.CATEGORY}
    records = RecordSet.from_frames({"l": left, "r": right}, types)
    pairs = FullComparison().candidates(records, PairScope.CROSS)
    matrix = Comparison.for_records(records, PairScope.CROSS).compare(pairs)
    scores = FellegiSunterScorer().unsupervised(ScoringTask(records, pairs, matrix, 0.01))
    values = MetricView.for_records(records, matrix.attributes).compute(pairs)
    choice = select_metrics(matrix, scores.attribute_weights(matrix.levels), scores.priors, values)["name"]
    assert choice.scores["token_set"] > choice.scores["levenshtein"]
    assert choice.metric in ("token_set", "tfidf", "jaccard")


def test_co_training_falls_back_to_fellegi_sunter_without_metric_values(monkeypatch):
    monkeypatch.setattr(feature_module, "MAX_FEATURE_VALUES", 1)
    records = restaurants()
    pairs = FullComparison().candidates(records, PairScope.CROSS)
    comparison = Comparison.for_records(records, PairScope.CROSS).compare(pairs)
    plain = FellegiSunterScorer().score(ScoringTask(records, pairs, comparison, 0.1))
    trained = CoTraining().score(ScoringTask(records, pairs, comparison, 0.1))
    assert trained.rounds == 0
    np.testing.assert_allclose(trained.weights, plain.weights)


def test_co_training_stops_when_decisions_stop_changing():
    records, gold = synthetic(300, seed=7)
    result = Linker(LinkageSettings(policy=DuplicatePolicy.ACROSS)).run(records)
    assert 1 <= result.training_rounds < 50
    assert f1(records, result, gold, PairScope.CROSS) > 0.9


def test_reviewed_pairs_correct_the_model():
    records = restaurants()
    pairs = FullComparison().candidates(records, PairScope.CROSS)
    comparison = Comparison.for_records(records, PairScope.CROSS).compare(pairs)
    target = int(np.flatnonzero((pairs[:, 0] == 2) & (pairs[:, 1] == 7))[0])
    automatic = FellegiSunterScorer().score(ScoringTask(records, pairs, comparison, 0.1))
    reviewed = np.full(len(pairs), -1, dtype=np.int8)
    reviewed[target] = 0
    reviewed[int(np.argmax(automatic.probabilities))] = 1
    guided = FellegiSunterScorer().score(ScoringTask(records, pairs, comparison, 0.1, reviewed=reviewed))
    assert not automatic.corrected and guided.corrected
    assert guided.probabilities[target] < automatic.probabilities[target]
    assert guided.independent is not None


def test_review_correction_stays_close_to_the_model_with_one_answer():
    records, _ = synthetic(200, seed=8)
    pairs = FullComparison().candidates(records, PairScope.CROSS)
    comparison = Comparison.for_records(records, PairScope.CROSS).compare(pairs)
    base = FellegiSunterScorer().unsupervised(ScoringTask(records, pairs, comparison, 0.01))
    reviewed = np.full(len(pairs), -1, dtype=np.int8)
    reviewed[int(np.argmax(base.probabilities))] = 1
    corrected = ReviewCorrection().apply(ScoringTask(records, pairs, comparison, 0.01, reviewed=reviewed), base)
    assert np.corrcoef(base.weights, corrected.weights)[0, 1] > 0.9


def test_reliability_reports_expected_errors_against_tolerances():
    records, _ = synthetic(300, seed=9)
    result = Linker(LinkageSettings(policy=DuplicatePolicy.ACROSS)).run(records)
    reliability = result.reliability()
    assert reliability.accepted == int(result.settings.decision.accepted(result.effective_probabilities).sum())
    assert 0.0 <= reliability.false_share <= 1.0 and 0.0 <= reliability.missed_share <= 1.0
    assert reliability.suggested_reviews == (0 if reliability.reliable else reliability.possible)
    strict = Linker(LinkageSettings(policy=DuplicatePolicy.ACROSS, decision=DecisionRule(false_match_rate=0.0))).run(
        records
    )
    assert strict.count(MatchStatus.POSSIBLE) >= result.count(MatchStatus.POSSIBLE)
    if strict.reliability().expected_false > 1.0:
        assert FALSE_MATCHES in strict.reliability().reasons


def test_review_queue_alternates_sides_of_the_threshold_starting_near_it():
    records, _ = synthetic(300, seed=10)
    result = Linker(LinkageSettings(policy=DuplicatePolicy.ACROSS, decision=DecisionRule(false_match_rate=0.0))).run(
        records
    )
    queue = result.review_queue()
    assert len(queue) == len(set(queue.tolist())) == result.count(MatchStatus.POSSIBLE)
    assert (result.reviewed[queue] == -1).all()
    probabilities = result.probabilities[queue]
    above = probabilities >= 0.5
    if above.any() and (~above).any():
        assert above[0] != above[1]
        assert np.all(np.diff(probabilities[above]) >= 0)
        assert np.all(np.diff(probabilities[~above]) <= 0)
    extended = result.review_queue(extended=True)
    assert len(extended) == int((result.reviewed == -1).sum())


def test_sources_without_internal_repeats_are_detected_as_clean():
    records, _ = synthetic(300, seed=11)
    structure = Linker(LinkageSettings(blocker=FullComparison())).run(records).structure
    assert structure.decided
    assert structure.scope is PairScope.CROSS and structure.clean


def test_internal_repeats_are_detected():
    records, gold = synthetic(300, seed=12, internal=0.3)
    result = Linker(LinkageSettings(blocker=FullComparison())).run(records)
    assert result.structure.decided
    assert result.scope is PairScope.ALL and not result.structure.clean
    assert result.structure.evidence > 0
    assert f1(records, result, gold, PairScope.ALL) > 0.9


def test_declared_policy_is_not_second_guessed():
    records, _ = synthetic(200, seed=13, internal=0.3)
    result = Linker(LinkageSettings(policy=DuplicatePolicy.ACROSS_CLEAN, blocker=FullComparison())).run(records)
    assert not result.structure.decided
    assert result.scope is PairScope.CROSS
    assert SourceStructure.declared(DuplicatePolicy.AUTO, 2) is None
    assert SourceStructure.declared(DuplicatePolicy.AUTO, 1).scope is PairScope.ALL


def test_confirmed_internal_match_settles_the_structure():
    records, _ = synthetic(200, seed=14)
    review = ClericalReview.of([(("left", "a0"), ("left", "a1"), True)])
    result = Linker(LinkageSettings(blocker=FullComparison())).run(records, review)
    assert result.scope is PairScope.ALL


def test_internal_evidence_compares_models_with_and_without_internal_matches():
    records, _ = synthetic(200, seed=15, internal=0.3)
    pairs = TokenBlocking(neighbours=8).candidates(records, PairScope.ALL)
    comparison = Comparison.for_records(records, PairScope.ALL).compare(pairs)
    groups = pair_groups(records, pairs)
    assert set(groups.tolist()) == {0, 1}
    assert internal_duplicates_evidence(comparison, groups, 0.05) > 0
    clean, _ = synthetic(200, seed=15)
    pairs = TokenBlocking(neighbours=8).candidates(clean, PairScope.ALL)
    comparison = Comparison.for_records(clean, PairScope.ALL).compare(pairs)
    assert internal_duplicates_evidence(comparison, pair_groups(clean, pairs), 0.05) < 0
    closed = fit_fellegi_sunter(
        comparison.levels,
        comparison.n_levels,
        comparison.attributes,
        prior=0.05,
        groups=pair_groups(clean, pairs),
        closed={0},
    )
    assert closed.priors[0] == 0.0


def test_few_attributes_ask_for_answers_spread_over_the_weight_range():
    records, gold = synthetic(200, seed=16)
    settings = LinkageSettings(policy=DuplicatePolicy.ACROSS, blocker=FullComparison(), attributes=("name",))
    result = Linker(settings).run(records)
    reliability = result.reliability()
    assert result.few_attributes
    assert FEW_ATTRIBUTES in reliability.reasons
    assert reliability.suggested_reviews == EVENTS_PER_PARAMETER * 2
    queue = result.review_queue()
    assert len(queue) == reliability.suggested_reviews
    low, high = result.weights.min(), result.weights.max()
    chosen = (result.weights[queue] - low) / (high - low)
    assert chosen.max() == 1.0 and chosen.min() < 0.1
    assert np.histogram(chosen, bins=4, range=(0, 1))[0].min() > 0
    truth = records.pairs_from_ids(gold)
    known = {tuple(pair) for pair in truth.tolist()}
    review = ClericalReview.of(
        (records.record_id(int(a)), records.record_id(int(b)), (int(a), int(b)) in known)
        for a, b in result.candidates[queue]
    )
    corrected = Linker(settings).run(records, review)
    assert corrected.corrected
    assert FEW_ATTRIBUTES not in corrected.reliability().reasons


def test_tables_without_text_are_blocked_by_sorted_numbers():
    generator = np.random.default_rng(4)
    values = generator.uniform(0, 1000, 60)
    frames = {
        "a": pd.DataFrame({"x": [f"{value:.3f}" for value in values[:40]]}, index=[f"a{i}" for i in range(40)]),
        "b": pd.DataFrame({"x": [f"{value:.2f}" for value in values[20:]]}, index=[f"b{i}" for i in range(20, 60)]),
    }
    records = RecordSet.from_frames(frames, {"x": ColumnType.NUMBER})
    blocker = TokenBlocking().resolve(records, PairScope.CROSS)
    assert isinstance(blocker, NumericNeighbourhood)
    pairs = {tuple(pair) for pair in blocker.candidates(records, PairScope.CROSS).tolist()}
    truth = records.pairs_from_ids({(("a", f"a{i}"), ("b", f"b{i}")) for i in range(20, 40)})
    assert all(tuple(sorted(pair)) in pairs for pair in truth.tolist())


def test_answers_of_one_kind_only_fix_the_reviewed_pairs():
    records, _ = synthetic(150, seed=21)
    settings = LinkageSettings(policy=DuplicatePolicy.ACROSS)
    result = Linker(settings).run(records)
    rejected = np.argsort(-result.probabilities)[5:15]
    review = ClericalReview.of(
        (records.record_id(int(a)), records.record_id(int(b)), False) for a, b in result.candidates[rejected]
    )
    corrected = Linker(settings).run(records, review)
    open_pairs = corrected.reviewed == -1
    assert not corrected.corrected
    assert np.allclose(corrected.probabilities[open_pairs], result.probabilities[open_pairs])
    assert (corrected.reviewed[~open_pairs] == 0).all()
