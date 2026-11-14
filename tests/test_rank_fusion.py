import math

import pytest

from src.retrieval.rank_fusion import fused_order, reciprocal_rank_scores


def test_default_fusion_matches_hand_calculated_scores_and_stable_ties():
    rankings = {"bm25": {0: 1, 1: 2, 2: 3}, "dense": {2: 1, 1: 2, 0: 3}}
    values = reciprocal_rank_scores(rankings)
    assert values[0] == pytest.approx(1/61 + 1/63)
    assert values[1] == pytest.approx(2/62)
    assert values[0] == values[2]
    assert fused_order(rankings) == [0, 2, 1]


def test_dense_weight_only_changes_fusion_and_retains_all_candidates():
    rankings = {"bm25": {0: 1, 1: 2, 2: 3}, "dense": {2: 1, 1: 2, 0: 3}}
    assert fused_order(rankings, weights={"bm25": 1, "dense": 1.5})[0] == 2
    assert set(fused_order(rankings, constant=10)) == {0, 1, 2}
    assert rankings["bm25"][0] == 1


@pytest.mark.parametrize("constant", [0, -1, math.nan, math.inf, True])
def test_invalid_constants_are_rejected(constant):
    with pytest.raises(ValueError):
        reciprocal_rank_scores({"dense": {0: 1}}, constant=constant)


@pytest.mark.parametrize("weights", [{"unknown": 1}, {"dense": -1}, {"dense": 0}, {"dense": math.nan}])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        reciprocal_rank_scores({"dense": {0: 1}}, weights=weights)


def test_unpaired_single_ranking_and_empty_candidates_are_supported():
    assert fused_order({"bm25": {}, "dense": {4: 1, 2: 2}}) == [4, 2]
    assert fused_order({"bm25": {}, "dense": {}}) == []


def test_fractional_or_boolean_ranks_cannot_silently_change_scores():
    for rank in (1.5, True, 0):
        with pytest.raises(ValueError):
            reciprocal_rank_scores({"dense": {0: rank}})


def test_validation_does_not_select_a_candidate_that_loses_hit_rate():
    from scripts.compare_rank_fusion import POLICIES, summarize_records
    baseline = {"hit": 1.0, "recall": .5, "mrr": .5, "ndcg": .5, "evidence_f1": .5}
    records = [{"paper_id": str(i), "methods": {
        "dense": dict(baseline), "hybrid_rrf60": dict(baseline),
        "hybrid_rrf10": {**baseline, "recall": .8, "hit": 0.0},
        "hybrid_dense1_5": dict(baseline)}} for i in range(2)]
    _, selected = summarize_records(records, POLICIES)
    assert selected == "hybrid_rrf60"


def test_validation_selects_a_recall_gain_only_when_guards_pass():
    from scripts.compare_rank_fusion import POLICIES, summarize_records
    baseline = {"hit": 1.0, "recall": .5, "mrr": .5, "ndcg": .5, "evidence_f1": .5}
    records = [{"paper_id": str(i), "methods": {
        "dense": dict(baseline), "hybrid_rrf60": dict(baseline),
        "hybrid_rrf10": dict(baseline),
        "hybrid_dense1_5": {**baseline, "recall": .8}}} for i in range(2)]
    _, selected = summarize_records(records, POLICIES)
    assert selected == "hybrid_dense1_5"
