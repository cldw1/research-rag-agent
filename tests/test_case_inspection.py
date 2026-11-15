import re

import pytest

from src.evaluation.case_inspection import fenced_reference, inspect_rankings, score_explanation, select_cases
from src.evaluation.qasper_benchmark import EvidenceQuestion


def population():
    audit, baseline = [], []
    for qid, paper, category, dense, lexical, hybrid_hit10 in [
        ("a", "same", "fusion_regression", 1, 12, 1),
        ("b", "same", "fusion_regression", 1, 20, 1),
        ("c", "other", "fusion_regression", 1, 15, 1),
        ("d", "p3", "all_miss", 7, 9, 1), ("e", "p4", "all_miss", 8, 12, 1),
        ("f", "p5", "all_miss", None, None, 0), ("g", "p6", "all_miss", 14, 18, 0),
    ]:
        audit.append({"question_id": qid, "paper_id": paper, "category": category,
                      "first_dense_gold_rank": dense, "first_bm25_gold_rank": lexical})
        baseline.append({"question_id": qid, "methods": {
            "hybrid": {"10": {"hit": hybrid_hit10}},
            "dense": {"10": {"hit": int(dense is not None and dense <= 10)}},
            "bm25": {"10": {"hit": int(lexical is not None and lexical <= 10)}}}})
    return audit, baseline


def test_selection_is_order_independent_and_uses_distinct_papers_per_stratum():
    audit, baseline = population()
    result = select_cases(audit, baseline)
    assert result == select_cases(list(reversed(audit)), list(reversed(baseline)))
    assert [r["question_id"] for r in result["cases"]] == ["a", "c", "d", "e", "f", "g"]
    assert result["cohort_counts"]["fusion_dense_top1"] == 3


def test_insufficient_case_population_and_duplicate_baseline_ids_are_rejected():
    audit, baseline = population()
    with pytest.raises(ValueError, match="Insufficient"):
        select_cases(audit[:2], baseline[:2])
    with pytest.raises(ValueError, match="Duplicate"):
        select_cases(audit, baseline + [baseline[0]])


def test_missing_branch_does_not_invent_a_rank_or_vote():
    ranks = {"bm25": {0: 1}, "dense": {1: 1}}
    gold = score_explanation(1, ranks)
    assert gold["bm25_rank"] is None and gold["bm25_contribution"] == 0
    assert gold["rrf_score"] == pytest.approx(1/61)
    absent = score_explanation(2, ranks)
    assert not absent["in_candidate_union"] and absent["rrf_score"] is None


def test_full_pool_counterfactual_is_separate_from_original_candidate_ranks():
    chunks = [{"source_metadata": {"spans": [{"block_id": f"p{i}"}]}} for i in range(4)]
    query = EvidenceQuestion("paper", "q", "query", frozenset({"p3"}), (frozenset({"p3"}),))
    ranks = {"bm25": {0: 1, 1: 2}, "dense": {2: 1, 0: 2}}
    result = inspect_rankings(chunks, query, ranks, [3, 2, 1, 0], [0, 1, 2, 3], [2, 0, 1, 3])
    assert result["best_gold_hybrid_rank"] is None
    assert result["gold_chunks"][0]["full_dense_rank"] == 4
    assert result["counterfactual_full_candidate_pool"]["best_gold_hybrid_rank"] == 4
    changed = EvidenceQuestion("paper", "q", "query", frozenset({"p0"}), (frozenset({"p0"}),))
    other = inspect_rankings(chunks, changed, ranks, [3, 2, 1, 0], [0, 1, 2, 3], [2, 0, 1, 3])
    assert result["hybrid_top10_ordinals"] == other["hybrid_top10_ordinals"]


def test_local_source_fence_cannot_be_closed_by_external_markdown():
    text = '```python\nprint(1)\n```\n<script>example()</script>\n````'
    rendered = fenced_reference(text)
    first = rendered.splitlines()[0].removesuffix("text")
    assert len(first) > max(len(m.group()) for m in re.finditer(r"`+", text))
    assert rendered == first + "text\n" + text + "\n" + first
