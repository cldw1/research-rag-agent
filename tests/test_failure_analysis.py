from copy import deepcopy

import pytest

from scripts.analyze_qasper_failures import diagnose, failure_groups
from src.evaluation.qasper_benchmark import EvidenceQuestion


def record(qid, hits):
    return {"question_id": qid, "methods": {m: {"5": {"hit": h}, "10": {"hit": h}}
            for m, h in zip(("bm25", "dense", "hybrid"), hits)}}


def test_failure_population_is_disjoint_and_does_not_include_successes():
    records = [record("regression", (0, 1, 0)), record("miss", (0, 0, 0)),
               record("success", (0, 1, 1)), record("lexical", (1, 0, 0))]
    groups = failure_groups(records)
    assert {key: value[0] for key, value in groups.items()} == {
        "regression": "fusion_regression", "miss": "all_miss"}


def test_diagnosis_checks_actual_ranks_and_complete_source_coverage():
    question = EvidenceQuestion("paper", "q", "query", frozenset({"p0"}), (frozenset({"p0"}),))
    chunks = [{"source_metadata": {"spans": [{"block_id": "p0", "char_start": 0, "char_end": 10}]}}]
    baseline = record("q", (1, 1, 1))
    baseline["ranks"] = {m: [0] for m in ("bm25", "dense", "hybrid")}
    ranks = {"bm25": {0: 1}, "dense": {0: 1}}
    result, flags = diagnose(chunks, question, baseline, ranks, [1.0], "fixture", {"p0": 10})
    assert result["gold_chunk_count"] == 1 and flags["gold_present_in_either_candidate_pool"] == 1
    broken = deepcopy(baseline)
    broken["ranks"]["hybrid"] = [1]
    with pytest.raises(ValueError, match="rankings"):
        diagnose(chunks, question, broken, ranks, [1.0], "fixture", {"p0": 10})
    chunks[0]["source_metadata"]["spans"][0]["char_end"] = 5
    with pytest.raises(ValueError, match="completely"):
        diagnose(chunks, question, baseline, ranks, [1.0], "fixture", {"p0": 10})
