"""Public counts must trace to per-answer audits, with no private answer text."""

import json
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["pdf_qa_retry_summary.json", "pdf_qa_retry_v2_summary.json"])
def test_public_retry_comparison_accounts_for_reused_answers_and_all_score_changes(filename):
    report = json.loads((ROOT / "results" / filename).read_text(encoding="utf-8"))
    base = json.loads((ROOT / "results/pdf_qa_answer_summary.json").read_text(encoding="utf-8"))
    before = {r["question_id"]: r for r in base["review_records"]}
    after = {r["question_id"]: r for r in report["review_records"]}
    choices = {r["question_id"]: r for r in report["selections"]}
    assert set(before) == set(after) == set(choices)
    assert len(after) == report["question_count"]
    deltas = [after[qid]["correctness"]-before[qid]["correctness"] for qid in after]
    assert report["correctness_changes"] == {"improved": sum(d>0 for d in deltas),
        "unchanged": sum(d==0 for d in deltas), "worsened": sum(d<0 for d in deltas)}
    for qid, choice in choices.items():
        assert choice["answer_sha256"] == after[qid]["answer_sha256"]
        if choice["selected"] == "baseline":
            assert before[qid] == after[qid]
        if choice["retry_status"] == "uncertain":
            assert choice["selected"] == "baseline"
    for key in ("baseline_ai_review", "policy_ai_review"):
        assert report[key]["human_metrics"] is None and not report[key]["independent_human_review"]
    incremental = report["incremental_generation"]
    assert len(report["retried_questions"]) == incremental["question_count"]
    assert incremental["known_returned_usage"]["prompt_tokens"] + incremental["known_returned_usage"]["completion_tokens"] == incremental["known_returned_usage"]["total_tokens"]
    assert report["total_known_tokens_with_baseline"] == incremental["known_returned_usage"]["total_tokens"] + base["known_returned_usage"]["total_tokens"]


@pytest.mark.parametrize("filename", ["pdf_qa_retry_summary.json", "pdf_qa_retry_v2_summary.json"])
def test_public_retry_report_contains_diagnostics_without_raw_answers_or_passages(filename):
    report = json.loads((ROOT / "results" / filename).read_text(encoding="utf-8"))
    forbidden = {"answer", "prompt", "messages", "api_key", "authorization", "provided_sources", "context", "gold_evidence"}
    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for nested in value.values():
                check(nested)
        elif isinstance(value, list):
            for nested in value:
                check(nested)
    check(report)
    assert report["independent_human_metrics"] is None
