"""Recompute the public failure cohort and Validation comparison without models."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from scripts.analyze_qasper_failures import failure_groups

ROOT = Path(__file__).resolve().parents[1] / "results"


def validation_data():
    summary = json.loads((ROOT / "qasper_fusion_validation.json").read_text(encoding="utf-8"))
    with (ROOT / "qasper_fusion_validation_per_question.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return summary, rows


def test_audit_contains_exactly_the_two_baseline_failure_groups():
    baseline = json.loads((ROOT / "qasper_test_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "qasper_failure_audit.json").read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    with (ROOT / "qasper_test_per_question.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["question_id"]][row["method"]] = {"5": {"hit": float(row["hit"])}}
    selected = failure_groups([{"question_id": qid, "methods": methods} for qid, methods in grouped.items()])
    assert {r["question_id"]: r["category"] for r in audit["records"]} == {qid: value[0] for qid, value in selected.items()}
    assert len(audit["records"]) == audit["diagnosed_questions"]
    assert audit["baseline_records_sha256"] == baseline["artifacts"]["per_question_jsonl_sha256"]
    assert audit["original_rankings_reproduced"] and audit["source_chunks_completely_cover_all_gold"]
    categories = Counter(r["category"] for r in audit["records"])
    for category, number in categories.items():
        values = audit["counts"][category]
        assert values["questions"] == number
        assert values["gold_outside_both_candidate_pools"] + values["gold_present_in_either_candidate_pool"] == number
        assert values["hybrid_hit_at10"] <= values["any_method_hit_at10"] <= number


@pytest.mark.parametrize("policy", ["dense", "hybrid_rrf60", "hybrid_rrf10", "hybrid_dense1_5"])
def test_validation_means_are_recomputed_from_the_complete_public_population(policy):
    summary, rows = validation_data()
    selected = [r for r in rows if r["policy"] == policy]
    assert len(selected) == summary["question_count"]
    assert len({r["question_id"] for r in selected}) == len(selected)
    assert len({r["paper_id"] for r in selected}) == summary["corpus"]["eligible_papers"]
    for key, expected in summary["results"][policy]["metrics_at_5"].items():
        values = [float(r[key]) for r in selected]
        assert all(0 <= value <= 1 for value in values)
        assert sum(values) / len(values) == pytest.approx(expected, abs=1e-12)


def test_validation_selection_matches_the_frozen_acceptance_conditions():
    summary, _ = validation_data()
    assert summary["protocol"]["split"] == "validation"
    results = summary["results"]
    baseline = results["hybrid_rrf60"]["metrics_at_5"]
    accepted = [name for name in summary["protocol"]["policies"] if name != "hybrid_rrf60"
        and results[name]["metrics_at_5"]["hit"] >= baseline["hit"]
        and results[name]["metrics_at_5"]["mrr"] >= baseline["mrr"]
        and results[name]["paired_minus_rrf60"]["recall"]["low"] > 0]
    expected = max(accepted, key=lambda name: results[name]["metrics_at_5"]["recall"]) if accepted else "hybrid_rrf60"
    assert summary["selected_policy"] == expected


def test_public_validation_hash_and_text_exclusion():
    summary, rows = validation_data()
    artifact = summary["public_artifacts"]
    assert hashlib.sha256((ROOT / artifact["per_question"]).read_bytes()).hexdigest() == artifact["sha256"]
    assert artifact["rows"] == len(rows)
    forbidden = {"question", "gold_evidence", "full_text", "abstract", "api_key", "messages"}
    def visit(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(summary)
    visit(json.loads((ROOT / "qasper_failure_audit.json").read_text(encoding="utf-8")))
    assert not forbidden.intersection(rows[0])
