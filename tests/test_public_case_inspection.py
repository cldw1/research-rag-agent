"""Check case provenance, numeric explanations and reference-only public exports."""

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "results"


def report():
    return json.loads((ROOT / "qasper_case_inspection.json").read_text(encoding="utf-8"))


def test_case_population_is_fixed_and_belongs_to_the_published_baseline():
    data = report()
    baseline = json.loads((ROOT / "qasper_test_summary.json").read_text(encoding="utf-8"))
    assert data["baseline_records_sha256"] == baseline["artifacts"]["per_question_jsonl_sha256"]
    assert data["question_count"] == len(data["cases"]) == 6
    assert data["selection"]["cases"] == [
        {key: case[key] for key in ("case", "stratum", "paper_id", "question_id")} for case in data["cases"]]
    assert data["original_rankings_reproduced"] and data["gold_source_coverage_verified"]
    with (ROOT / "qasper_test_per_question.csv").open(encoding="utf-8", newline="") as stream:
        values = {(r["question_id"], r["method"]): r for r in csv.DictReader(stream)}
    for case in data["cases"]:
        assert float(values[(case["question_id"], "hybrid")]["hit"]) == 0
        if case["stratum"] == "fusion_dense_top1":
            assert float(values[(case["question_id"], "dense")]["mrr"]) == 1


@pytest.mark.parametrize("case_id", ["C01", "C02", "C03", "C04", "C05", "C06"])
def test_public_score_decomposition_and_reference_ids(case_id):
    case = next(c for c in report()["cases"] if c["case"] == case_id)
    assert sorted(set().union(*(set(r) for r in case["reference_paragraph_sets"]))) == case["gold_paragraph_ids"]
    for row in case["top5"] + case["gold_chunks"]:
        bm25 = 1 / (60 + row["bm25_rank"]) if row["bm25_rank"] is not None else 0
        dense = 1 / (60 + row["dense_rank"]) if row["dense_rank"] is not None else 0
        assert row["bm25_contribution"] == pytest.approx(bm25)
        assert row["dense_contribution"] == pytest.approx(dense)
        if row["in_candidate_union"]:
            assert row["rrf_score"] == pytest.approx(bm25 + dense)
        else:
            assert row["rrf_score"] is None and row["hybrid_rank"] is None
        assert row["gold_source_overlap"] == bool(set(row["source_block_ids"]) & set(case["gold_paragraph_ids"]))
    assert all(not row["gold_source_overlap"] for row in case["top5"])


def test_candidate_cutoff_and_reference_union_are_kept_distinct():
    cases = {case["case"]: case for case in report()["cases"]}
    outside = cases["C05"]
    assert outside["best_gold_hybrid_rank"] is None
    assert all(r["full_bm25_rank"] > 50 and r["full_dense_rank"] > 50 for r in outside["gold_chunks"])
    assert outside["counterfactual_full_candidate_pool"]["best_gold_hybrid_rank"] > 5
    alternative = cases["C04"]
    assert len(alternative["reference_paragraph_sets"]) == 2
    assert all(len(reference) == 1 for reference in alternative["reference_paragraph_sets"])
    assert alternative["baseline_context_metrics"]["anchor_first"]["full_paragraph_recall"] == .5


def test_context_counts_and_public_text_boundary():
    data = report()
    counts = data["diagnostic_context_rescues"]
    assert sum(r["questions"] for r in counts.values()) == 263
    assert sum(r["anchor_first_any_gold"] for r in counts.values()) == 40
    assert sum(r["anchor_first_all_gold"] for r in counts.values()) == 18
    forbidden = {"question", "text", "content", "title", "abstract", "gold_evidence", "api_key", "messages"}
    def visit(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(data)
