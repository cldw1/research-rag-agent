"""Recompute the published chunking comparison from its numeric question files."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.qasper_benchmark import paper_bootstrap

ROOT = Path(__file__).resolve().parents[1] / "results"


def data():
    summary = json.loads((ROOT / "qasper_chunking_validation.json").read_text(encoding="utf-8"))
    def rows(name):
        with (ROOT / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    return summary, rows("qasper_chunking_validation_per_question.csv"), rows("qasper_chunking_validation_contexts.csv")


def test_chunking_population_and_the_only_changed_configuration():
    summary, rows, _ = data()
    prior = json.loads((ROOT / "qasper_fusion_validation.json").read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    assert protocol["split"] == "validation"
    assert protocol["baseline_records_sha256"] == prior["records_sha256"]
    assert protocol["question_ids_sha256"] == prior["protocol"]["question_ids_sha256"]
    assert protocol["embedding_profile"] == prior["protocol"]["embedding_profile"]
    a, s = protocol["configs"]["adaptive"], protocol["configs"]["structure"]
    assert a == {**s, "strategy": "adaptive"}
    assert (a["max_tokens"], a["max_chars"], a["overlap_chars"]) == (480, 1200, 0)
    assert summary["question_count"] == len({r["question_id"] for r in rows}) == prior["question_count"]
    assert summary["adaptive_baseline_reproduced"] and summary["all_source_characters_preserved"]
    assert summary["production_default_unchanged"] and protocol["generation_api_calls"] == 0
    for method, old_name in (("dense", "dense"), ("hybrid", "hybrid_rrf60")):
        assert summary["results"]["adaptive"]["methods"][method]["5"] == prior["results"][old_name]["metrics_at_5"]


@pytest.mark.parametrize("strategy", ["adaptive", "structure"])
@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_every_retrieval_mean_recomputes(strategy, method):
    summary, rows, _ = data()
    rows = [r for r in rows if r["strategy"] == strategy and r["method"] == method]
    assert len(rows) == len({r["question_id"] for r in rows}) == summary["question_count"]
    assert len({r["paper_id"] for r in rows}) == summary["corpus"]["eligible_papers"]
    for key, expected in summary["results"][strategy]["methods"][method]["5"].items():
        values = [float(r[key]) for r in rows]
        assert all(0 <= v <= 1 for v in values)
        assert sum(values) / len(values) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("strategy", ["adaptive", "structure"])
@pytest.mark.parametrize("variant", ["top5", "anchor_first"])
def test_context_means_and_actual_length_recompute(strategy, variant):
    summary, _, rows = data()
    rows = [r for r in rows if r["strategy"] == strategy and r["variant"] == variant]
    assert len(rows) == len({r["question_id"] for r in rows}) == summary["question_count"]
    assert all(0 <= int(r["characters"]) <= summary["protocol"]["evidence_char_budget"] for r in rows)
    for key, expected in summary["results"][strategy]["contexts"][variant].items():
        assert sum(float(r[key]) for r in rows) / len(rows) == pytest.approx(expected, abs=1e-12)


def test_paired_primary_interval_guardrails_and_question_transitions_recompute():
    summary, rows, context = data()
    paired = {}
    for row in context:
        if row["variant"] == "anchor_first":
            item = paired.setdefault(row["question_id"], {"paper_id": row["paper_id"]})
            item[row["strategy"]] = float(row["full_paragraph_recall"])
    interval = paper_bootstrap(list(paired.values()), lambda r: r["structure"] - r["adaptive"])
    actual = summary["paired_structure_minus_adaptive"]["contexts"]["anchor_first"]["full_paragraph_recall"]
    assert interval == actual
    a = summary["results"]["adaptive"]["methods"]["hybrid"]["5"]
    s = summary["results"]["structure"]["methods"]["hybrid"]["5"]
    accepted = interval["low"] > 0 and s["hit"] >= a["hit"] and s["mrr"] >= a["mrr"]
    assert summary["recommended_for_followup"] == ("structure" if accepted else "adaptive")
    hits = {}
    for row in rows:
        if row["method"] == "hybrid":
            hits.setdefault(row["question_id"], {})[row["strategy"]] = float(row["hit"])
    counts = dict(both_hit=0, structure_only_hit=0, adaptive_only_hit=0, both_miss=0)
    for item in hits.values():
        a, s = item["adaptive"], item["structure"]
        counts["both_hit" if a and s else "structure_only_hit" if s else "adaptive_only_hit" if a else "both_miss"] += 1
    assert counts == summary["hybrid_hit_transitions_at_5"]


def test_public_hashes_vector_sizes_and_text_boundary():
    summary, rows, context = data()
    for name, metadata in summary["public_artifacts"].items():
        raw = (ROOT / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == metadata["sha256"]
        assert len(raw.splitlines()) - 1 == metadata["rows"]
        assert b"\r\n" not in raw
    dimension = summary["protocol"]["embedding_profile"]["dimension"]
    for value in summary["engineering"].values():
        assert value["vector_bytes"] == value["chunk_count"] * dimension * 4
    assert summary["engineering"]["structure"]["merged_paragraph_chunks"] == 0
    forbidden = {"question", "text", "content", "title", "abstract", "full_text", "gold_evidence", "api_key", "messages"}
    def visit(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(summary)
    assert not forbidden.intersection(rows[0]) and not forbidden.intersection(context[0])
