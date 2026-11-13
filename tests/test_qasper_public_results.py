"""Recompute the published benchmark and protect its export boundary, offline."""

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.export_qasper_results import export_run

ROOT = Path(__file__).resolve().parents[1]


def public_results():
    summary = json.loads((ROOT / "results/qasper_test_summary.json").read_text(encoding="utf-8"))
    with (ROOT / "results/qasper_test_per_question.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return summary, rows


def test_published_population_exclusions_and_method_pairs_are_complete():
    summary, rows = public_results()
    corpus = summary["corpus"]
    exclusions = sum(corpus[key] for key in ("unanswerable", "no_text_evidence", "unmapped_text_evidence", "ambiguous_text_evidence"))
    assert corpus["eligible_questions"] + exclusions == corpus["total_questions"]
    assert summary["question_count"] == corpus["eligible_questions"]
    assert len(rows) == summary["question_count"] * len(summary["protocol"]["methods"])
    assert len({r["paper_id"] for r in rows}) == summary["paper_count"]
    expected = {(r["question_id"], method) for r in rows for method in summary["protocol"]["methods"]}
    actual = {(r["question_id"], r["method"]) for r in rows}
    assert actual == expected and len(actual) == len(rows)
    assert all(r["k"] == "5" for r in rows)


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_summary_and_intervals_match_public_question_values(method):
    summary, rows = public_results()
    relevant = [r for r in rows if r["method"] == method]
    for metric, expected in summary["methods"][method]["metrics"]["5"].items():
        values = [float(r[metric]) for r in relevant]
        assert all(0 <= v <= 1 + 1e-12 for v in values)
        assert sum(values) / len(values) == pytest.approx(expected, abs=1e-12)
        interval = summary["methods"][method]["ci95_at_5"][metric]
        assert interval["mean"] == pytest.approx(expected, abs=1e-12)
        assert interval["low"] <= expected <= interval["high"]
        assert interval["sampling_unit"] == "paper" and interval["resamples"] == 2000


def test_public_hashes_and_paired_differences_are_recomputable():
    summary, rows = public_results()
    metadata = summary["public_artifacts"]
    for name_key, hash_key in [("per_question_at_5", "per_question_at_5_sha256"), ("bad_cases", "bad_cases_sha256")]:
        assert hashlib.sha256((ROOT / "results" / metadata[name_key]).read_bytes()).hexdigest() == metadata[hash_key]
    groups = {method: {r["question_id"]: r for r in rows if r["method"] == method}
              for method in summary["protocol"]["methods"]}
    for metric, result in summary["paired_hybrid_minus_dense_at_5"].items():
        differences = [float(row[metric]) - float(groups["dense"][key][metric]) for key, row in groups["hybrid"].items()]
        assert sum(differences) / len(differences) == pytest.approx(result["mean"], abs=1e-12)
    cases = json.loads((ROOT / "results/qasper_test_bad_cases.json").read_text(encoding="utf-8"))
    missed = {key for key in groups["hybrid"] if all(float(groups[m][key]["hit"]) == 0 for m in groups)}
    assert cases["all_methods_miss_at_5"]["count"] == len(missed)


def test_public_outputs_contain_ids_and_numbers_rather_than_document_text():
    summary, rows = public_results()
    forbidden = {"question", "gold_evidence", "full_text", "abstract", "messages", "api_key"}
    def visit(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(summary)
    visit(json.loads((ROOT / "results/qasper_test_bad_cases.json").read_text(encoding="utf-8")))
    assert not forbidden.intersection(rows[0])


def fixture_run(path):
    path.mkdir()
    metrics = {key: 1.0 for key in ("hit", "precision", "mrr", "ndcg", "recall", "full_paragraph_recall", "all_gold_covered", "evidence_f1")}
    methods = {m: {str(k): deepcopy(metrics) for k in (1, 5, 10)} for m in ("bm25", "dense", "hybrid")}
    contexts = {"top5": {"recall": 1.0, "characters": 100}}
    record = {"paper_id": "p", "question_id": "q", "question": "Fixture text must stay private",
              "methods": methods, "contexts": contexts, "ranks": {m: [0] for m in methods}}
    protocol = {"methods": list(methods), "k_values": [1, 5, 10]}
    raw = (json.dumps(record) + "\n").encode()
    protocol_bytes = json.dumps(protocol).encode()
    (path / "per_question.jsonl").write_bytes(raw)
    (path / "protocol.json").write_bytes(protocol_bytes)
    summary = {"question_count": 1, "paper_count": 1, "protocol": protocol, "corpus": {},
               "methods": {m: {"metrics": values} for m, values in methods.items()},
               "contexts": contexts, "paired_hybrid_minus_dense_at_5": {}, "timings": {},
               "generation_api_calls": 0, "scope": "fixture", "artifacts": {
                   "per_question_jsonl_sha256": hashlib.sha256(raw).hexdigest(),
                   "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest()}}
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def test_export_whitelists_records_and_never_overwrites_existing_results(tmp_path):
    source, target = tmp_path / "run", tmp_path / "public"
    fixture_run(source)
    export_run(source, target)
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    assert all(b"Fixture text must stay private" not in content for content in before.values())
    with pytest.raises(FileExistsError):
        export_run(source, target)
    assert before == {p.name: p.read_bytes() for p in target.iterdir()}


@pytest.mark.parametrize("tamper", ["summary", "records", "protocol"])
def test_tampered_run_never_exports_successful_public_results(tmp_path, tamper):
    source, target = tmp_path / "run", tmp_path / "public"
    summary = fixture_run(source)
    if tamper == "summary":
        summary["methods"]["hybrid"]["metrics"]["5"]["hit"] = 0
        (source / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    else:
        name = "per_question.jsonl" if tamper == "records" else "protocol.json"
        with (source / name).open("a", encoding="utf-8") as stream:
            stream.write(" ")
    with pytest.raises(ValueError):
        export_run(source, target)
    assert not target.exists()
