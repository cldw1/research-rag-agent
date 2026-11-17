"""Review provenance and export replay; semantic scores never derive from citation format."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.export_pdf_answers import export_answers
from src.evaluation.answer_review import summarize_reviews, validate_reviews
from src.evaluation.pdf_answers import generate_one, write_new
from src.evaluation.pdf_benchmark import native_blocks, sha256
from src.llm.provider import Completion, ModelConfig
from tests.test_structured_ingestion import make_pdf


ROOT = Path(__file__).resolve().parents[1]
RUBRIC = json.loads((ROOT / "benchmarks/pdf_qa/answer_review_rubric.json").read_text(encoding="utf-8"))


def reviewed():
    pack = [{"question_id": "a", "answer_sha256": "hash-a", "cited_numbers": [1], "answerable": True},
            {"question_id": "b", "answer_sha256": "hash-b", "cited_numbers": [], "answerable": False}]
    review = {"rubric_version": RUBRIC["rubric_version"], "reviewer_type": "ai_source_audit",
              "reviewer": "AI fixture", "reviewed_at": "2026-11-16", "independent_human_review": False,
              "records": [{"question_id": "a", "answer_sha256": "hash-a", "correctness": 1,
                "faithfulness": 2, "citation_support": 1, "citation_coverage": 1,
                "checked_citations": [1], "failure_categories": ["incomplete_context"], "reason": "Partial answer"},
                {"question_id": "b", "answer_sha256": "hash-b", "correctness": 2,
                 "faithfulness": 2, "citation_support": None, "citation_coverage": None,
                 "checked_citations": [], "failure_categories": [], "reason": "Explicit absence response"}]}
    return pack, review


def test_ai_source_scores_are_separate_from_human_metrics_and_applicability():
    pack, review = reviewed()
    result = summarize_reviews(pack, review, RUBRIC)
    assert result["complete"] and result["human_metrics"] is None
    assert result["metrics"]["citation_support"]["applicable"] == 1
    assert result["metrics"]["correctness"]["applicable"] == 2
    assert result["unanswerable_fully_correct"] == 1


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r.update(independent_human_review=True), "AI source"),
    (lambda r: r["records"][0].update(answer_sha256="changed"), "changed"),
    (lambda r: r["records"][0].update(correctness=True), "score"),
    (lambda r: r["records"][0].update(checked_citations=[]), "Citation audit"),
    (lambda r: r["records"].append(deepcopy(r["records"][0])), "Duplicate"),
])
def test_review_cannot_relabel_provenance_changed_answers_or_partial_citation_checks(mutation, reason):
    pack, review = reviewed()
    mutation(review)
    with pytest.raises(ValueError, match=reason):
        validate_reviews(pack, review, RUBRIC)


def test_export_replays_fixed_attempts_without_network_and_keeps_raw_text_private(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    pdf = source / "fixture.pdf"
    pdf.write_bytes(make_pdf(["Momentum uses a position update."]))
    native, pages, engine = native_blocks(pdf)
    (page, index), block = next(iter(native.items()))
    anchor = {"evidence_id": "e1", "page_number": page, "block_index": index,
        "bbox": block["bbox"], "char_start": 0, "char_end": len(block["text"]),
        "text_sha256": sha256(block["text"].encode()), "section": "Methods"}
    paper = {"paper_id": "p", "split": "dev", "filename": "fixture.pdf", "pdf_sha256": sha256(pdf.read_bytes()),
             "page_count": pages, "locator_engine": engine}
    qs = [{"question_id": f"q{i:02}", "paper_id": "p", "question": "What does momentum use?",
           "category": "fact", "answerable": True, "reference_points": ["A position update"],
           "evidence": [anchor], "annotation": {"human_review": "pending"}} for i in range(45)]
    dataset = {"schema_version": "pdf-qa-v1", "status": "development_pilot", "papers": [paper], "questions": qs}
    run, retrieval = tmp_path / "answers", tmp_path / "retrieval"
    write_new(run / "dataset.json", dataset)
    write_new(retrieval / "dataset.json", dataset)
    sources = [{"citation": 1, "chunk_id": "c1", "paper_id": "p", "page_number": 1}]
    requests = [{"question_id": q["question_id"], "paper_id": "p", "variant": "anchor_first",
                 "messages": [{"role": "user", "content": "Only source text"}], "input_estimate": 100,
                 "prompt_sha256": "fixture", "sources": sources} for q in qs]
    write_new(run / "requests.json", requests)
    config = ModelConfig()
    manifest = {"question_ids": [q["question_id"] for q in qs], "variant": "anchor_first", "model_config": config.to_dict(),
        "thinking": "disabled", "retrieval_run_id": "fixture", "price_reference": {}, "transport": {},
        "dataset_sha256": sha256((run / "dataset.json").read_bytes()),
        "requests_sha256": sha256(json.dumps(requests, sort_keys=True, ensure_ascii=False).encode())}
    write_new(run / "manifest.json", manifest)
    write_new(retrieval / "chunks.json", {"p": [{"chunk_id": "c1", "text": block["text"]}]})
    write_new(retrieval / "records.json", [{"question_id": q["question_id"], "contexts": {"anchor_first": {"metrics": {"evidence_recall": 1}}}} for q in qs])
    provider = SimpleNamespace(config=config, complete=Mock(return_value=Completion("A position update [1].",
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}, finish_reason="stop")))
    for request in requests:
        generate_one(run, request, provider)
    assert provider.complete.call_count == 45
    report = export_answers(run, retrieval, source, tmp_path / "public.json")
    assert provider.complete.call_count == 45  # Export never regenerates answers.
    assert report["generation_api_attempts"] == 45 and report["known_returned_usage"]["total_tokens"] == 630
    assert report["source_review"] is None and report["answer_quality_metrics"] is None
    assert block["text"] not in (tmp_path / "public.json").read_text(encoding="utf-8")
    assert (run / "human_answer_review.csv").exists()
    with pytest.raises(FileExistsError):
        export_answers(run, retrieval, source, tmp_path / "public.json")
