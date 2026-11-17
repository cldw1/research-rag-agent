"""Evidence identity, parsing misses and annotation/generation isolation."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from src.evaluation.pdf_benchmark import (
    SCHEMA_VERSION, align_evidence, evidence_coverage, load_dataset, native_blocks,
    retrieval_metrics, sha256, validate_dataset, verify_sources, verify_candidate_release,
)
from src.ingestion.parsers import UnifiedDocumentParser
from src.ingestion.structured_chunker import ByteBudgetCounter
from src.storage.embedding_profile import EmbeddingProfile
from tests.test_document_store import FixtureEmbedding
from tests.test_structured_ingestion import make_pdf


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample(tmp_path):
    content = make_pdf(["Momentum uses a position update.", "Experiment measures accuracy."])
    path = tmp_path / "sample.pdf"
    path.write_bytes(content)
    native, pages, engine = native_blocks(path)
    position = next(k for k in native if k[0] == 1)
    block = native[position]
    anchor = {"evidence_id": "e1", "page_number": position[0], "block_index": position[1],
              "bbox": block["bbox"], "char_start": 0, "char_end": len(block["text"]),
              "text_sha256": sha256(block["text"].encode()), "section": "Methods"}
    question = {"question_id": "q1", "paper_id": "p", "question": "What does momentum use?",
                "category": "fact", "answerable": True, "reference_points": ["gold-answer-not-in-prompt"],
                "evidence": [anchor], "annotation": {"human_review": "pending"}}
    paper = {"paper_id": "p", "title": "Fixture study", "arxiv_version": "0000.00000v1",
             "split": "dev", "filename": path.name,
             "pdf_sha256": sha256(content), "page_count": pages, "locator_engine": engine}
    dataset = {"schema_version": SCHEMA_VERSION, "status": "development_pilot",
               "papers": [paper], "questions": [question]}
    document = UnifiedDocumentParser().parse(content, path.name)
    return dataset, native, document, tmp_path


def chunk(block, start, end):
    return {"source_metadata": {"spans": [{"block_id": block, "char_start": start, "char_end": end}]}}


def test_every_source_locator_is_verified_against_real_pdf(sample):
    data, _, _, folder = sample
    _, report = verify_sources(data, folder)
    assert report["verified_evidence_locators"] == 1


@pytest.mark.parametrize("field,value", [
    ("page_number", 2), ("char_end", 200), ("text_sha256", "0"*64), ("bbox", [0., 0., 1., 1.]),
])
def test_stale_or_forged_evidence_fails_before_retrieval(sample, field, value):
    data, _, _, folder = sample
    data["questions"][0]["evidence"][0][field] = value
    with pytest.raises(ValueError):
        verify_sources(data, folder)


def test_changed_pdf_cannot_reuse_old_labels(sample):
    data, _, _, folder = sample
    (folder / "sample.pdf").write_bytes(make_pdf(["Different paper."]))
    with pytest.raises(ValueError, match="checksum"):
        verify_sources(data, folder)


def test_duplicate_questions_and_split_leaks_are_rejected(sample):
    data, _, _, _ = sample
    data["questions"].append(deepcopy(data["questions"][0]))
    with pytest.raises(ValueError, match="Duplicate question"):
        validate_dataset(data)
    data["questions"].pop()
    data["papers"][0]["split"] = "random-question-split"
    with pytest.raises(ValueError, match="split"):
        validate_dataset(data)


def test_partial_gold_ranges_and_gaps_do_not_count_as_complete(sample):
    data, native, doc, _ = sample
    aligned = align_evidence(data["questions"][0], native, doc)
    block, start, end = aligned[0]["span"]
    pieces = [chunk(block, start, start+5), chunk(block, start+6, end)]
    scores = evidence_coverage(pieces, aligned)
    assert scores == {"evidence_recall": 1., "full_evidence_recall": 0., "all_evidence_covered": 0.}
    assert evidence_coverage(pieces + [chunk(block, start+5, start+6)], aligned)["full_evidence_recall"] == 1


def test_parse_loss_stays_in_retrieval_denominator(sample):
    data, native, doc, _ = sample
    lost = replace(doc, blocks=tuple(b for b in doc.blocks if b.location.page_number != 1))
    aligned = align_evidence(data["questions"][0], native, lost)
    assert aligned[0]["status"] == "missing"
    all_chunks = [chunk(lost.blocks[0].block_id, 0, len(lost.blocks[0].text))]
    result = retrieval_metrics(all_chunks, all_chunks, aligned, 5)
    assert all(value == 0 for value in result.values())


def test_duplicate_text_at_same_location_is_ambiguous_not_arbitrarily_mapped(sample):
    data, native, doc, _ = sample
    repeated = replace(doc.blocks[0], block_id="duplicate")
    doc = replace(doc, blocks=doc.blocks + (repeated,))
    assert align_evidence(data["questions"][0], native, doc)[0]["status"] == "ambiguous"


def test_same_text_on_different_page_is_not_gold(sample):
    data, native, doc, _ = sample
    other = replace(doc.blocks[0], block_id="other", location=replace(doc.blocks[0].location, page_number=2))
    doc = replace(doc, blocks=(other,))
    assert align_evidence(data["questions"][0], native, doc)[0]["status"] == "missing"


def test_mrr_ndcg_and_precision_use_rank_and_fixed_k(sample):
    data, native, doc, _ = sample
    aligned = align_evidence(data["questions"][0], native, doc)
    block, start, end = aligned[0]["span"]
    relevant, other = chunk(block, start, end), chunk("outside", 0, 10)
    result = retrieval_metrics([relevant, other], [other, relevant], aligned, 5)
    assert result["hit"] == 1 and result["precision"] == .2 and result["mrr"] == .5
    assert result["ndcg"] == pytest.approx(1 / __import__('math').log2(3))


def test_unanswerable_is_not_zero_scored_retrieval(sample):
    data, _, _, _ = sample
    question = data["questions"][0]
    question.update(answerable=False, category="unanswerable", evidence=[], absence_review={"reason": "not reported"})
    validate_dataset(data)
    with pytest.raises(ValueError, match="undefined"):
        evidence_coverage([], [])


def test_public_pilot_has_two_dev_papers_and_30_provisional_questions():
    dataset = load_dataset(ROOT / "benchmarks/pdf_qa/pilot.json")
    assert len(dataset["papers"]) == 2 and len(dataset["questions"]) == 30
    assert {p["split"] for p in dataset["papers"]} == {"dev"}
    assert sum(q["answerable"] for q in dataset["questions"]) == 28
    assert all(q["annotation"]["human_review"] == "pending" for q in dataset["questions"])
    assert any(q["category"] == "source_conflict" for q in dataset["questions"])
    assert all("text" not in a for q in dataset["questions"] for a in q["evidence"])


def test_candidate_release_has_complete_disjoint_paper_splits_and_preserves_pilot():
    report = verify_candidate_release(ROOT / "benchmarks/pdf_qa")
    assert report == {"papers": 10, "questions": 150, "pages": 179,
        "splits": {"dev": {"papers": 3, "questions": 45}, "test": {"papers": 7, "questions": 105}},
        "answerable": 140, "evidence_locators": 205, "human_review_pending": 150, "final_test_ready": False}


@pytest.mark.parametrize("same_content", [True, False])
def test_duplicate_pdf_or_arxiv_versions_cannot_cross_split_boundaries(sample, same_content):
    data, _, _, _ = sample
    other = {**data["papers"][0], "paper_id": "duplicate", "split": "test", "filename": "other.pdf"}
    if not same_content:
        other.update(pdf_sha256="0"*64, arxiv_version="0000.00000v2")
    data["papers"].append(other)
    with pytest.raises(ValueError, match="Duplicate PDF|arXiv"):
        validate_dataset(data)


def test_candidate_release_detects_derived_view_drift_even_after_rehash(tmp_path):
    import shutil
    source = ROOT / "benchmarks/pdf_qa"
    for name in ("candidate_manifest.json", "selection.json", "corpus.json", "development.json",
                 "test_candidates.json", "development_protocol.json", "pilot.json"):
        shutil.copyfile(source / name, tmp_path / name)
    subset = json.loads((tmp_path / "test_candidates.json").read_text(encoding="utf-8"))
    subset["questions"][0]["question"] = "A substituted test question"
    (tmp_path / "test_candidates.json").write_text(json.dumps(subset), encoding="utf-8")
    manifest = json.loads((tmp_path / "candidate_manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts_sha256"]["test_candidates.json"] = sha256((tmp_path / "test_candidates.json").read_bytes())
    (tmp_path / "candidate_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Derived split"):
        verify_candidate_release(tmp_path)


@pytest.mark.parametrize("name", ["test_candidates.json", "corpus.json"])
def test_unreviewed_test_or_mixed_corpus_stops_before_models_or_source_parsing(tmp_path, monkeypatch, name):
    import scripts.run_pdf_benchmark as runner
    import src.embedding
    def forbidden(*args, **kwargs):
        raise AssertionError("Test candidate must never reach parsing/embedding/retrieval")
    monkeypatch.setattr(runner, "verify_sources", forbidden)
    monkeypatch.setattr(src.embedding, "load_embedding_resources", forbidden)
    output = tmp_path / "no-test-run"
    monkeypatch.setattr(sys, "argv", ["run_pdf_benchmark.py", "--dataset", str(ROOT / "benchmarks/pdf_qa" / name),
                                     "--output-root", str(output)])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2 and not output.exists()


def test_offline_runner_uses_real_store_and_keeps_gold_out_of_prompts(sample, monkeypatch):
    from scripts.run_pdf_benchmark import main
    import src.embedding

    data, _, _, folder = sample
    path = folder / "dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    protocol = json.loads((ROOT / "benchmarks/pdf_qa/protocol.json").read_text(encoding="utf-8"))
    protocol["embedding"]["dimension"] = 3
    protocol_path = folder / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    counter = ByteBudgetCounter()
    model = FixtureEmbedding()
    profile = EmbeddingProfile("BAAI/bge-small-en-v1.5", "test-only", 3, counter.identity, 512)
    monkeypatch.setattr(src.embedding, "load_embedding_resources", lambda **kwargs: (model, counter, profile))
    monkeypatch.setattr(sys, "argv", ["run_pdf_benchmark.py", "--dataset", str(path), "--sources", str(folder),
        "--protocol", str(protocol_path), "--output-root", str(folder / "runs")])
    main()
    run = next((folder / "runs").iterdir())
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    requests = [json.loads(line) for line in (run / "generation_requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["generation_api_calls"] == 0 and summary["answer_metrics"] is None
    assert summary["alignment"] == {"mapped": 1}
    assert len(requests) == 2
    assert all("gold-answer-not-in-prompt" not in r["prompt"] for r in requests)
    assert all("reference_points" not in r and "answerable" not in r for r in requests)
    assert all(sha256((run / "source_snapshot" / name).read_bytes()) == value
               for name, value in summary["source_code_sha256"].items())
    from scripts.export_pdf_results import export_result
    exported = export_result(run, folder / "public.json")
    assert exported["question_count"] == 1 and exported["human_review_pending"] == 1
    with pytest.raises(FileExistsError):
        export_result(run, folder / "public.json")
    summary["answerable_count"] += 1
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="denominator"):
        export_result(run, folder / "forged.json")
    (run / "records.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        export_result(run, folder / "corrupt.json")
