"""Independent checks of evidence alignment, denominators and clustered sampling."""

from copy import deepcopy
import math

import pytest

from src.evaluation.qasper_benchmark import (
    EvidenceQuestion, evidence_coverage, fully_covered, paper_bootstrap,
    prepare_dataset, retrieval_metrics, structured_paper,
)


def paper(evidence=None):
    return {"id": "paper", "title": "Study", "full_text": {"section_name": ["Methods"],
            "paragraphs": [["First paragraph.", "Second paragraph."]]},
            "qas": {"question_id": ["q"], "question": ["Which method?"], "answers": [
                {"answer": [{"unanswerable": False, "evidence": evidence or ["First paragraph."]}]}]}}


def chunk(key, start=0, end=10):
    return {"source_metadata": {"spans": [{"block_id": key, "char_start": start, "char_end": end}]}}


def test_annotations_never_enter_document_or_change_its_version():
    original = paper()
    changed = deepcopy(original)
    changed["qas"]["question"] = ["A completely different query"]
    changed["qas"]["answers"][0]["answer"][0]["evidence"] = ["Second paragraph."]
    assert structured_paper(original, "test") == structured_paper(changed, "test")


def test_evidence_normalizes_unicode_whitespace_but_does_not_use_fuzzy_matching():
    row = paper([" First\nparagraph. "])
    _, questions, counts, excluded = prepare_dataset([row], "test")
    assert questions[0].gold == {"p0000_0000"} and not excluded
    row["qas"]["answers"][0]["answer"][0]["evidence"] = ["First paragraph changed."]
    _, questions, counts, excluded = prepare_dataset([row], "test")
    assert not questions and counts["unmapped_text_evidence"] == 1


def test_duplicate_text_positions_are_excluded_instead_of_guessing():
    row = paper()
    row["full_text"]["paragraphs"][0].append("First paragraph.")
    _, questions, counts, _ = prepare_dataset([row], "test")
    assert not questions and counts["ambiguous_text_evidence"] == 1


@pytest.mark.parametrize("answer,reason", [
    ({"unanswerable": True, "evidence": []}, "unanswerable"),
    ({"unanswerable": False, "evidence": ["FLOAT SELECTED: figure"]}, "no_text_evidence"),
])
def test_unanswerable_and_visual_only_questions_have_explicit_exclusions(answer, reason):
    row = paper()
    row["qas"]["answers"] = [{"answer": [answer]}]
    _, questions, counts, excluded = prepare_dataset([row], "test")
    assert not questions and counts[reason] == 1 and excluded[0]["reason"] == reason


def test_recall_deduplicates_gold_paragraphs_and_precision_keeps_k_denominator():
    q = EvidenceQuestion("p", "q", "query", frozenset({"p1", "p2"}), (frozenset({"p1", "p2"}),))
    chunks = [chunk("p0"), chunk("p1", 0, 5), chunk("p1", 5, 10), chunk("p2")]
    scores = retrieval_metrics(chunks, chunks[:3], q, {"p1": 10, "p2": 10}, 5)
    assert scores["precision"] == .4 and scores["recall"] == .5 and scores["mrr"] == .5
    assert scores["full_paragraph_recall"] == .5 and scores["all_gold_covered"] == 0
    expected = (1/math.log2(3) + 1/math.log2(4)) / (1 + 1/math.log2(3) + 1/math.log2(4))
    assert scores["ndcg"] == pytest.approx(expected)


def test_full_coverage_requires_no_gap_and_alternative_reference_f1_is_separate():
    assert not fully_covered([(0, 4), (5, 10)], 10)
    assert fully_covered([(0, 6), (4, 10)], 10)
    q = EvidenceQuestion("p", "q", "query", frozenset({"p1", "p2"}),
                         (frozenset({"p1"}), frozenset({"p2"})))
    scores = evidence_coverage([chunk("p1", 0, 4)], q, {"p1": 10, "p2": 10})
    assert scores["recall"] == .5 and scores["full_paragraph_recall"] == 0
    assert scores["evidence_f1"] == 1


def test_cluster_bootstrap_preserves_question_weighting_and_seed():
    records = [{"paper_id": "a", "score": 1}, {"paper_id": "a", "score": 1},
               {"paper_id": "b", "score": 0}]
    result = paper_bootstrap(records, lambda row: row["score"], resamples=100)
    assert result["mean"] == pytest.approx(2/3)
    assert result == paper_bootstrap(records, lambda row: row["score"], resamples=100)
    assert result["low"] <= result["mean"] <= result["high"]


def test_duplicate_question_ids_fail_before_evaluation():
    first = paper()
    second = deepcopy(first)
    second["id"] = "other"
    with pytest.raises(ValueError, match="question ID"):
        prepare_dataset([first, second], "test")
