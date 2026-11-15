from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.evaluation.chunking_comparison import summarize_comparison, verify_source_coverage


def population(delta=.25, structure_hit=1, structure_mrr=1):
    records = []
    for i in range(4):
        strategies = {}
        for name in ("adaptive", "structure"):
            metrics = dict(hit=structure_hit if name == "structure" else 1,
                mrr=structure_mrr if name == "structure" else 1,
                recall=.5, ndcg=.5, full_paragraph_recall=.5, evidence_f1=.5)
            context = dict(metrics, full_paragraph_recall=.5 + (delta if name == "structure" else 0),
                           all_gold_covered=0, characters=1000)
            strategies[name] = {"methods": {m: {"5": dict(metrics)} for m in ("bm25", "dense", "hybrid")},
                                "contexts": {v: dict(context) for v in ("top5", "anchor_first")}}
        records.append({"paper_id": f"p{i//2}", "question_id": f"q{i}", "strategies": strategies})
    return records


def test_recommendation_requires_positive_complete_paragraph_gain():
    result = summarize_comparison(population())
    paired = result["paired_structure_minus_adaptive"]["contexts"]["anchor_first"]["full_paragraph_recall"]
    assert paired["low"] == paired["high"] == .25
    assert result["recommended_for_followup"] == "structure"
    assert summarize_comparison(population(delta=0))["recommended_for_followup"] == "adaptive"
    assert result["hybrid_hit_transitions_at_5"] == dict(both_hit=4, structure_only_hit=0, adaptive_only_hit=0, both_miss=0)


@pytest.mark.parametrize("kwargs", [dict(structure_hit=.5), dict(structure_mrr=.5)])
def test_better_source_coverage_does_not_override_retrieval_guardrails(kwargs):
    assert summarize_comparison(population(**kwargs))["recommended_for_followup"] == "adaptive"


def test_duplicate_or_empty_questions_cannot_be_reported_as_a_complete_comparison():
    with pytest.raises(ValueError):
        summarize_comparison([])
    rows = population()
    with pytest.raises(ValueError):
        summarize_comparison(rows + [rows[0]])


def test_source_check_requires_headings_and_non_gold_text_and_exact_source_slices():
    document = SimpleNamespace(blocks=[SimpleNamespace(block_id="h1", text="Heading"),
        SimpleNamespace(block_id="p1", text="First para"), SimpleNamespace(block_id="p2", text="Other")])
    def chunk(ids):
        texts = {b.block_id: b.text for b in document.blocks}
        return {"text": "\n\n".join(texts[k] for k in ids), "source_metadata": {"spans": [
            dict(block_id=k, char_start=0, char_end=len(texts[k])) for k in ids]}}
    chunks = [chunk(["h1"]), chunk(["p1", "p2"])]
    assert verify_source_coverage(document, chunks) == dict(blocks=3, paragraphs=2, merged_paragraph_chunks=1)
    with pytest.raises(ValueError, match="lost"):
        verify_source_coverage(document, chunks[1:])
    altered = deepcopy(chunks)
    altered[1]["text"] = "Wrong source"
    with pytest.raises(ValueError, match="does not match"):
        verify_source_coverage(document, altered)
    altered = deepcopy(chunks)
    altered[1]["source_metadata"]["spans"][0]["char_end"] += 100
    with pytest.raises(ValueError, match="Invalid"):
        verify_source_coverage(document, altered)
