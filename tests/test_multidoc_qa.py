from copy import deepcopy
from pathlib import Path

import pytest

from src.evaluation.multidoc_qa import audit_attribution, resolve_cases, score_context
from src.evaluation.pdf_answers import read_json


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_questions_join_only_frozen_development_sources():
    spec = read_json(ROOT / "benchmarks/multidoc_qa/questions.json")
    dataset = read_json(ROOT / "benchmarks/pdf_qa/development.json")
    cases = resolve_cases(spec, dataset)
    assert len(cases) == 12
    assert all(len(c["targets"]) >= 2 for c in cases)
    assert sum(all(not t["answerable"] for t in c["targets"]) for c in cases) == 2
    changed = deepcopy(dataset)
    changed["papers"][0]["split"] = "test"
    with pytest.raises(ValueError, match="Development"):
        resolve_cases(spec, changed)
    changed = deepcopy(spec)
    changed["questions"][0]["source_question_ids"] *= 2
    with pytest.raises(ValueError, match="different"):
        resolve_cases(changed, dataset)


def test_identical_block_ids_in_another_paper_cannot_cover_gold():
    chunks = [{"chunk_id":"b0", "paper_id":"B", "source_metadata":{"spans":[
        {"block_id":"same-block", "char_start":0, "char_end":20}]}}]
    sources = [{"citation":1,"chunk_id":"b0","paper_id":"B"}]
    targets = [{"question_id":"a", "paper_id":"a", "answerable":True},
               {"question_id":"b", "paper_id":"b", "answerable":False}]
    score = score_context(sources,chunks,targets,{"a":"A","b":"B"},
        {"a":[{"status":"mapped","span":["same-block",0,20]}]})
    assert score["per_paper_gold"]["a"]["evidence_recall"] == 0
    assert score["all_required_papers_hit"] == 0 and score["document_presence"] == .5
    targets[0]["answerable"] = False
    assert score_context(sources,chunks,targets,{"a":"A","b":"B"},{})["all_gold_complete"] is None


@pytest.mark.parametrize("citations,expected,status", [([1],["A"],"matched"),
    ([1],["B"],"wrong_document"),([1,2],["A"],"wrong_document"),
    ([1,2],["A","B"],"matched"),([],["A"],"missing_citation"),([9],["A"],"unknown_citation")])
def test_reviewer_claim_labels_detect_swapped_or_extra_paper_citations(citations,expected,status):
    sources=[{"citation":1,"paper_id":"A"},{"citation":2,"paper_id":"B"}]
    result=audit_attribution([{"claim_id":"c1","expected_document_ids":expected,"citations":citations}],sources,["A","B"])
    assert result["claims"][0]["status"] == status and result["semantic_faithfulness"] is None


def test_unselected_document_citations_are_not_validated_by_a_real_number():
    result=audit_attribution([{"claim_id":"c1","expected_document_ids":["A"],"citations":[1]}],
        [{"citation":1,"paper_id":"private-other"}],["A"])
    assert result["claims"][0]["status"] == "out_of_scope"
