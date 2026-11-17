from dataclasses import replace
from pathlib import Path

import pytest

from src.agent.research_agent import ResearchAgent
from src.chat_service import ResearchChatService
from src.generation.answer_policy import ANSWER_POLICY_VERSION, ANSWER_GROUNDING_RULES
from src.generation.citations import extract_citation_numbers
from src.generation.rag_pipeline import ResearchRAGPipeline
from src.llm.provider import Completion
from tests.chat_helpers import FakeProvider, make_resources, tool_answer


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('text,numbers', [
    ('Claim [1, 2] and another [2][4].', [1, 2, 4]),
    ('Claim [1 3] and [ 2 ].', [1, 2, 3]),
    ('[1-3] is not a supported citation range; [n] is a placeholder.', []),
])
def test_numeric_citation_groups_are_read_without_inventing_ranges(text, numbers):
    assert extract_citation_numbers(text) == numbers


@pytest.fixture
def resource(tmp_path):
    item = make_resources(tmp_path/'catalog')
    yield item
    item.store.close()


@pytest.mark.parametrize('tools', [False, True])
def test_actual_outgoing_prompt_and_saved_result_share_answer_policy(resource, tools):
    config = replace(resource.config, supports_tools=tools)
    answer = 'Supported evidence [1, 2], invalid source [1, 99].'
    llm = tool_answer(resource, answer=answer) if tools else FakeProvider(config, [Completion(answer)])
    llm.config = config
    cid = resource.conversations.create(resource.document.document_id, config)
    result = ResearchChatService(resource.conversations).ask(cid, 'momentum experiment', llm,
                                                           resource.retriever, 'grounded', answer_policy='grounded')
    assert ANSWER_GROUNDING_RULES in llm.requests[0]['messages'][0]['content']
    assert result['answer_policy_version'] == ANSWER_POLICY_VERSION
    assert result['cited_numbers'] == [1, 2, 99] and result['invalid_citations'] == [99]
    assert [s['citation'] for s in result['cited_sources']] == [1, 2]
    saved = resource.conversations.history(cid)[0]['result']
    assert saved['answer_policy_version'] == ANSWER_POLICY_VERSION
    assert saved['cited_numbers'] == result['cited_numbers']


def test_legacy_pipeline_generation_reads_grouped_citations_and_keeps_source_mapping(resource, monkeypatch):
    pipeline = ResearchRAGPipeline(resource.retriever, answer_policy='grounded')
    prepared = pipeline.prepare('momentum experiment', resource.document.document_id)
    assert prepared['answer_policy_version'] == ANSWER_POLICY_VERSION
    assert ANSWER_GROUNDING_RULES in prepared['prompt']
    monkeypatch.setattr('src.generation.rag_pipeline.generate_answer', lambda prompt: 'A claim [1, 2, 99].')
    result = pipeline.generate(prepared)
    assert result['cited_numbers'] == [1, 2, 99]
    assert result['invalid_citations'] == [99]
    assert [s['citation'] for s in result['valid_citations']] == [1, 2]


def test_default_rule_and_idempotent_turn_keep_the_original_policy(resource):
    service = ResearchChatService(resource.conversations)
    cid = resource.conversations.create(resource.document.document_id, resource.config)
    llm = tool_answer(resource)
    first = service.ask(cid, 'momentum', llm, resource.retriever, 'same')
    assert first['answer_policy_version'] == 'legacy-evidence-v1'
    assert ANSWER_GROUNDING_RULES not in llm.requests[0]['messages'][0]['content']
    unused = tool_answer(resource)
    assert service.ask(cid, 'momentum', unused, resource.retriever, 'same', answer_policy='grounded') == first
    assert not unused.requests
    with pytest.raises(ValueError, match='answer policy'):
        service.ask(cid, 'momentum', unused, resource.retriever, 'invalid', answer_policy='unsupported')
    assert len(resource.conversations.history(cid)) == 1


@pytest.mark.skipif(not (ROOT/'data/pdf_benchmark/multidoc_answer_runs/20261116T124957Z-914040f8/requests.json').exists(),
                    reason='Optional local frozen answer inputs')
def test_paired_regression_uses_exact_same_evidence_and_no_gold_in_generation():
    from scripts.run_answer_policy_check import prepare_requests
    protocol, requests, policy = prepare_requests(ROOT)
    assert len(requests) == protocol['max_api_calls'] == 12
    assert set(protocol['known_failure_ids']) | set(protocol['control_ids']) == set(protocol['question_ids'])
    for index in range(0, 12, 2):
        a, b = requests[index:index+2]
        assert a['comparison_question_id'] == b['comparison_question_id']
        assert a['sources'] == b['sources'] and a['source_prompt_sha256'] == b['source_prompt_sha256']
        assert b['messages'][0]['content'] == f"Answer policy: {policy['version']}\n{policy['rules']}\n\n"+a['messages'][0]['content']
        assert a['prompt_sha256'] != b['prompt_sha256']
        assert not {'gold_evidence', 'references', 'reference_points', 'answerable'} & set(a)
        assert not {'gold_evidence', 'references', 'reference_points', 'answerable'} & set(b)
