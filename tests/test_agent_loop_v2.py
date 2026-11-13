import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from src.agent.context_window import validate_turn
from src.agent.research_agent import ResearchAgent
from src.llm.provider import Completion, ModelError, ToolCall
from tests.chat_helpers import FakeProvider, make_resources, tool_answer


@pytest.fixture
def resource(tmp_path):
    item = make_resources(tmp_path / "catalog")
    yield item
    item.store.close()


def test_parallel_tool_overflow_has_one_result_for_every_call(resource):
    arguments = json.dumps({"paper_id": resource.document.document_id})
    calls = tuple(ToolCall(f"call{i}", "get_paper_outline", arguments) for i in range(3))
    provider = FakeProvider(resource.config, [Completion(tool_calls=calls), Completion("These are the real sections.")])
    result = ResearchAgent(provider, resource.retriever).run("outline")
    validate_turn(result["messages"])
    assert len([m for m in result["messages"] if m["role"] == "tool"]) == 3
    assert result["tool_calls"][-1]["error_code"] == "tool_budget_exceeded"
    assert provider.requests[1]["tools"] is None


@pytest.mark.parametrize("name,args,code", [
    ("read_api_key", {"paper_id": "x"}, "unknown_tool"),
    ("get_paper_outline", {"paper_id": "other"}, "document_scope_violation"),
    ("get_paper_outline", {"paper_id": "placeholder", "extra": "injected"}, "invalid_tool_arguments"),
])
def test_unregistered_or_out_of_scope_tools_never_execute(resource, name, args, code):
    provider = FakeProvider(resource.config, [Completion(tool_calls=(ToolCall("call", name, json.dumps(args)),)),
                                            Completion("Cannot answer.")])
    with pytest.raises(ModelError, match="没有取得可用工具结果"):
        ResearchAgent(provider, resource.retriever).run("question")
    assert code in provider.requests[1]["messages"][-1]["content"]


def test_context_overflow_prevents_model_request(resource):
    config = replace(resource.config, context_window=512, max_output_tokens=128, safety_margin=128)
    provider = FakeProvider(config, [])
    with pytest.raises(ModelError, match="输入预算"):
        ResearchAgent(provider, resource.retriever).run("question" * 200)
    assert not provider.requests


def test_known_citations_resend_evidence_after_history_is_dropped(resource):
    first_provider = tool_answer(resource)
    first = ResearchAgent(first_provider, resource.retriever).run("momentum")
    old_turn = [{"role": "user", "content": "old question " * 10000},
                {"role": "assistant", "content": "old answer"}]
    provider = tool_answer(resource)
    second = ResearchAgent(provider, resource.retriever, sources=first["registered_sources"]).run("momentum", [old_turn])
    assert second["context_selections"][0]["dropped_turns"] == 1
    assert "Momentum uses a position update" in provider.requests[1]["messages"][-1]["content"]
    assert second["cited_numbers"] == first["cited_numbers"]


def test_direct_answer_without_a_successful_tool_result_is_rejected(resource):
    provider = FakeProvider(resource.config, [Completion("Unsupported direct answer.")])
    with pytest.raises(ModelError) as error:
        ResearchAgent(provider, resource.retriever).run("question")
    assert error.value.code == "no_evidence" and len(provider.requests) == 1


def test_malformed_json_returns_paired_error_and_model_can_recover_within_budget(resource):
    provider = tool_answer(resource, call_id="valid_search")
    provider.responses.insert(0, Completion(tool_calls=(ToolCall("invalid_search", "search_research_paper", "{"),)))
    result = ResearchAgent(provider, resource.retriever).run("question")
    validate_turn(result["messages"])
    error = json.loads(provider.requests[1]["messages"][-1]["content"])
    assert error["error"] == "invalid_tool_arguments"
    assert provider.requests[1]["messages"][-1]["tool_call_id"] == "invalid_search"
    assert [item["status"] for item in result["tool_calls"]] == ["failed", "completed"]
    assert result["model_calls"] == 3 and provider.requests[-1]["tools"] is None
    assert result["cited_numbers"] == [1]


def test_model_repeating_tool_requests_stops_at_round_limit_and_executes_only_two(resource):
    replies = [Completion(tool_calls=(ToolCall(f"call_{i}", "get_paper_outline", json.dumps({
        "paper_id": resource.document.document_id})),)) for i in range(5)]
    provider = FakeProvider(resource.config, replies)
    agent = ResearchAgent(provider, resource.retriever)
    agent._execute = Mock(wraps=agent._execute)
    with pytest.raises(ModelError) as error:
        agent.run("outline")
    assert error.value.code == "agent_round_limit"
    assert len(provider.requests) == agent.max_rounds == 5
    assert agent._execute.call_count == agent.max_tool_calls == 2
    assert all(request["tools"] is None for request in provider.requests[2:])
    assert json.loads(provider.requests[-1]["messages"][-1]["content"])["error"] == "tool_budget_exceeded"


def test_tool_exception_does_not_echo_private_exception_body(resource, monkeypatch):
    provider = tool_answer(resource)
    agent = ResearchAgent(provider, resource.retriever)
    monkeypatch.setattr(agent.search, "run", Mock(side_effect=RuntimeError("private credential or document path")))
    with pytest.raises(ModelError) as error:
        agent.run("question")
    assert error.value.code == "no_evidence"
    payload = provider.requests[1]["messages"][-1]
    assert payload["tool_call_id"] == "call1" and json.loads(payload["content"])["error"] == "tool_failed"
    assert "private credential" not in payload["content"]
