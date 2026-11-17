"""Online retry uses fixed responses, real scoped storage and stable citations."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from src.agent.context_window import validate_turn
from src.agent.paper_search_tool import PaperSearchTool
from src.chat_service import ResearchChatService
from src.generation.strict_context_builder import build_strict_context
from src.generation_task import GenerationTask
from src.llm.control import CancellationToken, GenerationCancelled
from src.llm.provider import Completion, ModelError
from src.retrieval.evidence_retry import EvidenceRetryPolicy
from tests.chat_helpers import FakeProvider, make_resources, tool_answer


POLICY = EvidenceRetryPolicy(enabled=True, preserve_citations=True, balance_novel=True,
                             version="cited-evidence-novel-balance-v2")


@pytest.fixture
def resource(tmp_path, monkeypatch):
    item = make_resources(tmp_path / "catalog")
    def one_source(tool, question, paper_id):
        evidence, sources = build_strict_context([tool.pipeline.retriever.chunks[0]])
        return {"sources": sources, "evidence": evidence}
    monkeypatch.setattr(PaperSearchTool, "run", one_source)
    yield item
    item.store.close()


@pytest.mark.parametrize("tools", [False, True])
def test_retry_is_persisted_with_stable_grouped_citations_and_paired_tools(resource, tools):
    config = replace(resource.config, supports_tools=tools)
    provider = tool_answer(resource, answer="Insufficient evidence [1].") if tools else FakeProvider(config, [Completion("Insufficient evidence [1].")])
    provider.config = config
    if tools:
        provider.requires_tool_definitions_for_history = True
    provider.responses.append(Completion("The additional passage supplies more evidence [1, 2].", finish_reason="stop"))
    cid = resource.conversations.create(resource.document.document_id, config)
    result = ResearchChatService(resource.conversations).ask(cid, "momentum", provider, resource.retriever, "request", retry_policy=POLICY)
    assert result["evidence_retry"]["status"] == "accepted"
    assert result["evidence_retry"]["model_attempts"] == 1
    assert result["evidence_retry"]["new_sources"]
    assert result["cited_numbers"] == [1, 2] and not result["invalid_citations"]
    assert len(provider.requests) == (3 if tools else 2)
    if tools:
        assert provider.requests[-1]["tools"] and provider.requests[-1]["tool_choice"] == "none"
    assert result["messages"][0]["content"] == "momentum"
    assert "Insufficient evidence" not in str(provider.requests[-1]["messages"])
    validate_turn(result["messages"])
    saved = resource.conversations.history(cid)[0]["result"]
    assert saved["evidence_retry"] == result["evidence_retry"]
    assert resource.conversations.sources(cid)[0]["citation"] == 1
    # Idempotent application request must not start a second retry.
    ResearchChatService(resource.conversations).ask(cid, "momentum", provider, resource.retriever, "request", retry_policy=POLICY)
    assert len(provider.requests) == (3 if tools else 2)


@pytest.mark.parametrize("response", [Completion("", finish_reason="stop"),
    Completion("Partial [2]", finish_reason="length"), Completion("Invented [99]"),
    ModelError("model_timeout", "normalized timeout")])
def test_retry_failure_keeps_original_answer_and_does_not_save_trial_citations(resource, response):
    provider = tool_answer(resource, answer="Insufficient evidence [1].")
    provider.responses.append(response)
    cid = resource.conversations.create(resource.document.document_id, provider.config)
    result = ResearchChatService(resource.conversations).ask(cid, "momentum", provider, resource.retriever, "request", retry_policy=POLICY)
    assert result["answer"] == "Insufficient evidence [1]."
    assert result["evidence_retry"]["status"] == "fallback" and result["model_calls"] == 3
    assert len(resource.conversations.sources(cid)) == 1
    validate_turn(result["messages"])


def test_retry_cancellation_does_not_commit_original_or_partial_answer(resource):
    provider = tool_answer(resource, answer="Insufficient evidence [1].")
    provider.responses.append(GenerationCancelled())
    cid = resource.conversations.create(resource.document.document_id, provider.config)
    with pytest.raises(GenerationCancelled):
        ResearchChatService(resource.conversations).ask(cid, "momentum", provider, resource.retriever,
            "request", retry_policy=POLICY, cancellation=CancellationToken())
    assert resource.conversations.history(cid)[0]["status"] != "completed"
    assert not resource.conversations.sources(cid)


def test_disabled_or_untriggered_retry_does_not_add_requests(resource):
    for enabled, answer in [(False, "Insufficient evidence [1]."), (True, "A supported fact [1].")]:
        provider = tool_answer(resource, answer=answer)
        cid = resource.conversations.create(resource.document.document_id, provider.config)
        result = ResearchChatService(resource.conversations).ask(cid, "momentum", provider, resource.retriever,
            "request", retry_policy=replace(POLICY, enabled=enabled))
        assert len(provider.requests) == 2
        assert result.get("evidence_retry", {}).get("status", "not_triggered") == "not_triggered"


def test_retry_overflow_keeps_draft_without_issuing_another_model_request(resource, monkeypatch):
    from src.agent.context_window import SlidingWindow
    provider = tool_answer(resource, answer="Insufficient evidence [1].")
    select = SlidingWindow.select
    def bounded(window, *args, **kwargs):
        if len(provider.requests) >= 2:
            raise ModelError("context_budget_exceeded", "retry would exceed budget")
        return select(window, *args, **kwargs)
    monkeypatch.setattr(SlidingWindow, "select", bounded)
    cid = resource.conversations.create(resource.document.document_id, provider.config)
    result = ResearchChatService(resource.conversations).ask(cid, "momentum", provider, resource.retriever,
        "request", retry_policy=POLICY)
    assert result["answer"] == "Insufficient evidence [1]." and len(provider.requests) == 2
    assert result["evidence_retry"]["model_attempts"] == 0
    assert result["evidence_retry"]["status"] == "fallback"


def test_second_turn_reuses_the_same_registered_source_numbers(resource):
    cid = resource.conversations.create(resource.document.document_id, resource.config)
    results = []
    for request in ("first", "second"):
        provider = tool_answer(resource, answer="Insufficient evidence [1].")
        provider.responses.append(Completion("More evidence [1, 2].", finish_reason="stop"))
        results.append(ResearchChatService(resource.conversations).ask(cid, "momentum", provider,
            resource.retriever, request, retry_policy=POLICY))
    assert results[0]["cited_numbers"] == results[1]["cited_numbers"] == [1, 2]
    assert [(s["citation"],s["chunk_id"]) for s in results[0]["registered_sources"]] == [(s["citation"],s["chunk_id"]) for s in results[1]["registered_sources"]]


def test_worker_displays_retry_status_and_replaces_failed_stream_text():
    task = GenerationTask("conversation", Mock())
    task.update({"kind": "retry_start"})
    assert "补检" in task.snapshot()["phase"]
    task.update({"kind": "model_start"})
    task.update({"kind": "text_delta", "text": "partial retry"})
    task.update({"kind": "retry_end", "status": "fallback", "text": "original answer"})
    assert task.snapshot()["text"] == "original answer" and "保留" in task.snapshot()["phase"]
