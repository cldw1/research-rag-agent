"""Persisted wire state is scoped to provider/model; retries append new turns."""

import json
from dataclasses import replace

import pytest

from src.chat_service import ResearchChatService
from src.generation_task import GenerationPool
from src.llm.native import native_state
from src.llm.provider import Completion, ModelError, ModelEvent, ToolCall
from src.storage.conversations import ConversationStore
from src.storage.document_store import LocalDocumentStore
from tests.chat_helpers import FakeProvider, make_resources
from tests.test_native_providers import config


def scripted(config, document_id, answer="Answer [1]."):
    call = ToolCall("c1", "search_research_paper", json.dumps({"paper_id": document_id, "question": "momentum"}))
    return FakeProvider(config, [Completion(tool_calls=(call,), provider_state=native_state(config, [
        {"type": "reasoning", "encrypted_content": "opaque-fixture"}])), Completion(answer)])


@pytest.mark.parametrize("change", ["same", "model", "provider", "base_url"])
def test_restart_keeps_native_history_only_for_same_service_and_model(tmp_path, change):
    resource = make_resources(tmp_path / "catalog")
    original = replace(resource.config, provider="openai_compatible", model="fixture-v1", base_url="https://example.com/v1")
    cid = resource.conversations.create(resource.document.document_id, original)
    first = scripted(original, resource.document.document_id)
    ResearchChatService(resource.conversations).ask(cid, "first", first, resource.retriever, "one")
    resource.store.close()
    reopened = LocalDocumentStore(tmp_path / "catalog")
    try:
        chats = ConversationStore(reopened)
        new_config = original
        if change == "model":
            new_config = replace(original, model="fixture-v2")
        elif change == "provider":
            new_config = config("OpenAI Responses")
        elif change == "base_url":
            new_config = replace(original, base_url="https://other.example.com/v1")
        second = scripted(new_config, resource.document.document_id, "Second answer [1].")
        retriever = reopened.load_retriever(resource.document.document_id, resource.model, resource.profile)
        result = ResearchChatService(chats).ask(cid, "second", second, retriever, "two")
        assert result["answer"] == "Second answer [1]."
        old_messages = second.requests[0]["messages"][1:-1]
        if change == "same":
            assert any(message.get("provider_state") for message in old_messages)
            assert any(message["role"] == "tool" for message in old_messages)
        else:
            assert old_messages == [{"role": "user", "content": "first"}, {"role": "assistant", "content": "Answer [1]."}]
        assert chats.history(cid)[0]["messages"][1]["provider_state"]["content"][0]["encrypted_content"] == "opaque-fixture"
    finally:
        reopened.close()


def test_successful_stream_keeps_only_final_answer_and_regeneration_appends(tmp_path):
    resource = make_resources(tmp_path / "catalog")
    cfg = replace(resource.config, stream=True)
    cid = resource.conversations.create(resource.document.document_id, cfg)
    class StreamingProvider:
        config = cfg
        def __init__(self):
            self.calls = 0
        def stream_complete(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                yield ModelEvent("text_delta", "I will search first.")
                yield ModelEvent("completed", completion=Completion(tool_calls=(ToolCall("c1", "search_research_paper", json.dumps({
                    "paper_id": resource.document.document_id, "question": "momentum"})),)))
            else:
                yield ModelEvent("text_delta", "Answer ")
                yield ModelEvent("text_delta", "[1].")
                yield ModelEvent("completed", completion=Completion("Answer [1]."))
    pool = GenerationPool(1)
    try:
        service = ResearchChatService(resource.conversations)
        task = pool.start(cid, lambda token, callback: service.ask(cid, "question", StreamingProvider(), resource.retriever,
            "first", cancellation=token, on_event=callback))
        task.future.result(timeout=15)
        assert task.snapshot()["text"] == "Answer [1]." and task.work is None
        second = scripted(resource.config, resource.document.document_id, "Regenerated answer [1].")
        service.ask(cid, "question", second, resource.retriever, "regenerated")
        history = resource.conversations.history(cid)
        assert [turn["result"]["answer"] for turn in history] == ["Answer [1].", "Regenerated answer [1]."]
        assert "I will search first" not in history[0]["messages"][-1]["content"]
    finally:
        pool.close()
        resource.store.close()


def test_retry_uses_new_request_without_failed_tool_exchange(tmp_path):
    resource = make_resources(tmp_path / "catalog")
    cid = resource.conversations.create(resource.document.document_id, resource.config)
    service = ResearchChatService(resource.conversations)
    provider = scripted(resource.config, resource.document.document_id)
    provider.responses[-1] = ModelError("model_connection", "Fixture disconnected")
    try:
        with pytest.raises(ModelError):
            service.ask(cid, "question", provider, resource.retriever, "failed")
        retry = scripted(resource.config, resource.document.document_id)
        service.ask(cid, "question", retry, resource.retriever, "retry")
        assert retry.requests[0]["messages"][-1] == {"role": "user", "content": "question"}
        assert len(retry.requests[0]["messages"]) == 2
        assert [turn["status"] for turn in resource.conversations.history(cid)] == ["failed", "completed"]
    finally:
        resource.store.close()
