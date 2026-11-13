import json
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.chat_service import ResearchChatService
from src.generation_task import GenerationPool
from src.llm.control import CancellationToken, GenerationCancelled
from src.llm.provider import Completion, ModelEvent, ModelError, OpenAICompatibleProvider, ToolCall
from tests.chat_helpers import FakeProvider, make_resources
from tests.test_llm_provider import public_resolver
from tests.test_native_providers import ClosingEvents


def chunk(text=None, call=None, finish=None, usage=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=[call] if call else []),
                                                   finish_reason=finish)], usage=usage)


def call(index, arguments, name=None, identifier=None):
    return SimpleNamespace(index=index, id=identifier, function=SimpleNamespace(name=name, arguments=arguments))


def test_compatible_stream_combines_tool_argument_fragments_before_execution():
    client = Mock()
    events = ClosingEvents([chunk(call=call(0, '{"paper_', "get_paper_outline", "c1")),
                            chunk(call=call(0, 'id":"p"}')), chunk(finish="tool_calls")])
    client.chat.completions.create.return_value = events
    from src.llm.provider import ModelConfig
    provider = OpenAICompatibleProvider(ModelConfig(stream=True), "dummy", client=client, resolver=public_resolver)
    result = list(provider.stream_complete([{"role": "user", "content": "q"}]))[-1].completion
    assert result.tool_calls[0].id == "c1" and json.loads(result.tool_calls[0].arguments) == {"paper_id": "p"}
    assert events.closed


def test_disconnected_chat_stream_is_not_retried_or_marked_complete():
    client = Mock()
    client.chat.completions.create.return_value = ClosingEvents([chunk("partial")])
    from src.llm.provider import ModelConfig
    provider = OpenAICompatibleProvider(ModelConfig(stream=True), "dummy", client=client, resolver=public_resolver)
    with pytest.raises(ModelError, match="中断"):
        list(provider.stream_complete([{"role": "user", "content": "q"}]))
    assert client.chat.completions.create.call_count == 1


def test_cancellation_releases_stream_and_never_emits_completed_event():
    client, token = Mock(), CancellationToken()
    events = ClosingEvents([chunk("partial"), chunk("remaining"), chunk(finish="stop")])
    client.chat.completions.create.return_value = events
    from src.llm.provider import ModelConfig
    provider = OpenAICompatibleProvider(ModelConfig(stream=True), "dummy", client=client, resolver=public_resolver)
    iterator = provider.stream_complete([{"role": "user", "content": "q"}], cancellation=token)
    assert next(iterator).text == "partial"
    token.cancel()
    with pytest.raises(GenerationCancelled):
        list(iterator)
    assert events.closed


def test_worker_stop_cancels_database_turn_and_partial_answer_is_not_history(tmp_path):
    resource = make_resources(tmp_path / "catalog")
    config = replace(resource.config, stream=True)
    cid = resource.conversations.create(resource.document.document_id, config)
    ready = Event()
    class GatedProvider:
        def __init__(self):
            self.config, self.requests = config, 0
        def stream_complete(self, messages, cancellation=None, **kwargs):
            self.requests += 1
            if self.requests == 1:
                yield ModelEvent("completed", completion=Completion(tool_calls=(ToolCall("c1", "search_research_paper", json.dumps({
                    "paper_id": resource.document.document_id, "question": "momentum"})),)))
            else:
                yield ModelEvent("text_delta", "incomplete answer")
                ready.set()
                cancellation.event.wait(5)
                cancellation.check()
                raise AssertionError("cancelled generation must stop")
    provider = GatedProvider()
    pool = GenerationPool(1)
    try:
        task = pool.start(cid, lambda token, callback: ResearchChatService(resource.conversations).ask(
            cid, "question", provider, resource.retriever, "cancel", cancellation=token, on_event=callback))
        assert ready.wait(10)
        assert task.snapshot()["text"] == "incomplete answer"
        assert task.cancel()
        task.future.result(timeout=10)
        assert task.snapshot()["status"] == "cancelled" and not task.snapshot()["text"]
        history = resource.conversations.history(cid)
        assert history[0]["status"] == "cancelled" and len(history[0]["messages"]) == 1
        assert not resource.conversations.sources(cid)
    finally:
        pool.close()
        resource.store.close()


def test_no_forced_tool_capability_prefetches_evidence_and_keeps_user_question(tmp_path):
    resource = make_resources(tmp_path / "catalog")
    try:
        cid = resource.conversations.create(resource.document.document_id, resource.config)
        provider = FakeProvider(resource.config, [Completion("Grounded answer [1].")])
        provider.supports_required_tools = False
        result = ResearchChatService(resource.conversations).ask(cid, "question", provider, resource.retriever, "auto")
        assert provider.requests[0]["tool_choice"] == "auto"
        assert "reference_context" in result["messages"][0]
        assert result["messages"][0]["content"] == "question"
        assert resource.conversations.history(cid)[0]["status"] == "completed"
    finally:
        resource.store.close()
