import json
from dataclasses import replace
from unittest.mock import Mock

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.agent.context_window import validate_turn
from src.chat_service import ResearchChatService
from src.llm.provider import Completion, ModelError, ToolCall
from src.storage.artifacts import StorageError
from src.storage.backup import create_backup, restore_backup
from src.storage.conversations import ConversationStore
from src.storage.database import write_session
from src.storage.document_store import LocalDocumentStore
from src.storage.models import ConversationRecord, ConversationTurn, MessageRecord
from tests.chat_helpers import FakeProvider, make_resources, tool_answer


@pytest.fixture
def resource(tmp_path):
    result = make_resources(tmp_path / "catalog")
    yield result
    result.store.close()


def start(resource):
    return resource.conversations.create(resource.document.document_id, resource.config, "文献讨论")


def test_restart_recovers_complete_messages_and_stable_citation_numbers(resource):
    cid = start(resource)
    result = ResearchChatService(resource.conversations).ask(cid, "momentum?", tool_answer(resource), resource.retriever, "first")
    assert result["cited_numbers"] == [1]
    resource.store.close()
    reopened = LocalDocumentStore(resource.store.root)
    try:
        conversations = ConversationStore(reopened)
        history = conversations.history(cid)
        validate_turn(history[0]["messages"])
        assert history[0]["result"]["answer"] == result["answer"]
        assert conversations.sources(cid)[0]["citation"] == 1
        restored = reopened.load_retriever(resource.document.document_id, resource.model, resource.profile)
        provider = tool_answer(resource, call_id="second")
        second = ResearchChatService(conversations).ask(cid, "再解释一下", provider, restored, "second")
        assert second["cited_numbers"] == [1]
        assert "Momentum uses a position update" in provider.requests[1]["messages"][-1]["content"]
    finally:
        reopened.close()


def test_same_request_is_idempotent_but_different_question_conflicts(resource):
    cid = start(resource)
    service = ResearchChatService(resource.conversations)
    provider = tool_answer(resource)
    first = service.ask(cid, "question", provider, resource.retriever, "same")
    assert service.ask(cid, "question", provider, resource.retriever, "same") == first
    assert len(provider.requests) == 2
    with pytest.raises(StorageError, match="不同问题"):
        service.ask(cid, "different question", provider, resource.retriever, "same")


def test_failure_keeps_user_message_but_never_replays_incomplete_tool_history(resource):
    cid = start(resource)
    provider = FakeProvider(resource.config, [RuntimeError("private provider credential")])
    with pytest.raises(RuntimeError):
        ResearchChatService(resource.conversations).ask(cid, "question", provider, resource.retriever, "failed")
    history = resource.conversations.history(cid)
    assert history[0]["status"] == "failed" and len(history[0]["messages"]) == 1
    assert history[0]["error_code"] == "generation_failed"
    assert "credential" not in json.dumps(history)
    fresh = tool_answer(resource)
    ResearchChatService(resource.conversations).ask(cid, "next", fresh, resource.retriever, "fresh")
    assert len(fresh.requests[0]["messages"]) == 2  # system + current user only


def test_sessions_do_not_share_history_or_citation_registry(resource):
    first, second = start(resource), start(resource)
    service = ResearchChatService(resource.conversations)
    service.ask(first, "first question", tool_answer(resource), resource.retriever, "first")
    assert resource.conversations.sources(second) == resource.conversations.history(second) == []
    provider = tool_answer(resource)
    service.ask(second, "second question", provider, resource.retriever, "second")
    assert "first question" not in json.dumps(provider.requests)


def test_index_change_preserves_history_but_requires_new_conversation(resource):
    cid = start(resource)
    ResearchChatService(resource.conversations).ask(cid, "question", tool_answer(resource), resource.retriever, "first")
    resource.store.build_index(resource.document.document_id, resource.model, resource.profile, resource.counter, force=True)
    assert resource.conversations.history(cid)[0]["status"] == "completed"
    with pytest.raises(StorageError, match="新建会话"):
        resource.conversations.details(cid, require_active=True)


def test_deleted_document_blocks_history_and_new_generation(resource):
    cid = start(resource)
    resource.store.delete_document(resource.document.document_id)
    assert not resource.conversations.list()
    with pytest.raises(StorageError, match="移出文档库"):
        resource.conversations.history(cid)


def test_rename_delete_and_export_are_persistent_and_contain_sources(resource):
    cid = start(resource)
    ResearchChatService(resource.conversations).ask(cid, "question", tool_answer(resource), resource.retriever, "first")
    resource.conversations.rename(cid, "新名称")
    markdown = resource.conversations.export_markdown(cid)
    assert "# 新名称" in markdown and "Momentum uses a position update" in markdown
    resource.conversations.delete(cid)
    assert not resource.conversations.list()
    with pytest.raises(StorageError):
        resource.conversations.export_markdown(cid)


def test_active_turn_prevents_parallel_requests_and_expired_turn_can_be_recovered(resource):
    cid = start(resource)
    turn_id, _ = resource.conversations.begin(cid, "question", resource.config, "one")
    with pytest.raises(StorageError, match="正在生成"):
        resource.conversations.begin(cid, "second", resource.config, "two")
    assert resource.conversations.recover_expired() == 0
    with write_session(resource.store.engine) as session:
        session.get(ConversationTurn, turn_id).lease_until = "1970-01-01T00:00:00+00:00"
    assert resource.conversations.recover_expired() == 1
    assert resource.conversations.begin(cid, "second", resource.config, "two")[0] != turn_id


def test_plain_rag_fallback_removes_tool_pairs_together_when_switching_models(resource):
    cid = start(resource)
    service = ResearchChatService(resource.conversations)
    service.ask(cid, "first", tool_answer(resource), resource.retriever, "first")
    config = replace(resource.config, provider="openai_compatible", supports_tools=False)
    provider = FakeProvider(config, [Completion("Plain RAG answer [1].")])
    result = service.ask(cid, "second", provider, resource.retriever, "second")
    assert result["model_calls"] == 1
    assert all(m["role"] != "tool" and not m.get("tool_calls") for m in provider.requests[0]["messages"])
    assert resource.conversations.history(cid)[0]["messages"][1]["tool_calls"]


def test_backup_restores_messages_and_source_registry(resource, tmp_path):
    cid = start(resource)
    ResearchChatService(resource.conversations).ask(cid, "question", tool_answer(resource), resource.retriever, "first")
    create_backup(resource.store, tmp_path / "backup")
    restore_backup(tmp_path / "backup", tmp_path / "restored")
    reopened = LocalDocumentStore(tmp_path / "restored")
    try:
        restored = ConversationStore(reopened)
        assert restored.history(cid) == resource.conversations.history(cid)
        assert restored.sources(cid) == resource.conversations.sources(cid)
    finally:
        reopened.close()


def test_credential_is_never_stored_in_database_transcript_or_export(resource):
    import socket
    from types import SimpleNamespace
    from src.llm.provider import OpenAICompatibleProvider
    credential = "do-not-persist-this-credential"
    cid = start(resource)
    call = SimpleNamespace(id="call", function=SimpleNamespace(name="search_research_paper", arguments=json.dumps({
        "paper_id": resource.document.document_id, "question": "momentum"})))
    def response(content, calls):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls), finish_reason="stop")], usage=None)
    client = Mock()
    client.chat.completions.create.side_effect = [response(None, [call]), response("Answer [1].", [])]
    provider = OpenAICompatibleProvider(resource.config, credential, client=client, resolver=lambda *a, **k:
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))])
    ResearchChatService(resource.conversations).ask(cid, "question", provider, resource.retriever, "credential-test")
    assert credential not in resource.conversations.export_markdown(cid)
    assert credential not in json.dumps(resource.conversations.history(cid))
    assert credential.encode() not in resource.store.database_path.read_bytes()


def test_deletion_during_model_generation_does_not_publish_an_answer(resource):
    cid = start(resource)
    provider = tool_answer(resource)
    complete = provider.complete
    def delete_then_complete(*args, **kwargs):
        response = complete(*args, **kwargs)
        if not response.tool_calls:
            resource.conversations.delete(cid)
        return response
    provider.complete = delete_then_complete
    with pytest.raises(StorageError):
        ResearchChatService(resource.conversations).ask(cid, "question", provider, resource.retriever, "cancel")
    with Session(resource.store.engine) as session:
        turn = session.scalar(select(ConversationTurn).where(ConversationTurn.conversation_id == cid))
        assert turn.status == "cancelled" and turn.result is None


def test_failed_turn_commit_does_not_save_assistant_messages_or_registry(resource, monkeypatch):
    cid = start(resource)
    original = resource.conversations.complete
    def invalid_sources(conversation_id, turn_id, result):
        result["registered_sources"][0]["paper_id"] = "other-document"
        return original(conversation_id, turn_id, result)
    monkeypatch.setattr(resource.conversations, "complete", invalid_sources)
    with pytest.raises(StorageError, match="绑定的文档索引"):
        ResearchChatService(resource.conversations).ask(cid, "question", tool_answer(resource), resource.retriever, "bad-source")
    history = resource.conversations.history(cid)
    assert history[0]["status"] == "failed" and len(history[0]["messages"]) == 1
    assert not resource.conversations.sources(cid)


def test_context_pruning_keeps_full_database_history_and_resends_cited_evidence(resource):
    cid = start(resource)
    service = ResearchChatService(resource.conversations)
    old_answer = "Earlier discussion of retrieved evidence. " * 350 + "[1]."
    first = tool_answer(resource, answer=old_answer)
    first.config = replace(resource.config, max_output_tokens=4096)
    service.ask(cid, "first question", first, resource.retriever, "first")
    saved = resource.conversations.history(cid)[0]
    provider = tool_answer(resource, call_id="current_search")
    provider.config = replace(resource.config, context_window=8192)
    result = service.ask(cid, "current question", provider, resource.retriever, "current")
    assert result["context_selections"][0]["dropped_turns"] == 1
    assert all(message.get("content") != old_answer for message in provider.requests[0]["messages"])
    stored = resource.conversations.history(cid)
    assert len(stored) == 2 and stored[0] == saved
    validate_turn(stored[0]["messages"])
    assert "Momentum uses a position update" in provider.requests[1]["messages"][-1]["content"]
    assert result["cited_numbers"] == [1]
