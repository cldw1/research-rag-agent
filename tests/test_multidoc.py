"""Multi-document scope and persistence with real SQLite/FAISS and fixture vectors."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from src.agent.paper_outline_tool import PaperOutlineTool
from src.agent.paper_search_tool import PaperSearchTool
from src.agent.research_agent import ResearchAgent
from src.chat_service import ResearchChatService
from src.generation.context_builder import build_context
from src.ingestion.structured_chunker import ChunkConfig
from src.llm.provider import Completion, ModelError, ToolCall
from src.retrieval.context_expander import expand_anchors_first
from src.storage.artifacts import StorageError
from src.storage.conversations import ConversationStore
from src.storage.document_store import LocalDocumentStore
from tests.chat_helpers import FakeProvider, make_resources


@pytest.fixture
def collection(tmp_path):
    r = make_resources(tmp_path / "library")
    r.second = r.store.ingest(b"# Methods\n\nExperiment uses a control group.\n\n# Results\n\nThe validation compares two methods.", "second.md").document
    r.outside = r.store.ingest(b"# Methods\n\nAn unselected document must never be included.", "outside.md").document
    for doc in (r.second, r.outside):
        r.store.build_index(doc.document_id, r.model, r.profile, r.counter, ChunkConfig(strategy="structure"))
    r.ids = [r.document.document_id, r.second.document_id]
    r.retriever = r.store.load_scoped_retriever(r.ids, r.model, r.profile)
    r.cid = r.conversations.create(r.ids, r.config, "两篇论文对比")
    yield r
    r.store.close()


def provider(r, call_id="search", target=None):
    return FakeProvider(r.config, [Completion(tool_calls=(ToolCall(call_id, "search_research_paper",
        json.dumps({"paper_id": target or r.retriever.paper_id, "question": "momentum experiment methods"})),)),
        Completion("第一篇讨论动量 [1]，第二篇描述对照实验 [2]。")])


def test_saved_vectors_are_combined_without_document_encoding_and_queries_encode_once(collection):
    r = collection
    r.model.encode = Mock(wraps=r.model.encode)
    retriever = r.store.load_scoped_retriever(r.ids, r.model, r.profile)
    r.model.encode.assert_not_called()
    assert retriever.bm25.corpus_size == len(retriever.chunks) == retriever.index.ntotal
    hits = retriever.search("momentum experiment", retriever.paper_id, top_k=5)
    assert r.model.encode.call_count == 1
    assert {h["paper_id"] for h in hits} == set(r.ids)
    assert [h["paper_id"] for h in hits[:2]] == r.ids
    assert all(h["persistent_index_id"] == retriever.document_bindings[h["paper_id"]] for h in hits)
    assert all(h["selection"] == "document_balanced" for h in hits)
    assert r.outside.document_id not in {h["paper_id"] for h in hits}


def test_agent_persists_both_papers_and_restores_stable_citations(collection):
    r = collection
    service = ResearchChatService(r.conversations)
    first = service.ask(r.cid, "比较两篇方法", provider(r), r.retriever, "first")
    assert set(first["evidence_document_ids"]) == set(r.ids)
    assert not first["missing_document_ids"]
    assert {s["paper_id"] for s in first["cited_sources"]} == set(r.ids)
    before = r.conversations.history(r.cid)
    sources = r.conversations.sources(r.cid)
    assert {s["persistent_index_id"] for s in sources} == set(r.retriever.document_bindings.values())
    assert "paper.md" in r.conversations.export_markdown(r.cid) and "second.md" in r.conversations.export_markdown(r.cid)
    r.store.close()
    reopened = LocalDocumentStore(r.store.root)
    try:
        conversations = ConversationStore(reopened)
        assert conversations.history(r.cid) == before and conversations.sources(r.cid) == sources
        details = conversations.details(r.cid)
        assert [item["document_id"] for item in details["documents"]] == r.ids
        retriever = reopened.load_scoped_retriever(r.ids, r.model, r.profile)
        second = ResearchChatService(conversations).ask(r.cid, "继续比较", provider(r, "second"), retriever, "second")
        assert second["cited_numbers"] == [1, 2]
        assert conversations.sources(r.cid) == sources
    finally:
        reopened.close()


def test_request_idempotency_does_not_repeat_multi_document_generation(collection):
    r = collection
    model = provider(r)
    service = ResearchChatService(r.conversations)
    first = service.ask(r.cid, "比较", model, r.retriever, "same")
    assert service.ask(r.cid, "比较", model, r.retriever, "same") == first
    assert len(model.requests) == 2


def test_tools_can_target_one_selected_paper_but_never_an_unselected_paper(collection):
    r = collection
    agent = ResearchAgent(provider(r), r.retriever)
    outside = ToolCall("outside", "search_research_paper", json.dumps({"paper_id": r.outside.document_id, "question": "methods"}))
    with pytest.raises(ModelError, match="当前文档以外"):
        agent._execute(outside, set())
    allowed = ToolCall("allowed", "search_research_paper", json.dumps({"paper_id": r.second.document_id, "question": "experiment"}))
    result = agent._execute(allowed, set())
    assert {s["paper_id"] for s in result["sources"]} == {r.second.document_id}
    outlines = PaperOutlineTool(r.retriever).run(r.retriever.paper_id)
    assert [item["paper_id"] for item in outlines["documents"]] == r.ids
    assert [item["filename"] for item in outlines["documents"]] == ["paper.md", "second.md"]
    payload = PaperSearchTool(r.retriever).run("methods", r.retriever.paper_id)
    assert [item["filename"] for item in payload["selected_documents"]] == ["paper.md", "second.md"]
    with pytest.raises(ValueError, match="选中的文档"):
        r.retriever.search("question", r.outside.document_id)


def test_service_rejects_missing_or_extra_document_retrievers_before_generation(collection):
    r = collection
    llm = provider(r)
    for ids in ([r.ids[0]], [*r.ids, r.outside.document_id]):
        wrong = r.store.load_scoped_retriever(ids, r.model, r.profile)
        with pytest.raises(StorageError, match="索引版本不一致"):
            ResearchChatService(r.conversations).ask(r.cid, "question", llm, wrong, "wrong")
    assert not llm.requests and not r.conversations.history(r.cid)


@pytest.mark.parametrize("kind", ["outside_document", "wrong_index"])
def test_citation_submission_rejects_forged_scope_atomically(collection, monkeypatch, kind):
    r = collection
    original = r.conversations.complete
    def forged(cid, tid, result):
        if kind == "outside_document":
            result["registered_sources"][0]["paper_id"] = r.outside.document_id
        else:
            result["registered_sources"][0]["persistent_index_id"] = "unbound-index"
        return original(cid, tid, result)
    monkeypatch.setattr(r.conversations, "complete", forged)
    with pytest.raises(StorageError, match="绑定的文档索引"):
        ResearchChatService(r.conversations).ask(r.cid, "compare", provider(r), r.retriever, "forged")
    assert not r.conversations.sources(r.cid)
    history = r.conversations.history(r.cid)
    assert history[0]["status"] == "failed" and len(history[0]["messages"]) == 1


def test_rebuilding_secondary_index_preserves_history_but_requires_a_new_conversation(collection):
    r = collection
    ResearchChatService(r.conversations).ask(r.cid, "compare", provider(r), r.retriever, "first")
    before = r.conversations.history(r.cid)
    r.store.build_index(r.second.document_id, r.model, r.profile, r.counter, force=True)
    assert r.conversations.history(r.cid) == before
    with pytest.raises(StorageError, match="新建会话"):
        r.conversations.details(r.cid, require_active=True)
    with pytest.raises(StorageError):
        r.retriever.search("methods", r.retriever.paper_id)


def test_removing_secondary_document_invalidates_scope_but_not_single_document_chat(collection):
    r = collection
    single = r.conversations.create(r.ids[0], r.config)
    r.store.delete_document(r.second.document_id)
    assert [item["id"] for item in r.conversations.list()] == [single]
    with pytest.raises(StorageError, match="移出文档库"):
        r.conversations.details(r.cid)
    with pytest.raises(StorageError):
        r.retriever.get_paper_data(r.ids[0])


def test_plain_rag_supports_multi_document_evidence(collection):
    r = collection
    llm = FakeProvider(replace(r.config, supports_tools=False), [Completion("两篇论文的证据 [1] [2]。")])
    result = ResearchChatService(r.conversations).ask(r.cid, "compare methods", llm, r.retriever, "plain")
    assert result["model_calls"] == 1 and not result["missing_document_ids"]
    assert {s["paper_id"] for s in result["cited_sources"]} == set(r.ids)
    assert all(m["role"] != "tool" for m in llm.requests[0]["messages"])


def test_neighbor_expansion_never_uses_another_papers_same_paragraph_numbers():
    def chunk(did, number):
        return dict(paper_id=did, chunk_id=f"{did}:{number}", section_index=0, paragraph_index=number, piece_index=0)
    anchor, neighbor, other = chunk("a", 0), chunk("a", 1), chunk("b", 1)
    found = expand_anchors_first([anchor], [anchor, neighbor, other])
    assert [c["chunk_id"] for c in found] == [anchor["chunk_id"], neighbor["chunk_id"]]


def test_budget_limited_context_reports_which_paper_has_no_evidence(collection):
    r = collection
    first = r.retriever.search("methods", r.retriever.paper_id, top_k=10)[0]
    budget = len(build_context([first])[0])
    result = PaperSearchTool(r.retriever, max_chars=budget).run("methods", r.retriever.paper_id)
    assert len(result["sources"]) == 1
    assert set(result["covered_document_ids"]) | set(result["missing_document_ids"]) == set(r.ids)
    assert len(result["missing_document_ids"]) == 1


@pytest.mark.parametrize("ids", [[], ["same", "same"], ["a", "b", "c", "d", "e", "f"], [None]])
def test_invalid_document_sets_are_rejected_without_partial_conversations(collection, ids):
    with pytest.raises(StorageError, match="1–5"):
        collection.conversations.create(ids, collection.config)
    assert len(collection.conversations.list()) == 1


def test_incompatible_embedding_profile_in_secondary_index_is_not_combined(collection):
    r = collection
    r.store.build_index(r.second.document_id, r.model, replace(r.profile, revision="other"), r.counter)
    with pytest.raises(StorageError, match="模型或配置"):
        r.store.load_scoped_retriever(r.ids, r.model, r.profile)


def test_five_document_boundary_each_gets_an_anchor_and_single_document_keeps_legacy_retriever(collection):
    r = collection
    ids = list(r.ids)
    for number in range(3):
        doc = r.store.ingest(f"# Methods\n\nExperiment number {number} uses a validation dataset.".encode(), f"extra{number}.md").document
        r.store.build_index(doc.document_id, r.model, r.profile, r.counter)
        ids.append(doc.document_id)
    cid = r.conversations.create(ids, r.config)
    retriever = r.store.load_scoped_retriever(ids, r.model, r.profile)
    assert len(r.conversations.details(cid)["documents"]) == 5
    assert {hit["paper_id"] for hit in retriever.search("experiment methods", retriever.paper_id, top_k=5)} == set(ids)
    single = r.store.load_scoped_retriever(ids[:1], r.model, r.profile)
    assert single.paper_id == ids[0] and single.index_id == r.retriever.document_bindings[ids[0]]
