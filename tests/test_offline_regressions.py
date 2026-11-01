"""Offline regression tests. No downloaded data, weights or LLM service required."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.demo_agent_live_dedup import ResearchAgent, deduplicate_tool_result
from scripts.test_step69_strategy_proxy import FakeRetriever as RankingOnlyRetriever
from src.agent.citation_registry import GlobalCitationRegistry
from src.agent.paper_outline_tool import PaperOutlineTool
from src.generation.context_builder import build_context
from src.generation.rag_pipeline import ResearchRAGPipeline
from src.generation.strict_context_builder import build_strict_context
from src.retrieval.retrieval_strategy import retrieve_with_strategy
from src.retrieval.strategy_proxy import RetrievalStrategyProxy
from src.retrieval.verified_v3_pdf_retriever import VerifiedV3PDFRetriever, _load_markers


ROOT = Path(__file__).resolve().parents[1]


class FakeRetriever(RankingOnlyRetriever):
    """Include the location metadata required by the full RAG pipeline."""

    def __init__(self):
        super().__init__()
        for index, chunk in enumerate(self.chunks):
            chunk.update(section_index=index, paragraph_index=0, piece_index=0)


@pytest.fixture(autouse=True)
def prohibit_network_and_paid_calls(monkeypatch):
    import socket
    from openai.resources.chat.completions import Completions

    def blocked(*args, **kwargs):
        raise AssertionError("Offline tests must not call a network or LLM service")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(Completions, "create", blocked)


@pytest.mark.parametrize("module", [
    "scripts.test_retrieval_strategy", "scripts.test_section_aware_reranker",
    "scripts.test_step69_strategy_proxy", "scripts.test_step70_smart_expander",
])
def test_existing_offline_assertions(module):
    importlib.import_module(module).main()


@pytest.mark.parametrize("strategy", ["hybrid", "section_aware"])
def test_agent_search_reaches_strategy_proxy(strategy):
    base = FakeRetriever()
    proxy = RetrievalStrategyProxy(base, strategy)
    agent = ResearchAgent(client=Mock(), retriever=proxy, paper_id="paper_test")
    call = SimpleNamespace(function=SimpleNamespace(
        name="search_research_paper",
        arguments=json.dumps({"paper_id": "paper_test", "question": "How is tuning performed?"}),
    ))
    payload, _ = agent.execute_tool(call)
    assert proxy.search_call_count == 1
    assert payload["sources"]
    assert payload["context_chars"] <= 8000
    assert all(s["paper_id"] == "paper_test" for s in payload["sources"])
    assert agent.search_tool.pipeline._use_smart_section_expansion() == (strategy == "section_aware")
    agent.client.chat.completions.create.assert_not_called()


def test_unverified_pdf_uses_hybrid_and_rejects_section_aware():
    base = FakeRetriever()
    base.has_verified_sections = False
    proxy = RetrievalStrategyProxy(base)
    results = proxy.search("tuning", "paper_test")
    assert [r["chunk_id"] for r in results] == list("ABCDE")
    assert not ResearchRAGPipeline(proxy)._use_smart_section_expansion()
    with pytest.raises(ValueError, match="验证"):
        RetrievalStrategyProxy(base, "section_aware")


@pytest.mark.parametrize("kwargs", [{"strategy": "bad"}, {"top_k": 0}, {"candidate_k": 0}, {"paper_id": ""}])
def test_strategy_rejects_invalid_parameters(kwargs):
    args = {"retriever": FakeRetriever(), "query": "tuning", "paper_id": "paper_test"}
    args.update(kwargs)
    with pytest.raises(ValueError):
        retrieve_with_strategy(**args)


def evidence(chunk_id, text, section="Methods", page=1):
    return {
        "chunk_id": chunk_id, "paper_id": "paper_test", "title": "Test",
        "section_name": section, "text": text, "section_index": page - 1,
        "paragraph_index": 0, "piece_index": 0, "page_number": page,
    }


def test_context_budget_counts_separators_and_preserves_complete_passages():
    items = [evidence("a", "first evidence"), evidence("b", "second evidence")]
    full, _ = build_context(items, max_chars=1000)
    exact, sources = build_context(items, max_chars=len(full))
    assert exact == full and len(sources) == 2
    context, sources = build_context(items, max_chars=len(full) - 1)
    assert len(context) <= len(full) - 1
    assert len(sources) == 1 and "first evidence" in context
    assert "second evidence" not in context
    assert build_strict_context(items, max_chars=1) == ("", [])
    with pytest.raises(ValueError):
        build_strict_context(items, max_chars=0)


def test_citations_deduplicate_across_tool_calls():
    a, b = evidence("a", "support A"), evidence("b", "support B")
    registry = GlobalCitationRegistry()

    def payload(items):
        context, sources = build_strict_context(items)
        return {"paper_id": "paper_test", "question": "test", "evidence": context, "sources": sources}

    first = deduplicate_tool_result(payload([a, a]), registry)
    second = deduplicate_tool_result(payload([a, b]), registry)
    assert first["new_chunk_count"] == 1
    assert second["new_chunk_count"] == 1 and second["reused_chunk_count"] == 1
    assert [s["citation"] for s in second["sources"]] == [1, 2]
    assert "support A" not in second["evidence"]
    assert "support B" in second["evidence"]
    assert registry.get_source(2)["chunk_id"] == "b"


def test_smart_expansion_budget_keeps_anchor_before_cross_page_neighbor():
    anchor = evidence("anchor", "Leapfrog integrator updates position.", page=2)
    neighbor = evidence("formula", "Leapfrog p = p - gradient (5.18). q = q + p (5.19).", page=1)
    unrelated = evidence("other", "Leapfrog equation (5.20) x = y", section="Other")
    proxy = SimpleNamespace(has_verified_sections=True, strategy="section_aware")
    pipeline = ResearchRAGPipeline(proxy, max_chars=1000)
    expanded = pipeline._smart_expand("leapfrog update equation", [anchor], [neighbor, anchor, unrelated])
    assert [r["chunk_id"] for r in expanded] == ["anchor", "formula"]
    pipeline.max_chars = len(build_strict_context([anchor])[0])
    assert [r["chunk_id"] for r in pipeline._smart_expand("leapfrog update", [anchor], [neighbor, anchor])] == ["anchor"]


def test_pipeline_blocks_cross_paper_evidence():
    item = evidence("x", "content")
    item["paper_id"] = "another_paper"
    retriever = Mock()
    retriever.get_paper_data.return_value = {"chunks": [item]}
    retriever.search.return_value = [item]
    with pytest.raises(RuntimeError, match="跨论文"):
        ResearchRAGPipeline(retriever).prepare("test", "paper_test", method="top5")


@pytest.mark.parametrize("arguments", ["[]", "bad json", '{"paper_id":"another_paper"}'])
def test_agent_rejects_invalid_tool_arguments(arguments):
    agent = ResearchAgent(Mock(), FakeRetriever(), "paper_test")
    call = SimpleNamespace(function=SimpleNamespace(name="search_research_paper", arguments=arguments))
    with pytest.raises(ValueError):
        agent.execute_tool(call)


def test_wrong_pdf_cannot_load_hmc_mapping():
    with pytest.raises(ValueError, match="校验"):
        VerifiedV3PDFRetriever(b"%PDF-unverified", "HandbookChapter5.pdf", Mock(), ROOT / "results/pdf_v3_section_mapping_report.json")


def test_runtime_mapping_is_structured_and_valid():
    path = ROOT / "results/pdf_v3_section_mapping_report.json"
    mapping = json.loads(path.read_text(encoding="utf-8"))
    assert "preview_chunks" not in mapping
    assert sum(map(len, _load_markers(path).values())) == 50


def test_qasper_missing_index_has_preparation_instructions(monkeypatch, tmp_path):
    from src.retrieval import validation_paper_scoped_retriever as validation
    monkeypatch.setattr(validation, "INDEX_PATH", tmp_path / "absent.faiss")
    with pytest.raises(FileNotFoundError, match="prepare_validation_sample"):
        validation.ValidationPaperScopedRetriever()


def test_verified_outline_preserves_headings_on_same_page_and_merges_across_pages():
    chunks = [evidence("a", "text", "5.1 Introduction", 1),
              evidence("b", "text", "5.2 Methods", 1),
              evidence("c", "text", "5.2 Methods", 2)]
    base = SimpleNamespace(has_verified_sections=True, get_paper_data=lambda pid: {"chunks": chunks})
    outline = PaperOutlineTool(base).run("paper_test")
    assert [s["section_name"] for s in outline["sections"]] == ["5.1 Introduction", "5.2 Methods"]
    assert [s["chunk_count"] for s in outline["sections"]] == [1, 2]


def test_agent_loop_tracks_citations_and_enforces_tool_budget():
    client = Mock()
    proxy = RetrievalStrategyProxy(FakeRetriever())
    agent = ResearchAgent(client, proxy, "paper_test")
    calls = [SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(
        name="search_research_paper",
        arguments=json.dumps({"paper_id": "paper_test", "question": "tuning"}),
    )) for i in range(3)]
    client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls, content=None))]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[], content="Evidence [1], invalid [999], placeholder [n]."))]),
    ]
    result = agent.ask("How does tuning work?")
    assert result["executed_tool_count"] == 2
    assert len(result["skipped_tool_calls"]) == 1
    assert proxy.search_call_count == 2
    assert result["invalid_citations"] == [999]
    assert result["cited_sources"][0]["citation"] == 1
    assert result["placeholder_found"] and "[n]" not in result["answer"]


def test_hmc_reproduction_evaluator_does_not_invent_new_manual_scores():
    from scripts.evaluate_hmc_v3 import evaluate
    retriever = FakeRetriever()
    retriever.paper_id = "paper_test"
    report = evaluate(retriever, [{
        "id": "fixture", "question": "tuning", "gold_section": "5.4.2",
        "gold_section_title": "5.4.2 Tuning HMC",
    }])
    assert report["metrics"]["hybrid"]["hit_at_5"] == 1
    assert report["metrics"]["section_aware"]["mrr_at_5"] == 1
    assert "evidence_score" not in report["records"][0]["hybrid"]
