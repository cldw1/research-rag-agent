"""Real BM25/FAISS plumbing with synthetic vectors, never a quality benchmark."""

from copy import deepcopy

import numpy as np
import pytest

from src.generation.rag_pipeline import ResearchRAGPipeline
from src.ingestion.parsers import UnifiedDocumentParser
from src.ingestion.structured_chunker import ChunkConfig, StructuredChunker
from src.retrieval.structured_retriever import StructuredDocumentRetriever


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text) + 2))


class TestModel:
    tokenizer = FakeTokenizer()
    max_seq_length = 512

    def encode(self, texts, **kwargs):
        vectors = np.array([[1, float("momentum" in t.lower()), float("experiment" in t.lower())]
                            for t in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def fixture():
    document = UnifiedDocumentParser().parse(
        b"# Methods\n\nMomentum uses a position update.\n\n## Experiments\n\nExperiment measures accuracy.", "paper.md")
    chunks = StructuredChunker(ChunkConfig(strategy="structure")).chunk(document)
    return document, chunks


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_methods_retain_locations_and_reject_cross_document_access(method):
    document, chunks = fixture()
    retriever = StructuredDocumentRetriever(document, chunks, TestModel())
    result = retriever.search("momentum position", document.document_id, method=method)
    assert retriever.index.ntotal == len(chunks)
    assert result[0]["text"].startswith("Momentum")
    assert result[0]["source_metadata"]["spans"][0]["line_start"] == 3
    assert all(r["page_number"] is None for r in result)
    with pytest.raises(ValueError, match="当前文档"):
        retriever.search("momentum", "unselected-document", method=method)


def test_old_rag_pipeline_receives_new_document_and_preserves_citation_locations():
    document, chunks = fixture()
    retriever = StructuredDocumentRetriever(document, chunks, TestModel())
    prepared = ResearchRAGPipeline(retriever).prepare("momentum", document.document_id)
    assert prepared["sources"]
    assert all(s["document_version"] == document.document_version for s in prepared["sources"])
    assert all(s["source_metadata"]["spans"] for s in prepared["sources"])
    assert prepared["context_chars"] <= 8000


def test_old_agent_keeps_new_source_metadata_and_blocks_foreign_document():
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock
    from scripts.demo_agent_live_dedup import ResearchAgent

    document, chunks = fixture()
    retriever = StructuredDocumentRetriever(document, chunks, TestModel())
    agent = ResearchAgent(Mock(), retriever, document.document_id)
    def call(paper_id):
        return SimpleNamespace(function=SimpleNamespace(name="search_research_paper",
            arguments=json.dumps({"paper_id": paper_id, "question": "momentum"})))
    payload, _ = agent.execute_tool(call(document.document_id))
    assert payload["sources"] and all(s["source_metadata"]["spans"] for s in payload["sources"])
    assert all(s["document_version"] == document.document_version for s in agent.registry.get_all_sources())
    with pytest.raises(ValueError, match="其他论文"):
        agent.execute_tool(call("document_not_authorized"))


def test_retriever_refuses_mixed_chunk_versions():
    document, chunks = fixture()
    chunks = deepcopy(chunks)
    chunks[-1]["index_version"] = "different-version"
    with pytest.raises(ValueError, match="版本"):
        StructuredDocumentRetriever(document, chunks, TestModel())


def test_embedding_cannot_silently_truncate_oversized_chunks():
    document, chunks = fixture()
    model = TestModel()
    model.max_seq_length = 5
    with pytest.raises(ValueError, match="输入上限"):
        StructuredDocumentRetriever(document, chunks, model)

    model.max_seq_length = 512
    retriever = StructuredDocumentRetriever(document, chunks, model)
    with pytest.raises(ValueError, match="缩短问题"):
        retriever.search("long query " * 100, document.document_id)


def test_chinese_text_is_not_discarded_by_english_bm25_tokenizer():
    document = UnifiedDocumentParser().parse("统计推断方法。\n\n概率模型研究。".encode(), "中文.txt")
    chunks = StructuredChunker(ChunkConfig(strategy="structure")).chunk(document)
    retriever = StructuredDocumentRetriever(document, chunks, TestModel())
    results = retriever.search("统计推断", document.document_id, method="bm25")
    assert results and results[0]["document_id"] == document.document_id
