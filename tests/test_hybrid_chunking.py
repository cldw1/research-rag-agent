"""Real optional Docling Core with a local, deterministic HuggingFace tokenizer."""

from dataclasses import replace

import numpy as np
import pytest

from src.ingestion.parsers import UnifiedDocumentParser
from src.ingestion.structured_chunker import ByteBudgetCounter, ChunkConfig, ModelTokenCounter, StructuredChunker
from src.storage.conversations import ConversationStore
from src.storage.document_store import LocalDocumentStore
from src.storage.embedding_profile import EmbeddingProfile
from tests.test_structured_ingestion import make_docx, make_pdf


@pytest.fixture
def counter():
    pytest.importorskip("docling_core", reason="Optional requirements-docling.txt is not installed")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3,
        "Methods": 4, "Results": 5, "fixed": 6, "evidence": 7}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", model_max_length=512)
    return ModelTokenCounter(tokenizer, "local-fixture")


def test_hybrid_obeys_actual_token_and_character_budget_and_preserves_source_blocks(counter):
    doc = UnifiedDocumentParser().parse(("# Methods\n\n" + "fixed evidence about science. " * 100 + "\n\n# Results\n\n" + "数据与来源。" * 40).encode(), "source.md")
    config = ChunkConfig(strategy="hybrid", max_tokens=40, max_chars=160)
    chunks = StructuredChunker(config, counter).chunk(doc)
    source = {block.block_id: block for block in doc.blocks}
    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk["token_count"] <= 40 and len(chunk["text"]) <= 160
        assert chunk["token_count"] == counter.count(chunk["text"])
        assert chunk["page_number"] is None and "\ufffd" not in chunk["text"]
        assert chunk["source_metadata"]["span_granularity"] == "source_block"
        for span in chunk["source_metadata"]["spans"]:
            assert span["char_start"] == 0 and span["char_end"] == len(source[span["block_id"]].text)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    other = StructuredChunker(replace(config, max_tokens=35), counter).chunk(doc)
    assert other[0]["index_version"] != chunks[0]["index_version"]


def test_hybrid_never_merges_pdf_pages(counter):
    doc = UnifiedDocumentParser().parse(make_pdf(["First page evidence.", "Second page evidence."]), "x.pdf")
    chunks = StructuredChunker(ChunkConfig(strategy="hybrid"), counter).chunk(doc)
    assert {chunk["page_number"] for chunk in chunks} == {1, 2}
    assert all({span["page_number"] for span in chunk["source_metadata"]["spans"]} == {chunk["page_number"]} for chunk in chunks)


def test_hybrid_word_table_retains_paragraph_locations_without_pages(counter):
    body = '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Metric</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr><w:tr><w:tc><w:p><w:r><w:t>Samples</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>32</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
    doc = UnifiedDocumentParser().parse(make_docx(body), "table.docx")
    chunks = StructuredChunker(ChunkConfig(strategy="hybrid"), counter).chunk(doc)
    assert "32" in " ".join(chunk["text"] for chunk in chunks)
    assert all(chunk["page_number"] is None for chunk in chunks)
    assert chunks[0]["source_metadata"]["spans"][0]["paragraph_index"] == 0


def test_hybrid_missing_tokenizer_and_character_overlap_are_explicit_errors():
    doc = UnifiedDocumentParser().parse(b"Some evidence.", "x.txt")
    with pytest.raises(ValueError, match="tokenizer"):
        StructuredChunker(ChunkConfig(strategy="hybrid"), ByteBudgetCounter()).chunk(doc)
    with pytest.raises(ValueError, match="overlap"):
        ChunkConfig(strategy="hybrid", overlap_chars=2)


def test_hybrid_index_restarts_with_same_chunks_and_sources(tmp_path, counter):
    class Embedding:
        max_seq_length = 512
        tokenizer = counter.tokenizer
        def encode(self, texts, **kwargs):
            vectors = np.array([[1., float("evidence" in text), 1.] for text in texts], dtype=np.float32)
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    model = Embedding()
    profile = EmbeddingProfile("local-fixture", "v1", 3, counter.identity, 512)
    root = tmp_path / "catalog"
    store = LocalDocumentStore(root)
    document = store.ingest(b"# Methods\n\nFixed research evidence with sources.", "x.md").document
    result = store.build_index(document.document_id, model, profile, counter, ChunkConfig(strategy="hybrid"))
    first = store.load_retriever(document.document_id, model, profile).search("evidence", document.document_id)
    store.close()
    reopened = LocalDocumentStore(root)
    try:
        assert reopened.active_index_info(document.document_id)["index_id"] == result.index_id
        restored = reopened.load_retriever(document.document_id, model, profile).search("evidence", document.document_id)
        assert [item["chunk_id"] for item in first] == [item["chunk_id"] for item in restored]
        assert first[0]["source_metadata"] == restored[0]["source_metadata"]
    finally:
        reopened.close()
