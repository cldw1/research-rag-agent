"""Exercise real PyMuPDF, BM25 and FAISS with test-only deterministic vectors."""

import numpy as np
import pymupdf
import pytest

from src.retrieval.strategy_proxy import RetrievalStrategyProxy
from src.retrieval.uploaded_pdf_retriever import MAX_PAGES, MAX_PDF_BYTES, UploadedPDFRetriever


class TestEmbedding:
    """Test fixture only; the application always uses the BGE model."""

    def encode(self, texts, **kwargs):
        vectors = np.array([[1.0, float("leapfrog" in t.lower()), float("momentum" in t.lower())] for t in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def make_pdf(texts):
    with pymupdf.open() as doc:
        for text in texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        return doc.tobytes()


def test_regular_pdf_indexes_and_searches_without_hmc_sections():
    retriever = UploadedPDFRetriever(make_pdf(["Introduction to research papers.", "Leapfrog momentum update method.", "Unrelated experimental background."]), "ordinary.pdf", TestEmbedding())
    proxy = RetrievalStrategyProxy(retriever)
    results = proxy.search("leapfrog momentum", retriever.paper_id, top_k=3)
    assert retriever.page_count == retriever.index.ntotal == len(retriever.chunks) == 3
    assert {r["section_name"] for r in results} == {"Page 1", "Page 2", "Page 3"}
    assert results[0]["page_number"] == 2
    with pytest.raises(ValueError, match="验证"):
        RetrievalStrategyProxy(retriever, "section_aware")


def test_empty_scanned_style_pdf_has_clear_error():
    with pytest.raises(ValueError, match="OCR"):
        UploadedPDFRetriever(make_pdf([""]), "empty.pdf", TestEmbedding())


@pytest.mark.parametrize("content, message", [(b"not a pdf", "有效"), (b"%PDF" + b" " * MAX_PDF_BYTES, "20 MB")])
def test_invalid_or_oversized_pdf_is_rejected_before_embedding(content, message):
    with pytest.raises(ValueError, match=message):
        UploadedPDFRetriever(content, "bad.pdf", None)


def test_too_many_pages_is_rejected_before_embedding():
    with pytest.raises(ValueError, match=str(MAX_PAGES)):
        UploadedPDFRetriever(make_pdf([""] * (MAX_PAGES + 1)), "long.pdf", None)
