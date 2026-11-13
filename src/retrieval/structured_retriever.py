"""New structured-document adapter; leaves all historical retrievers untouched."""

import re
from copy import deepcopy

from src.runtime import load_faiss

faiss = load_faiss()
import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.uploaded_pdf_retriever import QUERY_PREFIX, RRF_K


def lexical_tokens(text):
    # English word tokens + CJK characters: avoids an empty BM25 corpus on Chinese.
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[\u3400-\u9fff]", text.lower())


class StructuredDocumentRetriever:
    has_verified_sections = False  # Never enable HMC-specific section reranking.

    def __init__(self, document, chunks, model):
        self._initialize(document, chunks, model)
        self.vectors = np.ascontiguousarray(model.encode(
            [c["text"] for c in chunks], batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ), dtype=np.float32)
        self._check_vectors()
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    def _initialize(self, document, chunks, model):
        if not chunks or any(c["document_id"] != document.document_id or
                             c["document_version"] != document.document_version for c in chunks):
            raise ValueError("文档与 Chunk 版本不一致或内容为空")
        if len({c["index_version"] for c in chunks}) != 1:
            raise ValueError("不能混用不同分块版本")
        self.document, self.chunks, self.model = document, deepcopy(chunks), model
        self.paper_id, self.title, self.page_count = document.document_id, document.title, document.page_count
        self.paper_rows = {self.paper_id: list(range(len(chunks)))}
        tokenized = [lexical_tokens(c["text"]) or ["__no_lexical_tokens__"] for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        # Reject silent model truncation even if the caller forgot the exact tokenizer.
        for chunk in chunks:
            if len(model.tokenizer.encode(chunk["text"], add_special_tokens=True, truncation=False)) > model.max_seq_length:
                raise ValueError("Chunk 超过 Embedding 输入上限，请用当前模型 Tokenizer 重新分块")
    def _check_vectors(self):
        if (self.vectors.ndim != 2 or len(self.vectors) != len(self.chunks) or self.vectors.shape[1] < 1
                or not np.isfinite(self.vectors).all()):
            raise ValueError("Embedding 输出形状或数值无效")

    @classmethod
    def from_saved(cls, document, chunks, model, vectors, index):
        instance = cls.__new__(cls)
        instance._initialize(document, chunks, model)
        instance.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        instance._check_vectors()
        if (index.ntotal != len(chunks) or index.d != instance.vectors.shape[1]
                or not isinstance(index, faiss.IndexFlatIP)):
            raise ValueError("FAISS 索引形状或类型不匹配")
        # IndexFlatIP exposes exact stored vectors; validate the ordered mapping.
        restored = index.reconstruct_n(0, index.ntotal)
        if not np.array_equal(restored, instance.vectors):
            raise ValueError("FAISS 向量与持久化向量顺序不一致")
        instance.index = index
        return instance

    def get_paper_data(self, paper_id):
        if str(paper_id) != self.paper_id:
            raise ValueError("只允许访问当前文档")
        return {"chunks": self.chunks, "bm25": self.bm25, "vectors": self.vectors, "index": self.index}

    def search(self, query, paper_id, top_k=5, candidate_k=50, method="hybrid"):
        self.get_paper_data(paper_id)
        if not isinstance(query, str) or not query.strip() or top_k < 1 or candidate_k < 1:
            raise ValueError("检索参数无效")
        if method not in ("bm25", "dense", "hybrid"):
            raise ValueError("不支持的检索方法")
        count = min(len(self.chunks), max(top_k, candidate_k))
        ranks = {}
        if method in ("bm25", "hybrid"):
            order = np.argsort(-np.asarray(self.bm25.get_scores(lexical_tokens(query))), kind="stable")[:count]
            ranks["bm25"] = {int(i): r for r, i in enumerate(order, 1)}
        if method in ("dense", "hybrid"):
            if len(self.model.tokenizer.encode(QUERY_PREFIX + query, add_special_tokens=True,
                                               truncation=False)) > self.model.max_seq_length:
                raise ValueError("检索问题超过 Embedding 输入上限，请缩短问题")
            vector = np.ascontiguousarray(self.model.encode(
                [QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            ), dtype=np.float32)
            _, indices = self.index.search(vector, count)
            ranks["dense"] = {int(i): r for r, i in enumerate(indices[0], 1) if i >= 0}
        scores = {}
        for ranking in ranks.values():
            for index, rank in ranking.items():
                scores[index] = scores.get(index, 0) + 1 / (RRF_K + rank)
        ordered = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
        return [{**self.chunks[i], "rank": r, "rrf_score": scores[i],
                 "bm25_rank": ranks.get("bm25", {}).get(i),
                 "dense_rank": ranks.get("dense", {}).get(i)} for r, i in enumerate(ordered, 1)]
