"""Scoped multi-document search using saved vectors, global ranks and paper balancing."""

from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.structured_retriever import lexical_tokens
from src.retrieval.uploaded_pdf_retriever import QUERY_PREFIX, RRF_K
from src.runtime import load_faiss
from src.storage.artifacts import fingerprint


class MultiDocumentRetriever:
    has_verified_sections = False

    def __init__(self, retrievers, *, evidence_order='rrf'):
        if evidence_order not in ('rrf','bm25_first','dense_first'):
            raise ValueError("不支持的多文档证据顺序")
        self._evidence_order=evidence_order
        if not 2 <= len(retrievers) <= 5:
            raise ValueError("多文档检索需要 2–5 篇已索引文档")
        self.document_retrievers = {item.paper_id: item for item in retrievers}
        if len(self.document_retrievers) != len(retrievers):
            raise ValueError("不能重复选择同一文档")
        self.document_ids = tuple(self.document_retrievers)
        self.document_bindings = {item.paper_id: item.index_id for item in retrievers}
        self.paper_id = "scope_" + fingerprint(sorted(self.document_bindings.items()))[:24]
        self.allowed_paper_ids = frozenset([self.paper_id, *self.document_ids])
        self.title = "Selected research documents"
        self.model = retrievers[0].model
        dimension = retrievers[0].index.d
        if any(item.model is not self.model or item.index.d != dimension for item in retrievers):
            raise ValueError("多文档向量必须使用同一 Embedding 模型与配置")
        self.chunks = []
        for item in retrievers:
            if any(chunk["paper_id"] != item.paper_id for chunk in item.chunks):
                raise ValueError("文档片段来源不匹配")
            self.chunks.extend({**chunk, "persistent_index_id": item.index_id} for chunk in item.chunks)
        if len({chunk["chunk_id"] for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("多文档 Chunk 身份重复")
        self.vectors = np.ascontiguousarray(np.concatenate([item.vectors for item in retrievers]), dtype=np.float32)
        self.index = load_faiss().IndexFlatIP(dimension)
        self.index.add(self.vectors)  # Reuse persisted document vectors; no document encode.
        self.bm25 = BM25Okapi([lexical_tokens(c["text"]) or ["__no_lexical_tokens__"] for c in self.chunks])
        self.page_count = sum(item.page_count or 0 for item in retrievers)
        self.paper_rows = {did: [i for i, c in enumerate(self.chunks) if c["paper_id"] == did]
                           for did in self.document_ids}
        self._check_active()

    @property
    def evidence_order(self):
        return self._evidence_order

    def _check_active(self):
        for member in self.document_retrievers.values():
            member.get_paper_data(member.paper_id)

    def get_paper_data(self, paper_id):
        self._check_active()
        if paper_id == self.paper_id:
            return {"chunks": self.chunks, "vectors": self.vectors, "index": self.index,
                    "bm25": self.bm25, "document_ids": list(self.document_ids),
                    "documents": [{"paper_id": did, "filename": member.document.filename, "title": member.title}
                                  for did, member in self.document_retrievers.items()]}
        if paper_id not in self.document_retrievers:
            raise ValueError("只允许访问会话选中的文档")
        member = self.document_retrievers[paper_id]
        return {**member.get_paper_data(paper_id), "document_ids": [paper_id],
                "documents": [{"paper_id": paper_id, "filename": member.document.filename, "title": member.title}]}

    def _candidates(self, order, candidate_k):
        """Use global branch ranks; reserve candidate opportunities per document."""
        counts, ranks = defaultdict(int), {}
        for rank, index in enumerate(order, 1):
            if index < 0:
                continue
            index = int(index)
            did = self.chunks[index]["paper_id"]
            if counts[did] < candidate_k:
                ranks[index] = rank
                counts[did] += 1
        return ranks

    def _balance(self, ordered, top_k):
        buckets = {did: [] for did in self.document_ids}
        for index in ordered:
            buckets[self.chunks[index]["paper_id"]].append(index)
        chosen, position = [], 0
        while len(chosen) < min(top_k, len(ordered)):
            for did in self.document_ids:
                if position < len(buckets[did]):
                    chosen.append(buckets[did][position])
                    if len(chosen) == top_k:
                        return chosen
            position += 1
        return chosen

    def search(self, query, paper_id, top_k=5, candidate_k=50, method="hybrid"):
        self.get_paper_data(paper_id)
        if not isinstance(query, str) or not query.strip() or top_k < 1 or candidate_k < 1:
            raise ValueError("检索参数无效")
        if method not in ("bm25", "dense", "hybrid"):
            raise ValueError("不支持的检索方法")
        if paper_id != self.paper_id:
            return self.document_retrievers[paper_id].search(query, paper_id, top_k, candidate_k, method=method)
        rankings = {}
        if method in ("bm25", "hybrid"):
            lexical = np.asarray(self.bm25.get_scores(lexical_tokens(query)))
            rankings["bm25"] = self._candidates(np.argsort(-lexical, kind="stable"), candidate_k)
        if method in ("dense", "hybrid"):
            if len(self.model.tokenizer.encode(QUERY_PREFIX + query, add_special_tokens=True,
                                               truncation=False)) > self.model.max_seq_length:
                raise ValueError("检索问题超过 Embedding 输入上限，请缩短问题")
            vector = np.ascontiguousarray(self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
            _, indices = self.index.search(vector, len(self.chunks))
            rankings["dense"] = self._candidates(indices[0], candidate_k)
        scores = defaultdict(float)
        for ranks in rankings.values():
            for index, rank in ranks.items():
                scores[index] += 1 / (RRF_K + rank)
        ordered = sorted(scores, key=lambda index: (-scores[index], index))
        global_ranks = {index: rank for rank, index in enumerate(ordered, 1)}
        chosen = self._balance(ordered, top_k)
        if method=='hybrid' and self.evidence_order!='rrf':
            from src.retrieval.branch_heads import select_branch_heads
            fused={did:[self.chunks[i] for i in ordered if self.chunks[i]['paper_id']==did] for did in self.document_ids}
            heads={did:{} for did in self.document_ids}
            for did in self.document_ids:
                for branch,ranks in rankings.items():
                    owned=[i for i in ranks if self.chunks[i]['paper_id']==did]
                    heads[did][branch]=self.chunks[min(owned,key=lambda i:ranks[i])]['chunk_id'] if owned else None
            selected=select_branch_heads(self.document_ids,fused,heads,
                first_branch='bm25' if self.evidence_order=='bm25_first' else 'dense',top_k=top_k)
            positions={c['chunk_id']:i for i,c in enumerate(self.chunks)}
            chosen=[positions[c['chunk_id']] for c in selected]
        return [{**self.chunks[index], "rank": rank, "rrf_score": scores[index],
                 "global_fusion_rank": global_ranks[index], "selection": "document_balanced" if method!='hybrid' or self.evidence_order=='rrf' else self.evidence_order,
                 "bm25_rank": rankings.get("bm25", {}).get(index),
                 "dense_rank": rankings.get("dense", {}).get(index)} for rank, index in enumerate(chosen, 1)]
