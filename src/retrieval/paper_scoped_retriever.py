import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import tokenize
from src.retrieval.full_dense_retriever import FullDenseRetriever


class PaperScopedRetriever:

    def __init__(self, rrf_k=60):

        self.rrf_k = rrf_k

        print("[1] 加载论文知识库...")

        self.dense = FullDenseRetriever()

        # 建立论文 ID 到 Chunk 编号的对应关系
        self.paper_to_indices = defaultdict(list)

        for index_id, chunk in enumerate(self.dense.chunks):

            paper_id = str(chunk["paper_id"])

            self.paper_to_indices[paper_id].append(index_id)

        # 缓存已处理的论文
        self.paper_cache = {}

        print(
            f"\n可检索论文数量："
            f"{len(self.paper_to_indices)}"
        )

    def get_paper_data(self, paper_id):

        paper_id = str(paper_id)

        if paper_id in self.paper_cache:
            return self.paper_cache[paper_id]

        indices = self.paper_to_indices.get(paper_id)

        if not indices:
            raise ValueError(
                f"论文 {paper_id} 不存在或没有有效 Chunk"
            )

        chunks = [
            self.dense.chunks[i]
            for i in indices
        ]

        # 只使用当前论文的文本构建 BM25
        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        # 从已有 FAISS 索引中提取当前论文的向量
        vectors = np.vstack([
            self.dense.index.reconstruct(int(i))
            for i in indices
        ]).astype(np.float32)

        # 确保向量归一化
        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        vectors = vectors / np.maximum(norms, 1e-12)

        paper_data = {
            "chunks": chunks,
            "bm25": bm25,
            "vectors": vectors
        }

        self.paper_cache[paper_id] = paper_data

        return paper_data

    def search(
        self,
        query,
        paper_id,
        top_k=10,
        candidate_k=None
    ):

        if not query.strip():
            return []

        paper_data = self.get_paper_data(paper_id)

        chunks = paper_data["chunks"]

        n = len(chunks)

        if candidate_k is None:
            candidate_k = n

        candidate_k = min(candidate_k, n)

        # ----------------------------------
        # 1. 当前论文 BM25 检索
        # ----------------------------------

        bm25_scores = paper_data["bm25"].get_scores(
            tokenize(query)
        )

        bm25_order = sorted(
            range(n),
            key=lambda i: (-float(bm25_scores[i]), i)
        )[:candidate_k]

        # ----------------------------------
        # 2. 当前论文 Dense 检索
        # ----------------------------------

        instruction = (
            "Represent this sentence for searching "
            "relevant passages: "
        )

        query_vector = self.dense.model.encode(
            [instruction + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )[0].astype(np.float32)

        dense_scores = (
            paper_data["vectors"] @ query_vector
        )

        dense_order = sorted(
            range(n),
            key=lambda i: (-float(dense_scores[i]), i)
        )[:candidate_k]

        # ----------------------------------
        # 3. RRF 融合
        # ----------------------------------

        fused = {}

        for source, order in (
            ("bm25", bm25_order),
            ("dense", dense_order)
        ):

            for rank, local_id in enumerate(
                order,
                start=1
            ):

                if local_id not in fused:

                    chunk = chunks[local_id]

                    fused[local_id] = {
                        "paper_id": chunk["paper_id"],
                        "title": chunk.get("title", ""),
                        "section_name": chunk.get(
                            "section_name", ""
                        ),
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "rrf_score": 0.0,
                        "bm25_rank": None,
                        "dense_rank": None
                    }

                item = fused[local_id]

                item["rrf_score"] += (
                    1.0 / (self.rrf_k + rank)
                )

                item[f"{source}_rank"] = rank

        # ----------------------------------
        # 4. 最终排序
        # ----------------------------------

        ranked = sorted(
            fused.values(),
            key=lambda x: (
                -x["rrf_score"],
                x["chunk_id"]
            )
        )

        final_results = ranked[:top_k]

        for rank, item in enumerate(
            final_results,
            start=1
        ):

            item["rank"] = rank

        return final_results


def main():

    print("=" * 65)
    print("科研 RAG Agent - 指定论文混合检索")
    print("=" * 65)

    retriever = PaperScopedRetriever(rrf_k=60)

    question = "What is the seed lexicon?"

    paper_id = "1909.00694"

    print("\n[2] 测试问题")

    print(f"论文 ID：{paper_id}")
    print(f"问题：{question}")

    paper_data = retriever.get_paper_data(paper_id)

    print(
        f"当前论文有效 Chunk："
        f"{len(paper_data['chunks'])}"
    )

    print("\n[3] 开始指定论文混合检索...")

    results = retriever.search(
        query=question,
        paper_id=paper_id,
        top_k=10
    )

    print("\n[4] 检索结果")

    for item in results:

        print("\n" + "-" * 65)

        print(f"排名：{item['rank']}")

        print(f"RRF 分数：{item['rrf_score']:.6f}")

        print(f"BM25 排名：{item['bm25_rank']}")

        print(f"Dense 排名：{item['dense_rank']}")

        print(f"论文 ID：{item['paper_id']}")

        print(f"章节：{item['section_name']}")

        print(f"Chunk ID：{item['chunk_id']}")

        print("\n文本内容：")

        print(
            item["text"].replace("\n", " ")[:300]
        )

    print("\n" + "=" * 65)
    print("指定论文混合检索实验完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()