import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import tokenize


CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "qasper_train_chunks.jsonl"
)

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "qasper_train.faiss"
)

MODEL_CACHE_DIR = (
    PROJECT_ROOT
    / "models"
    / "embedding_cache"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ==========================================
# 2. 完整知识库 Dense Retriever
# ==========================================

class FullDenseRetriever:

    def __init__(self):

        print("[1] 加载完整 FAISS 索引...")

        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"找不到 FAISS 索引：{INDEX_PATH}"
            )

        self.index = faiss.read_index(str(INDEX_PATH))

        print(f"索引向量数量：{self.index.ntotal}")
        print(f"向量维度：{self.index.d}")

        # ----------------------------------
        # 加载 Chunk 元数据
        # ----------------------------------

        print("\n[2] 加载 Chunk 元数据...")

        self.chunks = []

        with open(CORPUS_PATH, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                chunk = json.loads(line)

                text = chunk.get("text", "")

                # 必须与之前生成向量时的筛选规则一致
                if tokenize(text):
                    self.chunks.append(chunk)

        print(f"有效 Chunk 数量：{len(self.chunks)}")

        # ----------------------------------
        # 验证 Chunk 和向量一一对应
        # ----------------------------------

        if len(self.chunks) != self.index.ntotal:

            raise ValueError(
                "Chunk 数量与 FAISS 向量数量不一致！"
                f"Chunk={len(self.chunks)}，"
                f"向量={self.index.ntotal}"
            )

        print("Chunk 与向量数量一致！")

        # ----------------------------------
        # 加载 Embedding 模型
        # ----------------------------------

        print("\n[3] 加载 Embedding 模型...")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu",
            cache_folder=str(MODEL_CACHE_DIR)
        )

        print("Embedding 模型加载成功！")

    # ======================================
    # 3. 语义检索
    # ======================================

    def search(self, query, top_k=10):

        # BGE 模型的问题检索提示词
        instruction = (
            "Represent this sentence for searching "
            "relevant passages: "
        )

        query_text = instruction + query

        # 将问题转换为向量
        query_vector = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        query_vector = np.ascontiguousarray(
            query_vector,
            dtype=np.float32
        )

        faiss.normalize_L2(query_vector)

        # 在完整 FAISS 索引中搜索
        scores, indices = self.index.search(
            query_vector,
            min(top_k, self.index.ntotal)
        )

        results = []

        for rank, (score, index_id) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            index_id = int(index_id)

            if index_id < 0:
                continue

            chunk = self.chunks[index_id]

            result = {
                "rank": rank,
                "score": float(score),
                "faiss_id": index_id,
                "paper_id": chunk.get("paper_id", ""),
                "title": chunk.get("title", ""),
                "section_name": chunk.get("section_name", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", "")
            }

            results.append(result)

        return results


# ==========================================
# 4. 测试完整知识库检索
# ==========================================

def main():

    print("=" * 65)
    print("科研 RAG Agent - 完整知识库 Dense Retrieval 测试")
    print("=" * 65)

    retriever = FullDenseRetriever()

    # 与之前 BM25 实验使用相同的问题
    question = "What is the seed lexicon?"

    # 这个问题来自的原始论文
    target_paper_id = "1909.00694"

    print("\n[4] 开始完整知识库语义检索...")

    print(f"用户问题：{question}")
    print(f"目标论文 ID：{target_paper_id}")

    results = retriever.search(
        query=question,
        top_k=10
    )

    print(f"\n检索结果数量：{len(results)}")

    print("\n[5] Top-10 检索结果")

    for item in results:

        print("\n" + "-" * 65)

        print(f"排名：{item['rank']}")

        print(f"语义相似度：{item['score']:.4f}")

        print(f"论文 ID：{item['paper_id']}")

        print(f"论文标题：{item['title']}")

        print(f"章节：{item['section_name']}")

        print(f"Chunk ID：{item['chunk_id']}")

        print("\n文本内容：")

        preview = item["text"].replace("\n", " ")

        print(preview[:300])

    # ======================================
    # 5. 观察目标论文是否被找到
    # ======================================

    print("\n[6] 检查目标论文...")

    target_results = [
        item
        for item in results
        if str(item["paper_id"]) == target_paper_id
    ]

    if target_results:

        first_rank = target_results[0]["rank"]

        print("目标论文进入 Top-10：是")
        print(f"目标论文首次出现排名：{first_rank}")

    else:

        print("目标论文进入 Top-10：否")

    print(
        "注意：找到正确论文，不代表一定找到正确证据。"
    )

    print("\n" + "=" * 65)
    print("完整知识库 Dense Retrieval 测试完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()