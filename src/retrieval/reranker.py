import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import HybridRetriever


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

CACHE_DIR = (
    PROJECT_ROOT
    / "models"
    / "reranker_cache"
)


# ==========================================
# 2. Reranker
# ==========================================

class Reranker:

    def __init__(self):

        print("[1] 初始化 Hybrid Retriever...")

        self.retriever = HybridRetriever(rrf_k=60)

        print("\n[2] 加载 Cross-Encoder 模型...")

        CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model = CrossEncoder(
            MODEL_NAME,
            device="cpu",
            max_length=512,
            cache_folder=str(CACHE_DIR)
        )

        print("Cross-Encoder 模型加载成功！")

    # ======================================
    # 3. 混合检索 + 重排序
    # ======================================

    def search(
        self,
        query,
        top_k=10,
        candidate_k=50
    ):

        if not query.strip():
            return []

        # ----------------------------------
        # 第一步：Hybrid 召回候选
        # ----------------------------------

        candidates = self.retriever.search(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k
        )

        if not candidates:
            return []

        print(
            f"\nHybrid 召回候选：{len(candidates)} 个"
        )

        # ----------------------------------
        # 第二步：构造问题-文本对
        # ----------------------------------

        pairs = [
            (query, item["text"])
            for item in candidates
        ]

        # ----------------------------------
        # 第三步：Cross-Encoder 打分
        # ----------------------------------

        print("正在进行 Cross-Encoder 重排序...")

        start_time = time.perf_counter()

        scores = self.model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=True
        )

        elapsed = time.perf_counter() - start_time

        scores = np.asarray(scores).reshape(-1)

        if len(scores) != len(candidates):
            raise ValueError(
                "重排序分数与候选数量不一致"
            )

        print(f"重排序耗时：{elapsed:.2f} 秒")

        # ----------------------------------
        # 第四步：重新排序
        # ----------------------------------

        reranked = []

        for item, score in zip(candidates, scores):

            result = item.copy()

            result["hybrid_rank"] = item["rank"]

            result["reranker_score"] = float(score)

            reranked.append(result)

        reranked.sort(
            key=lambda x: (
                -x["reranker_score"],
                x["hybrid_rank"]
            )
        )

        # 取最终 Top-K
        final_results = reranked[:top_k]

        for rank, item in enumerate(
            final_results,
            start=1
        ):

            item["rank"] = rank

        return final_results


# ==========================================
# 4. 单问题实验
# ==========================================

def main():

    print("=" * 65)
    print("科研 RAG Agent - Cross-Encoder 重排序实验")
    print("=" * 65)

    reranker = Reranker()

    question = "What is the seed lexicon?"

    target_paper_id = "1909.00694"

    print("\n[3] 测试问题：")
    print(question)

    results = reranker.search(
        query=question,
        top_k=10,
        candidate_k=50
    )

    print("\n[4] Reranker 最终 Top-10")

    for item in results:

        print("\n" + "-" * 65)

        print(f"新排名：{item['rank']}")

        print(f"原 Hybrid 排名：{item['hybrid_rank']}")

        print(
            f"Reranker 分数："
            f"{item['reranker_score']:.4f}"
        )

        print(f"论文 ID：{item['paper_id']}")

        print(f"章节：{item['section_name']}")

        print(f"Chunk ID：{item['chunk_id']}")

        print("\n文本内容：")

        print(
            item["text"].replace("\n", " ")[:300]
        )

    # --------------------------------------
    # 5. 检查目标论文
    # --------------------------------------

    print("\n[5] 检查目标论文...")

    matching = [
        item
        for item in results
        if str(item["paper_id"]) == target_paper_id
    ]

    if matching:

        print("目标论文进入 Top-10：是")

        print(
            f"首次出现排名：{matching[0]['rank']}"
        )

    else:

        print("目标论文进入 Top-10：否")

    print("\n注意：")
    print("Cross-Encoder 分数用于候选排序，")
    print("不能直接当作答案正确概率。")
    print("单问题结果不能证明 Reranker 有效。")

    print("\n" + "=" * 65)
    print("Cross-Encoder 重排序实验完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()