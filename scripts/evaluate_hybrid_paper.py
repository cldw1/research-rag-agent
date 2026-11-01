import json
import sys
from pathlib import Path


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import HybridRetriever


BM25_RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "bm25_train_debug_30.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "hybrid_paper_debug_30.json"
)


# ==========================================
# 2. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("科研 RAG Agent - Hybrid Retrieval 批量评估")
    print("=" * 65)

    # --------------------------------------
    # [1] 读取之前的 30 道问题
    # --------------------------------------

    print("\n[1] 读取评估问题...")

    with open(BM25_RESULT_PATH, "r", encoding="utf-8") as f:
        bm25_data = json.load(f)

    if isinstance(bm25_data, list):

        questions = bm25_data

    else:

        questions = bm25_data.get("detailed_results")

    if not isinstance(questions, list) or not questions:
        raise ValueError("未找到有效的评估问题")

    print(f"评估问题数量：{len(questions)}")

    # --------------------------------------
    # [2] 初始化混合检索器
    # --------------------------------------

    print("\n[2] 初始化 Hybrid Retriever...")

    retriever = HybridRetriever(rrf_k=60)

    # --------------------------------------
    # [3] 开始批量评估
    # --------------------------------------

    print("\n[3] 开始批量评估...")

    hit_counts = {
        1: 0,
        3: 0,
        5: 0,
        10: 0
    }

    reciprocal_rank_sum = 0.0

    detailed_results = []

    for i, sample in enumerate(questions, start=1):

        question = (
            sample.get("question")
            or sample.get("question_text")
        )

        target_paper_id = str(
            sample.get("paper_id")
            or sample.get("target_paper_id")
            or ""
        )

        if not question or not target_paper_id:
            raise ValueError(
                f"第 {i} 道问题缺少必要字段"
            )

        # BM25 + Dense + RRF
        results = retriever.search(
            query=question,
            top_k=10,
            candidate_k=50
        )

        # 正确论文首次出现的位置
        first_correct_rank = None

        for item in results:

            if str(item["paper_id"]) == target_paper_id:

                first_correct_rank = item["rank"]

                break

        # 更新指标
        if first_correct_rank is not None:

            reciprocal_rank_sum += (
                1.0 / first_correct_rank
            )

            for k in hit_counts:

                if first_correct_rank <= k:
                    hit_counts[k] += 1

        # 保存单问题详情
        detailed_results.append(
            {
                "question_id": sample.get("question_id"),
                "question": question,
                "target_paper_id": target_paper_id,
                "first_correct_paper_rank": first_correct_rank,
                "retrieved_chunks": [
                    {
                        "rank": item["rank"],
                        "chunk_id": item["chunk_id"],
                        "paper_id": item["paper_id"],
                        "rrf_score": item["rrf_score"],
                        "bm25_rank": item["bm25_rank"],
                        "dense_rank": item["dense_rank"]
                    }
                    for item in results
                ]
            }
        )

        print(
            f"已完成 {i}/{len(questions)} 道问题"
            f" | 正确论文首次排名：{first_correct_rank}"
        )

    # --------------------------------------
    # [4] 汇总评估结果
    # --------------------------------------

    print("\n[4] Hybrid Retrieval 评估结果")

    n = len(questions)

    metrics = {}

    for k in hit_counts:

        value = hit_counts[k] / n

        metrics[f"PaperHit@{k}"] = value

        print(f"PaperHit@{k}: {value:.4f}")

    mrr = reciprocal_rank_sum / n

    metrics["PaperMRR@10"] = mrr

    print(f"PaperMRR@10: {mrr:.4f}")

    # --------------------------------------
    # [5] 保存结果
    # --------------------------------------

    print("\n[5] 保存评估结果...")

    output = {
        "method": "BM25 + Dense + RRF",
        "dataset": "QASPER train",
        "question_count": n,
        "rrf_k": 60,
        "candidate_k": 50,
        "metrics": metrics,
        "detailed_results": detailed_results
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"结果保存位置：{OUTPUT_PATH}")

    # --------------------------------------
    # [6] 结束
    # --------------------------------------

    print("\n说明：")
    print("本次评估的是正确论文的检索排名。")
    print("尚未评估标准答案证据的命中情况。")
    print("使用训练集调试问题，不代表最终测试集表现。")

    print("\n" + "=" * 65)
    print("Hybrid Retrieval 批量评估完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()