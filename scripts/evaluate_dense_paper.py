import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.full_dense_retriever import FullDenseRetriever


BM25_RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "bm25_train_debug_30.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "dense_paper_debug_30.json"
)


def main():

    print("=" * 65)
    print("科研 RAG Agent - Dense Retrieval 论文定位评估")
    print("=" * 65)

    # 读取之前 BM25 使用的相同问题
    print("\n[1] 读取 BM25 评估问题...")

    with open(BM25_RESULT_PATH, "r", encoding="utf-8") as f:
        bm25_data = json.load(f)

    if isinstance(bm25_data, list):
        questions = bm25_data
    else:
        questions = bm25_data.get("detailed_results")

    if not isinstance(questions, list) or not questions:
        raise ValueError(
            "未找到详细评估问题，请检查 BM25 结果文件结构。"
        )

    print(f"评估问题数量：{len(questions)}")

    # 加载 Dense Retriever
    print("\n[2] 初始化 Dense Retriever...")

    retriever = FullDenseRetriever()

    # 批量评估
    print("\n[3] 开始批量评估...")

    detailed_results = []

    hit_counts = {
        1: 0,
        3: 0,
        5: 0,
        10: 0
    }

    reciprocal_rank_sum = 0.0

    for i, sample in enumerate(questions, start=1):

        question = sample.get("question") or sample.get(
            "question_text"
        )

        target_paper_id = str(
            sample.get("paper_id")
            or sample.get("target_paper_id")
            or ""
        )

        if not question or not target_paper_id:
            raise ValueError(
                f"第 {i} 道题缺少 question 或 paper_id"
            )

        results = retriever.search(
            question,
            top_k=10
        )

        first_correct_rank = None

        for item in results:

            if str(item["paper_id"]) == target_paper_id:

                first_correct_rank = item["rank"]
                break

        if first_correct_rank is not None:

            reciprocal_rank_sum += 1.0 / first_correct_rank

            for k in hit_counts:

                if first_correct_rank <= k:
                    hit_counts[k] += 1

        detailed_results.append(
            {
                "question_id": sample.get("question_id"),
                "question": question,
                "target_paper_id": target_paper_id,
                "first_correct_paper_rank": first_correct_rank,
                "retrieved_paper_ids": [
                    item["paper_id"]
                    for item in results
                ],
                "retrieved_chunk_ids": [
                    item["chunk_id"]
                    for item in results
                ]
            }
        )

        print(
            f"已完成 {i}/{len(questions)} 道问题"
            f" | 正确论文首次排名：{first_correct_rank}"
        )

    # 汇总指标
    print("\n[4] 论文定位评估结果")

    n = len(questions)

    metrics = {}

    for k in hit_counts:

        value = hit_counts[k] / n

        metrics[f"PaperHit@{k}"] = value

        print(f"PaperHit@{k}: {value:.4f}")

    mrr = reciprocal_rank_sum / n

    metrics["PaperMRR@10"] = mrr

    print(f"PaperMRR@10: {mrr:.4f}")

    # 保存结果
    print("\n[5] 保存评估结果...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "evaluation_type": "paper_retrieval",
        "dataset": "QASPER train",
        "method": "Dense Retrieval",
        "model": "BAAI/bge-small-en-v1.5",
        "question_count": n,
        "metrics": metrics,
        "detailed_results": detailed_results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"结果保存位置：{OUTPUT_PATH}")

    print("\n说明：")
    print("本次只评估正确论文是否进入 Top-K。")
    print("尚未评估标准证据是否被命中。")
    print("本次使用训练集调试问题，不是最终测试集结果。")

    print("\n" + "=" * 65)
    print("Dense Retrieval 论文定位评估完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()