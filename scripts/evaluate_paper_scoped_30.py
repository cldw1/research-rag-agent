import json
import sys
import time
from pathlib import Path

from datasets import load_dataset


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.paper_scoped_retriever import PaperScopedRetriever
from scripts.evaluate_evidence_30 import (
    collect_evidence,
    is_evidence_match
)


BM25_RESULT_PATH = (
    PROJECT_ROOT / "results" / "bm25_train_debug_30.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "paper_scoped_hybrid_30.json"
)

CACHE_DIR = (
    PROJECT_ROOT / "data" / "raw" / "hf_cache"
)


# ==========================================
# 2. 判断首次证据命中排名
# ==========================================

def first_evidence_rank(results, paper_id, gold_evidence):

    for rank, item in enumerate(results, start=1):

        if str(item["paper_id"]) != paper_id:
            continue

        if is_evidence_match(
            item["text"],
            gold_evidence
        ):
            return rank

    return None


# ==========================================
# 3. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("科研 RAG Agent - 指定论文混合检索批量评估")
    print("=" * 65)

    # --------------------------------------
    # [1] 读取原来的30道问题
    # --------------------------------------

    print("\n[1] 读取评估问题...")

    with open(BM25_RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data
    else:
        samples = data.get("detailed_results")

    if not isinstance(samples, list) or not samples:
        raise ValueError("无法读取评估问题")

    print(f"问题数量：{len(samples)}")

    # --------------------------------------
    # [2] 读取人工标准证据
    # --------------------------------------

    print("\n[2] 读取 QASPER 人工标准证据...")

    base_url = (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/refs%2Fconvert%2Fparquet/qasper"
    )

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{base_url}/train/0000.parquet"
        },
        split="train",
        cache_dir=str(CACHE_DIR)
    )

    needed_keys = {
        (
            str(sample.get("paper_id")
                or sample.get("target_paper_id")),
            str(sample.get("question_id"))
        )
        for sample in samples
    }

    gold_lookup = {}

    for paper in dataset:

        paper_id = str(paper["id"])
        qas = paper["qas"]

        for i, question_id in enumerate(qas["question_id"]):

            key = (paper_id, str(question_id))

            if key in needed_keys:

                gold_lookup[key] = collect_evidence(
                    qas["answers"][i]
                )

    print(f"找到标准问题数量：{len(gold_lookup)}")

    # --------------------------------------
    # [3] 初始化检索器
    # --------------------------------------

    print("\n[3] 初始化指定论文混合检索器...")

    retriever = PaperScopedRetriever(rrf_k=60)

    # --------------------------------------
    # [4] 批量评估
    # --------------------------------------

    print("\n[4] 开始批量评估...")

    hit_counts = {
        1: 0,
        3: 0,
        5: 0,
        10: 0
    }

    reciprocal_rank_sum = 0.0

    detailed_results = []

    start_time = time.perf_counter()

    for i, sample in enumerate(samples, start=1):

        question = (
            sample.get("question")
            or sample.get("question_text")
        )

        paper_id = str(
            sample.get("paper_id")
            or sample.get("target_paper_id")
            or ""
        )

        question_id = str(
            sample.get("question_id", "")
        )

        if not question or not paper_id or not question_id:
            raise ValueError(
                f"第 {i} 道问题缺少必要信息"
            )

        key = (paper_id, question_id)

        gold_evidence = gold_lookup.get(key, [])

        if not gold_evidence:
            raise ValueError(
                f"第 {i} 道问题没有标准证据"
            )

        # 指定论文范围内的 BM25 + Dense + RRF
        results = retriever.search(
            query=question,
            paper_id=paper_id,
            top_k=10
        )

        first_rank = first_evidence_rank(
            results,
            paper_id,
            gold_evidence
        )

        if first_rank is not None:

            reciprocal_rank_sum += 1.0 / first_rank

            for k in hit_counts:

                if first_rank <= k:
                    hit_counts[k] += 1

        detailed_results.append({
            "question_id": question_id,
            "paper_id": paper_id,
            "question": question,
            "first_evidence_rank": first_rank,
            "retrieved_chunk_ids": [
                item["chunk_id"]
                for item in results
            ]
        })

        print(
            f"完成 {i}/{len(samples)} 道问题"
            f" | 标准证据首次排名：{first_rank}"
        )

    elapsed = time.perf_counter() - start_time

    # --------------------------------------
    # [5] 计算指标
    # --------------------------------------

    print("\n[5] Paper-Scoped Hybrid 评估结果")

    n = len(samples)

    metrics = {}

    for k in hit_counts:

        value = hit_counts[k] / n

        metrics[f"Hit@{k}"] = value

        print(f"Hit@{k}: {value:.4f}")

    mrr = reciprocal_rank_sum / n

    metrics["MRR@10"] = mrr

    print(f"MRR@10: {mrr:.4f}")

    print(
        f"\n批量检索耗时：{elapsed:.2f} 秒"
    )

    # --------------------------------------
    # [6] 保存结果
    # --------------------------------------

    print("\n[6] 保存结果...")

    output = {
        "dataset": "QASPER train",
        "question_count": n,
        "retrieval_scope": "known_paper",
        "method": "Paper-Scoped BM25 + Dense + RRF",
        "rrf_k": 60,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
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

    print(f"保存位置：{OUTPUT_PATH}")

    print("\n说明：")
    print("本次评估使用已知论文 ID 作为检索范围。")
    print("论文 ID 是检索条件，不是自动预测结果。")
    print("仅使用训练集30道问题进行调试。")
    print("不是最终测试集实验结果。")

    print("\n" + "=" * 65)
    print("指定论文混合检索批量评估完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()