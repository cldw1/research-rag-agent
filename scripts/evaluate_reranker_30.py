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

from src.retrieval.reranker import Reranker
from scripts.evaluate_evidence_30 import (
    collect_evidence,
    first_evidence_rank
)


BM25_RESULT_PATH = (
    PROJECT_ROOT / "results" / "bm25_train_debug_30.json"
)

BASELINE_PATH = (
    PROJECT_ROOT / "results" / "evidence_comparison_30.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "reranker_evidence_30.json"
)

CACHE_DIR = (
    PROJECT_ROOT / "data" / "raw" / "hf_cache"
)


# ==========================================
# 2. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("科研 RAG Agent - Reranker 标准证据批量评估")
    print("=" * 65)

    # --------------------------------------
    # [1] 读取相同的30道问题
    # --------------------------------------

    print("\n[1] 读取评估问题...")

    with open(BM25_RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data
    else:
        samples = data.get("detailed_results")

    if not isinstance(samples, list) or not samples:
        raise ValueError("未找到有效评估问题")

    print(f"问题数量：{len(samples)}")

    # --------------------------------------
    # [2] 读取 QASPER 标准证据
    # --------------------------------------

    print("\n[2] 读取 QASPER 标准证据...")

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
    # [3] 初始化 Reranker
    # --------------------------------------

    print("\n[3] 初始化 Reranker...")

    reranker = Reranker()

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

    total_start = time.perf_counter()

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
                f"第 {i} 道问题缺少必要字段"
            )

        key = (paper_id, question_id)

        gold_evidence = gold_lookup.get(key, [])

        if not gold_evidence:
            raise ValueError(
                f"未找到第 {i} 道问题的标准证据"
            )

        # Hybrid Top-50 → Cross-Encoder → Top-10
        results = reranker.search(
            query=question,
            top_k=10,
            candidate_k=50
        )

        # 与前面完全相同的证据匹配方法
        first_rank = first_evidence_rank(
            results,
            paper_id,
            gold_evidence,
            reranker.retriever
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
            "retrieved_chunks": [
                {
                    "rank": item["rank"],
                    "hybrid_rank": item["hybrid_rank"],
                    "chunk_id": item["chunk_id"],
                    "paper_id": item["paper_id"],
                    "reranker_score": item["reranker_score"]
                }
                for item in results
            ]
        })

        print(
            f"\n已完成 {i}/{len(samples)} 道问题"
            f" | 标准证据首次排名：{first_rank}"
        )

    total_seconds = time.perf_counter() - total_start

    # --------------------------------------
    # [5] 汇总指标
    # --------------------------------------

    print("\n[5] Hybrid + Reranker 评估结果")

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
        f"\n批量评估耗时："
        f"{total_seconds / 60:.2f} 分钟"
    )

    # --------------------------------------
    # [6] 对比 Hybrid 基线
    # --------------------------------------

    print("\n[6] 与 Hybrid 基线比较")

    baseline_metrics = None

    if BASELINE_PATH.exists():

        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)

        baseline_metrics = (
            baseline_data
            .get("metrics", {})
            .get("Hybrid")
        )

    if baseline_metrics:

        for metric_name in (
            "Hit@1",
            "Hit@3",
            "Hit@5",
            "Hit@10",
            "MRR@10"
        ):

            before = baseline_metrics.get(metric_name)

            after = metrics[metric_name]

            if before is not None:

                print(
                    f"{metric_name}: "
                    f"{before:.4f} -> {after:.4f}"
                )

    else:

        print("未读取到 Hybrid 基线指标")
        print("本次 Reranker 结果仍会正常保存")

    # --------------------------------------
    # [7] 保存结果
    # --------------------------------------

    print("\n[7] 保存评估结果...")

    output = {
        "dataset": "QASPER train",
        "question_count": n,
        "method": "Hybrid + Cross-Encoder",
        "reranker_model": (
            "cross-encoder/ms-marco-MiniLM-L6-v2"
        ),
        "candidate_k": 50,
        "top_k": 10,
        "elapsed_seconds": total_seconds,
        "metrics": metrics,
        "hybrid_baseline": baseline_metrics,
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
    print("本次仍采用训练集30道调试问题。")
    print("使用与此前相同的人工证据文本匹配规则。")
    print("不是最终 QASPER 测试集成绩。")

    print("\n" + "=" * 65)
    print("Reranker 批量评估完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()