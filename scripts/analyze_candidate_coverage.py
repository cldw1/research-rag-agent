import json
import sys
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import HybridRetriever
from scripts.evaluate_evidence_30 import (
    collect_evidence,
    first_evidence_rank
)


BM25_RESULT_PATH = (
    PROJECT_ROOT / "results" / "bm25_train_debug_30.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "candidate_coverage_30.json"
)

CACHE_DIR = (
    PROJECT_ROOT / "data" / "raw" / "hf_cache"
)


def main():

    print("=" * 65)
    print("科研 RAG Agent - Reranker 候选证据覆盖率分析")
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
        raise ValueError("没有找到评估问题")

    print(f"问题数量：{len(samples)}")

    # --------------------------------------
    # [2] 读取标准证据
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

    print(f"匹配到标准问题：{len(gold_lookup)}")

    # --------------------------------------
    # [3] 初始化混合检索器
    # --------------------------------------

    print("\n[3] 初始化检索器...")

    retriever = HybridRetriever(rrf_k=60)

    methods = [
        "BM25@50",
        "Dense@50",
        "Hybrid@50",
        "Union@100"
    ]

    coverage_counts = {
        method: 0
        for method in methods
    }

    detailed_results = []

    # --------------------------------------
    # [4] 批量分析候选覆盖率
    # --------------------------------------

    print("\n[4] 开始分析候选证据覆盖率...")

    for i, sample in enumerate(samples, start=1):

        question = (
            sample.get("question")
            or sample.get("question_text")
        )

        paper_id = str(
            sample.get("paper_id")
            or sample.get("target_paper_id")
        )

        question_id = str(sample.get("question_id"))

        key = (paper_id, question_id)

        gold_evidence = gold_lookup.get(key, [])

        if not question or not gold_evidence:
            raise ValueError(
                f"第 {i} 道问题缺少问题文本或标准证据"
            )

        # 分别取前50个候选
        bm25_results = retriever.bm25.search(
            question,
            top_k=50
        )

        dense_results = retriever.dense.search(
            question,
            top_k=50
        )

        hybrid_results = retriever.search(
            question,
            top_k=50,
            candidate_k=50
        )

        # 构建 BM25 与 Dense 的候选并集
        union_results = []
        seen_chunk_ids = set()

        for item in bm25_results + dense_results:

            normalized = retriever.normalize_result(item)

            chunk_id = normalized.get("chunk_id")

            if not chunk_id:
                raise ValueError("候选缺少 chunk_id")

            if chunk_id not in seen_chunk_ids:

                seen_chunk_ids.add(chunk_id)

                union_results.append(normalized)

        result_groups = {
            "BM25@50": bm25_results,
            "Dense@50": dense_results,
            "Hybrid@50": hybrid_results,
            "Union@100": union_results
        }

        question_coverage = {}

        for method, results in result_groups.items():

            rank = first_evidence_rank(
                results,
                paper_id,
                gold_evidence,
                retriever
            )

            covered = rank is not None

            question_coverage[method] = covered

            if covered:
                coverage_counts[method] += 1

        detailed_results.append({
            "question_id": question_id,
            "paper_id": paper_id,
            "question": question,
            "union_candidate_count": len(union_results),
            "coverage": question_coverage
        })

        print(
            f"完成 {i}/{len(samples)}"
            f" | BM25={question_coverage['BM25@50']}"
            f" | Dense={question_coverage['Dense@50']}"
            f" | Hybrid={question_coverage['Hybrid@50']}"
            f" | Union={question_coverage['Union@100']}"
        )

    # --------------------------------------
    # [5] 汇总结果
    # --------------------------------------

    print("\n[5] 候选证据覆盖率结果")

    n = len(samples)

    coverage_metrics = {}

    for method in methods:

        count = coverage_counts[method]
        rate = count / n

        coverage_metrics[method] = {
            "count": count,
            "rate": rate
        }

        print(
            f"{method}: {count}/{n} = {rate:.4f}"
        )

    # --------------------------------------
    # [6] 保存结果
    # --------------------------------------

    print("\n[6] 保存分析结果...")

    output = {
        "dataset": "QASPER train",
        "question_count": n,
        "candidate_k": 50,
        "rrf_k": 60,
        "evaluation_type": "gold_evidence_candidate_coverage",
        "metrics": coverage_metrics,
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
    print("Union 是两路检索候选的去重并集，最多100个。")
    print("该指标只衡量候选中是否包含标准证据。")
    print("不代表最终排序后的 Hit@10。")

    print("\n" + "=" * 65)
    print("候选证据覆盖率分析完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()