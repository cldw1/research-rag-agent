import json
import re
import sys
from pathlib import Path

from datasets import load_dataset


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_retriever import HybridRetriever


BM25_RESULT_PATH = (
    PROJECT_ROOT / "results" / "bm25_train_debug_30.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "evidence_comparison_30.json"
)

CACHE_DIR = (
    PROJECT_ROOT / "data" / "raw" / "hf_cache"
)


# ==========================================
# 2. 文本标准化
# ==========================================

def normalize_text(text):

    text = str(text).lower().strip()

    return re.sub(r"\s+", " ", text)


# ==========================================
# 3. 提取 QASPER 人工标注证据
# ==========================================

def collect_evidence(obj):

    evidence = []

    if isinstance(obj, dict):

        if "evidence" in obj:

            value = obj["evidence"]

            if isinstance(value, str):
                evidence.append(value)

            elif isinstance(value, list):
                evidence.extend(
                    x for x in value
                    if isinstance(x, str)
                )

        for value in obj.values():
            evidence.extend(collect_evidence(value))

    elif isinstance(obj, list):

        for value in obj:
            evidence.extend(collect_evidence(value))

    # 去重并删除空文本
    return list(
        dict.fromkeys(
            text.strip()
            for text in evidence
            if text and text.strip()
        )
    )


# ==========================================
# 4. 判断 Chunk 是否命中标准证据
# ==========================================

def is_evidence_match(chunk_text, gold_evidence):

    chunk_text = normalize_text(chunk_text)

    if not chunk_text:
        return False

    for evidence in gold_evidence:

        evidence = normalize_text(evidence)

        if not evidence:
            continue

        # 标准证据完整出现在 Chunk 中
        if evidence in chunk_text:
            return True

        # Chunk 是标准证据的较长片段
        if len(chunk_text) >= 80 and chunk_text in evidence:
            return True

    return False


# ==========================================
# 5. 获取第一次证据命中排名
# ==========================================

def first_evidence_rank(
    results,
    target_paper_id,
    gold_evidence,
    retriever
):

    for rank, item in enumerate(results, start=1):

        item = retriever.normalize_result(item)

        paper_id = str(item.get("paper_id", ""))

        # 标准证据必须来自目标论文
        if paper_id != target_paper_id:
            continue

        if is_evidence_match(
            item.get("text", ""),
            gold_evidence
        ):
            return rank

    return None


# ==========================================
# 6. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("科研 RAG Agent - 三种检索方法标准证据评估")
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
        raise ValueError("无法读取 BM25 调试问题")

    print(f"评估问题数量：{len(samples)}")

    # --------------------------------------
    # [2] 读取 QASPER 人工标准证据
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

    gold_lookup = {}

    for paper in dataset:

        paper_id = str(paper["id"])

        qas = paper["qas"]

        question_ids = qas["question_id"]

        answers = qas["answers"]

        for i, question_id in enumerate(question_ids):

            key = (paper_id, str(question_id))

            gold_lookup[key] = collect_evidence(
                answers[i]
            )

    print(f"标准问题数量：{len(gold_lookup)}")

    # --------------------------------------
    # [3] 初始化检索器
    # --------------------------------------

    print("\n[3] 初始化检索器...")

    hybrid = HybridRetriever(rrf_k=60)

    methods = ["BM25", "Dense", "Hybrid"]

    hit_counts = {
        method: {1: 0, 3: 0, 5: 0, 10: 0}
        for method in methods
    }

    reciprocal_rank_sum = {
        method: 0.0
        for method in methods
    }

    detailed_results = []

    # --------------------------------------
    # [4] 批量评估
    # --------------------------------------

    print("\n[4] 开始批量评估...")

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
                f"第 {i} 道题缺少必要信息"
            )

        key = (paper_id, question_id)

        gold_evidence = gold_lookup.get(key, [])

        if not gold_evidence:
            raise ValueError(
                f"第 {i} 道题未找到人工标准证据：{key}"
            )

        # 三种方法分别检索
        bm25_results = hybrid.bm25.search(
            question,
            top_k=10
        )

        dense_results = hybrid.dense.search(
            question,
            top_k=10
        )

        hybrid_results = hybrid.search(
            question,
            top_k=10,
            candidate_k=50
        )

        retrieval_results = {
            "BM25": bm25_results,
            "Dense": dense_results,
            "Hybrid": hybrid_results
        }

        ranks = {}

        for method, results in retrieval_results.items():

            rank = first_evidence_rank(
                results,
                paper_id,
                gold_evidence,
                hybrid
            )

            ranks[method] = rank

            if rank is not None:

                reciprocal_rank_sum[method] += 1.0 / rank

                for k in (1, 3, 5, 10):

                    if rank <= k:
                        hit_counts[method][k] += 1

        detailed_results.append({
            "question_id": question_id,
            "paper_id": paper_id,
            "question": question,
            "first_evidence_rank": ranks
        })

        print(
            f"完成 {i}/{len(samples)}"
            f" | BM25={ranks['BM25']}"
            f" | Dense={ranks['Dense']}"
            f" | Hybrid={ranks['Hybrid']}"
        )

    # --------------------------------------
    # [5] 输出结果
    # --------------------------------------

    print("\n[5] 三种检索方法证据命中率对比")

    n = len(samples)

    all_metrics = {}

    for method in methods:

        print("\n" + "-" * 45)
        print(method)

        metrics = {}

        for k in (1, 3, 5, 10):

            value = hit_counts[method][k] / n

            metrics[f"Hit@{k}"] = value

            print(f"Hit@{k}: {value:.4f}")

        mrr = reciprocal_rank_sum[method] / n

        metrics["MRR@10"] = mrr

        print(f"MRR@10: {mrr:.4f}")

        all_metrics[method] = metrics

    # --------------------------------------
    # [6] 保存结果
    # --------------------------------------

    print("\n[6] 保存结果...")

    output = {
        "dataset": "QASPER train",
        "question_count": n,
        "evaluation_type": "gold_evidence_text_match",
        "candidate_k_hybrid": 50,
        "rrf_k": 60,
        "metrics": all_metrics,
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

    print("\n说明：")
    print("本次采用标准证据文本包含匹配。")
    print("这是一种调试用的证据命中判定方法，")
    print("不等同于 QASPER 官方完整评测。")
    print("训练集结果不能作为最终测试集成绩。")

    print("\n" + "=" * 65)
    print("三种检索方法标准证据评估完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()