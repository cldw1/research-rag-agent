import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever


# 固定实验配置，确保结果可以复现
EVAL_SIZE = 30
RANDOM_SEED = 42
TOP_K = 10


def normalize_text(text):
    return " ".join(text.lower().split())


def is_evidence_match(chunk_text, evidence_text):

    chunk = normalize_text(chunk_text)
    evidence = normalize_text(evidence_text)

    if not chunk or not evidence:
        return False

    if evidence in chunk:
        return True

    if len(chunk) >= 80 and chunk in evidence:
        return True

    return False


def extract_gold_evidences(answer_group):

    evidences = []

    for answer in answer_group["answer"]:

        if answer["unanswerable"]:
            continue

        for evidence in answer["evidence"]:

            if not evidence:
                continue

            if evidence.strip().startswith("FLOAT SELECTED"):
                continue

            evidences.append(evidence)

    # 去除重复证据
    unique_evidences = []
    seen = set()

    for evidence in evidences:

        normalized = normalize_text(evidence)

        if normalized not in seen:
            seen.add(normalized)
            unique_evidences.append(evidence)

    return unique_evidences


def main():

    print("=" * 60)
    print("科研 RAG Agent - BM25 批量检索评测")
    print("=" * 60)

    # ==================================
    # 1. 加载数据集
    # ==================================

    print("\n[1] 加载 QASPER 训练集...")

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
        cache_dir=str(
            PROJECT_ROOT / "data" / "raw" / "hf_cache"
        )
    )

    # ==================================
    # 2. 提取带有文本证据的问题
    # ==================================

    print("\n[2] 提取评测问题...")

    examples = []

    for paper in dataset:

        paper_id = paper["id"]
        qas = paper["qas"]

        for i, question in enumerate(qas["question"]):

            evidences = extract_gold_evidences(
                qas["answers"][i]
            )

            if not evidences:
                continue

            examples.append({
                "paper_id": paper_id,
                "question_id": qas["question_id"][i],
                "question": question,
                "gold_evidences": evidences
            })

    print("具有文本证据的问题数量:", len(examples))

    # 固定随机种子
    rng = random.Random(RANDOM_SEED)

    sample_size = min(EVAL_SIZE, len(examples))

    selected_examples = rng.sample(
        examples,
        sample_size
    )

    print("本次抽取问题数量:", len(selected_examples))
    print("随机种子:", RANDOM_SEED)

    # ==================================
    # 3. 构建 BM25 检索器
    # ==================================

    print("\n[3] 构建 BM25 检索器...")

    corpus_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "qasper_train_chunks.jsonl"
    )

    retriever = BM25Retriever(corpus_path)

    # 按论文整理 Chunk，供证据覆盖检查使用
    chunks_by_paper = defaultdict(list)

    for chunk in retriever.chunks:
        chunks_by_paper[chunk["paper_id"]].append(
            chunk["text"]
        )

    # ==================================
    # 4. 批量检索
    # ==================================

    print("\n[4] 开始批量评测...")

    hit_counts = {
        1: 0,
        3: 0,
        5: 0,
        10: 0
    }

    reciprocal_rank_sum = 0.0

    aligned_count = 0

    detailed_results = []

    for index, example in enumerate(
        selected_examples,
        start=1
    ):

        question = example["question"]
        paper_id = example["paper_id"]
        gold_evidences = example["gold_evidences"]

        # 检查标准证据能否在当前知识库中匹配
        gold_available = any(
            is_evidence_match(chunk_text, evidence)
            for chunk_text in chunks_by_paper[paper_id]
            for evidence in gold_evidences
        )

        if gold_available:
            aligned_count += 1

        results = retriever.search(
            question,
            top_k=TOP_K
        )

        first_hit_rank = None

        retrieved_details = []

        for rank, result in enumerate(results, start=1):

            is_hit = False

            if result["paper_id"] == paper_id:

                is_hit = any(
                    is_evidence_match(
                        result["text"],
                        evidence
                    )
                    for evidence in gold_evidences
                )

            retrieved_details.append({
                "rank": rank,
                "chunk_id": result["chunk_id"],
                "paper_id": result["paper_id"],
                "score": round(result["score"], 4),
                "evidence_hit": is_hit
            })

            if is_hit and first_hit_rank is None:
                first_hit_rank = rank

        # 计算 Hit@K
        for k in hit_counts:

            if (
                first_hit_rank is not None
                and first_hit_rank <= k
            ):
                hit_counts[k] += 1

        # 计算 MRR@10
        if first_hit_rank is not None:
            reciprocal_rank_sum += 1 / first_hit_rank

        detailed_results.append({
            "paper_id": paper_id,
            "question_id": example["question_id"],
            "question": question,
            "gold_available_in_index": gold_available,
            "first_hit_rank": first_hit_rank,
            "retrieved": retrieved_details
        })

        if index % 5 == 0 or index == sample_size:
            print(
                f"已完成 {index}/{sample_size} 道问题"
            )

    # ==================================
    # 5. 计算最终统计结果
    # ==================================

    print("\n[5] 批量评测结果")

    metrics = {}

    for k, count in hit_counts.items():

        value = count / sample_size

        metrics[f"Hit@{k}"] = value

        print(f"Hit@{k}: {value:.4f}")

    mrr = reciprocal_rank_sum / sample_size

    metrics["MRR@10"] = mrr

    print(f"MRR@10: {mrr:.4f}")

    print(
        "\n标准证据可匹配问题数:",
        f"{aligned_count}/{sample_size}"
    )

    # ==================================
    # 6. 保存实验结果
    # ==================================

    print("\n[6] 保存评测结果...")

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / "bm25_train_debug_30.json"
    )

    report = {
        "experiment": "BM25 Training Debug Evaluation",
        "dataset": "QASPER",
        "split": "train",
        "retrieval_scope": "all_train_papers",
        "evaluation_size": sample_size,
        "random_seed": RANDOM_SEED,
        "top_k": TOP_K,
        "gold_available_in_index": aligned_count,
        "metrics": metrics,
        "detailed_results": detailed_results
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("结果保存位置:", output_path)

    print("\n注意：")
    print("本次使用训练集中的30道问题进行调试。")
    print("命中判断采用文本包含匹配。")
    print("这些指标不能作为最终测试集实验结果。")

    print("\n" + "=" * 60)
    print("BM25 批量评测完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()