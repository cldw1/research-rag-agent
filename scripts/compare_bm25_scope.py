import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import (
    BM25Retriever,
    tokenize
)

from scripts.evaluate_bm25_batch import (
    extract_gold_evidences,
    is_evidence_match
)


def calculate_metrics(ranks):

    total = len(ranks)

    metrics = {}

    for k in [1, 3, 5, 10]:

        hits = sum(
            rank is not None and rank <= k
            for rank in ranks
        )

        metrics[f"Hit@{k}"] = hits / total

    metrics["MRR@10"] = sum(
        1 / rank
        for rank in ranks
        if rank is not None and rank <= 10
    ) / total

    return metrics


def main():

    print("=" * 60)
    print("科研 RAG Agent - BM25 检索范围对照实验")
    print("=" * 60)

    # ==================================
    # 1. 读取之前的30道评测问题
    # ==================================

    print("\n[1] 读取已有实验结果...")

    result_path = (
        PROJECT_ROOT
        / "results"
        / "bm25_train_debug_30.json"
    )

    with open(result_path, "r", encoding="utf-8") as f:
        old_report = json.load(f)

    cases = old_report["detailed_results"]

    print("评测问题数量:", len(cases))

    target_paper_ids = {
        case["paper_id"]
        for case in cases
    }

    # ==================================
    # 2. 获取这些问题的标准证据
    # ==================================

    print("\n[2] 加载标准证据...")

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

    gold_lookup = {}

    for paper in dataset:

        paper_id = paper["id"]

        if paper_id not in target_paper_ids:
            continue

        qas = paper["qas"]

        for i, question_id in enumerate(
            qas["question_id"]
        ):

            gold_lookup[
                (paper_id, question_id)
            ] = extract_gold_evidences(
                qas["answers"][i]
            )

    print("标准答案读取完成！")

    # ==================================
    # 3. 构建已有的BM25索引
    # ==================================

    print("\n[3] 构建 BM25 索引...")

    corpus_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "qasper_train_chunks.jsonl"
    )

    retriever = BM25Retriever(corpus_path)

    # 为每篇论文记录对应的Chunk索引位置
    paper_to_indices = defaultdict(list)

    for index, chunk in enumerate(retriever.chunks):

        paper_to_indices[
            chunk["paper_id"]
        ].append(index)

    # ==================================
    # 4. 开展限定论文检索
    # ==================================

    print("\n[4] 开始对照实验...")

    global_ranks = []
    paper_ranks = []

    detailed_results = []

    for number, case in enumerate(cases, start=1):

        paper_id = case["paper_id"]
        question_id = case["question_id"]
        question = case["question"]

        gold_evidences = gold_lookup.get(
            (paper_id, question_id),
            []
        )

        # 方法A：读取此前保存的全库检索结果
        global_rank = case["first_hit_rank"]

        global_ranks.append(global_rank)

        # 方法B：仅在目标论文内检索
        doc_ids = paper_to_indices[paper_id]

        query_tokens = tokenize(question)

        first_hit_rank = None

        if doc_ids and query_tokens:

            # 仅对目标论文的Chunk计算BM25分数
            scores = retriever.bm25.get_batch_scores(
                query_tokens,
                doc_ids
            )

            top_positions = np.argsort(scores)[::-1][:10]

            for rank, position in enumerate(
                top_positions,
                start=1
            ):

                position = int(position)

                if scores[position] <= 0:
                    continue

                chunk_index = doc_ids[position]

                chunk = retriever.chunks[chunk_index]

                is_hit = any(
                    is_evidence_match(
                        chunk["text"],
                        evidence
                    )
                    for evidence in gold_evidences
                )

                if is_hit:
                    first_hit_rank = rank
                    break

        paper_ranks.append(first_hit_rank)

        detailed_results.append({
            "paper_id": paper_id,
            "question_id": question_id,
            "question": question,
            "global_first_hit_rank": global_rank,
            "paper_first_hit_rank": first_hit_rank
        })

        if number % 5 == 0 or number == len(cases):

            print(
                f"已完成 {number}/{len(cases)} 道问题"
            )

    # ==================================
    # 5. 比较实验结果
    # ==================================

    global_metrics = calculate_metrics(global_ranks)

    paper_metrics = calculate_metrics(paper_ranks)

    print("\n[5] BM25 检索范围对比")

    print("-" * 60)

    print(
        f"{'指标':<12}"
        f"{'全库检索':<15}"
        f"{'限定论文检索':<15}"
    )

    print("-" * 60)

    for metric in [
        "Hit@1",
        "Hit@3",
        "Hit@5",
        "Hit@10",
        "MRR@10"
    ]:

        print(
            f"{metric:<12}"
            f"{global_metrics[metric]:<15.4f}"
            f"{paper_metrics[metric]:<15.4f}"
        )

    # ==================================
    # 6. 保存对照实验结果
    # ==================================

    output_path = (
        PROJECT_ROOT
        / "results"
        / "bm25_scope_comparison_30.json"
    )

    report = {
        "experiment": "BM25 Retrieval Scope Comparison",
        "dataset": "QASPER",
        "split": "train",
        "evaluation_size": len(cases),
        "retrieval_methods": [
            "global_bm25",
            "paper_scoped_bm25"
        ],
        "global_metrics": global_metrics,
        "paper_metrics": paper_metrics,
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

    print("\n[6] 保存实验结果")

    print("输出文件:", output_path)

    print("\n" + "=" * 60)
    print("BM25 检索范围对照实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()