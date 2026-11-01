
import json
import sys
from pathlib import Path
from statistics import mean

from datasets import load_dataset


# ==========================================
# 1. 项目根目录
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.paper_scoped_retriever import (
    PaperScopedRetriever
)

from src.generation.context_builder import (
    build_context
)

from scripts.evaluate_context_expansion import (
    expand_neighbors
)

from scripts.diagnose_rag_failures import (
    matches_gold_evidence
)


# ==========================================
# 2. 实验配置
# ==========================================

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "bm25_train_debug_30.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "context_expansion_ablation_30.json"
)

TOP_K = 5
WINDOW = 1
MAX_CHARS = 8000
MAX_QUESTIONS = 30


# ==========================================
# 3. 读取之前固定的 30 道题
# ==========================================

def load_questions():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    records = data["detailed_results"]

    questions = []

    for item in records:

        if not all(
            item.get(key)
            for key in (
                "paper_id",
                "question_id",
                "question"
            )
        ):
            continue

        questions.append({
            "paper_id": str(item["paper_id"]),
            "question_id": str(item["question_id"]),
            "question": str(item["question"])
        })

    return questions[:MAX_QUESTIONS]


# ==========================================
# 4. 提取 QASPER 人工证据
# ==========================================

def collect_evidence(obj):

    evidence = []

    if isinstance(obj, dict):

        for key, value in obj.items():

            if key == "evidence":

                if isinstance(value, str):
                    evidence.append(value)

                elif isinstance(value, list):

                    for item in value:

                        if isinstance(item, str):
                            evidence.append(item)

                        else:
                            evidence.extend(
                                collect_evidence(item)
                            )

            else:

                evidence.extend(
                    collect_evidence(value)
                )

    elif isinstance(obj, list):

        for item in obj:

            evidence.extend(
                collect_evidence(item)
            )

    # 去除空字符串和重复证据
    return list(dict.fromkeys(
        text.strip()
        for text in evidence
        if isinstance(text, str) and text.strip()
    ))


# ==========================================
# 5. 加载 30 道题对应的标准证据
# ==========================================

def load_gold_evidence(questions):

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

    target_keys = {
        (
            item["paper_id"],
            item["question_id"]
        )
        for item in questions
    }

    target_papers = {
        item["paper_id"]
        for item in questions
    }

    gold_map = {}

    for paper in dataset:

        paper_id = str(paper["id"])

        if paper_id not in target_papers:
            continue

        qas = paper["qas"]

        question_ids = qas.get(
            "question_id", []
        )

        answers = qas.get(
            "answers", []
        )

        for index, question_id in enumerate(
            question_ids
        ):

            key = (
                paper_id,
                str(question_id)
            )

            if key not in target_keys:
                continue

            if index >= len(answers):
                continue

            gold_map[key] = collect_evidence(
                answers[index]
            )

    return gold_map


# ==========================================
# 6. 计算最终上下文覆盖情况
# ==========================================

def evaluate_context(
    chunks,
    chunk_lookup,
    gold_evidence
):

    context, sources = build_context(
        chunks,
        max_chars=MAX_CHARS
    )

    included_ids = [
        source["chunk_id"]
        for source in sources
    ]

    matched_ids = []

    for chunk_id in included_ids:

        chunk = chunk_lookup.get(chunk_id)

        if chunk is None:
            continue

        if matches_gold_evidence(
            chunk["text"],
            gold_evidence
        ):

            matched_ids.append(chunk_id)

    return {
        "hit": bool(matched_ids),
        "matched_gold_ids": matched_ids,
        "included_chunk_ids": included_ids,
        "included_chunk_count": len(sources),
        "context_chars": len(context)
    }


# ==========================================
# 7. 汇总指标
# ==========================================

def summarize(records):

    valid = [
        item
        for item in records
        if item["status"] == "evaluated"
    ]

    n = len(valid)

    if n == 0:
        raise ValueError(
            "没有可评估的问题，请检查标准证据匹配。"
        )

    baseline_hits = sum(
        item["baseline"]["hit"]
        for item in valid
    )

    improved_hits = sum(
        item["improved"]["hit"]
        for item in valid
    )

    improved_only = [
        item for item in valid
        if (
            not item["baseline"]["hit"]
            and item["improved"]["hit"]
        )
    ]

    regressed = [
        item for item in valid
        if (
            item["baseline"]["hit"]
            and not item["improved"]["hit"]
        )
    ]

    baseline_chars = mean(
        item["baseline"]["context_chars"]
        for item in valid
    )

    improved_chars = mean(
        item["improved"]["context_chars"]
        for item in valid
    )

    baseline_chunks = mean(
        item["baseline"]["included_chunk_count"]
        for item in valid
    )

    improved_chunks = mean(
        item["improved"]["included_chunk_count"]
        for item in valid
    )

    anchors_lost = sum(
        bool(item["lost_baseline_chunk_ids"])
        for item in valid
    )

    return {
        "evaluated_questions": n,
        "baseline_hits": baseline_hits,
        "improved_hits": improved_hits,
        "baseline_coverage": baseline_hits / n,
        "improved_coverage": improved_hits / n,
        "coverage_change_pp": (
            improved_hits - baseline_hits
        ) / n * 100,
        "improved_only_count": len(improved_only),
        "regressed_count": len(regressed),
        "improved_question_ids": [
            item["question_id"]
            for item in improved_only
        ],
        "regressed_question_ids": [
            item["question_id"]
            for item in regressed
        ],
        "baseline_avg_chars": baseline_chars,
        "improved_avg_chars": improved_chars,
        "baseline_avg_chunks": baseline_chunks,
        "improved_avg_chunks": improved_chunks,
        "questions_with_baseline_chunks_lost": anchors_lost
    }


# ==========================================
# 8. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("QASPER 30 题相邻段落扩展消融实验")
    print("=" * 65)

    print("\n[1] 读取固定测试问题...")

    questions = load_questions()

    print(
        "本次问题数量:",
        len(questions)
    )

    print("\n[2] 读取 QASPER 人工证据...")

    gold_map = load_gold_evidence(
        questions
    )

    print(
        "成功匹配的题目数量:",
        len(gold_map)
    )

    print("\n[3] 初始化检索器...")

    retriever = PaperScopedRetriever()

    records = []

    # ======================================
    # 逐题评估
    # ======================================

    for index, item in enumerate(
        questions,
        start=1
    ):

        paper_id = item["paper_id"]
        question_id = item["question_id"]
        question = item["question"]

        key = (
            paper_id,
            question_id
        )

        record = {
            "paper_id": paper_id,
            "question_id": question_id,
            "question": question
        }

        gold_evidence = gold_map.get(
            key, []
        )

        if not gold_evidence:

            record["status"] = "missing_gold"
            records.append(record)

            print(
                f"[{index:02d}] 缺少标准证据，跳过"
            )

            continue

        # ----------------------------------
        # 获取论文全部 Chunk
        # ----------------------------------

        paper_data = retriever.get_paper_data(
            paper_id
        )

        all_chunks = paper_data["chunks"]

        chunk_lookup = {
            chunk["chunk_id"]: chunk
            for chunk in all_chunks
        }

        # 检查人工证据是否存在于知识库
        gold_chunk_ids = {
            chunk["chunk_id"]
            for chunk in all_chunks
            if matches_gold_evidence(
                chunk["text"],
                gold_evidence
            )
        }

        if not gold_chunk_ids:

            record["status"] = "gold_not_in_corpus"
            records.append(record)

            print(
                f"[{index:02d}] 知识库未匹配到标准证据"
            )

            continue

        # ----------------------------------
        # Baseline
        # ----------------------------------

        retrieved = retriever.search(
            query=question,
            paper_id=paper_id,
            top_k=TOP_K
        )

        baseline = evaluate_context(
            chunks=retrieved,
            chunk_lookup=chunk_lookup,
            gold_evidence=gold_evidence
        )

        # ----------------------------------
        # Improved
        # ----------------------------------

        expanded = expand_neighbors(
            retrieved=retrieved,
            all_chunks=all_chunks,
            window=WINDOW
        )

        improved = evaluate_context(
            chunks=expanded,
            chunk_lookup=chunk_lookup,
            gold_evidence=gold_evidence
        )

        # 检查扩展是否挤掉原本已经纳入的段落
        lost_baseline_ids = sorted(
            set(baseline["included_chunk_ids"])
            - set(improved["included_chunk_ids"])
        )

        record.update({
            "status": "evaluated",
            "gold_chunk_ids": sorted(
                gold_chunk_ids
            ),
            "baseline": baseline,
            "improved": improved,
            "expanded_candidate_count": len(expanded),
            "lost_baseline_chunk_ids": lost_baseline_ids
        })

        records.append(record)

        b = int(baseline["hit"])
        i = int(improved["hit"])

        print(
            f"[{index:02d}/{len(questions)}] "
            f"Baseline={b} | "
            f"Expanded={i} | "
            f"字符数="
            f"{baseline['context_chars']}/"
            f"{improved['context_chars']}"
        )

    # ======================================
    # 9. 输出汇总
    # ======================================

    summary = summarize(records)

    print("\n" + "=" * 65)
    print("30 道题消融实验结果")
    print("=" * 65)

    print(
        "有效评估问题数量:",
        summary["evaluated_questions"]
    )

    print(
        "\nBaseline 证据覆盖:",
        f"{summary['baseline_hits']}/"
        f"{summary['evaluated_questions']}"
    )

    print(
        "Improved 证据覆盖:",
        f"{summary['improved_hits']}/"
        f"{summary['evaluated_questions']}"
    )

    print(
        "\nBaseline 覆盖率:",
        f"{summary['baseline_coverage']:.2%}"
    )

    print(
        "Improved 覆盖率:",
        f"{summary['improved_coverage']:.2%}"
    )

    print(
        "覆盖率变化:",
        f"{summary['coverage_change_pp']:+.2f} 个百分点"
    )

    print(
        "\n改善的题目数量:",
        summary["improved_only_count"]
    )

    print(
        "退化的题目数量:",
        summary["regressed_count"]
    )

    print(
        "\nBaseline 平均字符数:",
        f"{summary['baseline_avg_chars']:.1f}"
    )

    print(
        "Improved 平均字符数:",
        f"{summary['improved_avg_chars']:.1f}"
    )

    print(
        "\nBaseline 平均 Chunk 数:",
        f"{summary['baseline_avg_chunks']:.2f}"
    )

    print(
        "Improved 平均 Chunk 数:",
        f"{summary['improved_avg_chunks']:.2f}"
    )

    print(
        "\n发生原始上下文 Chunk 丢失的题数:",
        summary["questions_with_baseline_chunks_lost"]
    )

    # ======================================
    # 10. 保存实验结果
    # ======================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "experiment": "Neighbor Expansion Ablation",
        "dataset": "QASPER",
        "split": "train_debug",
        "retriever": "PaperScopedHybrid",
        "top_k": TOP_K,
        "window": WINDOW,
        "max_chars": MAX_CHARS,
        "requested_questions": len(questions),
        "summary": summary,
        "results": records
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n结果保存位置:")
    print(OUTPUT_PATH)

    print("\n注意：")
    print("本实验评估的是证据覆盖，不是答案准确率。")
    print("人工标准证据仅用于评估，没有用于选择扩展段落。")
    print("这是训练集调试实验，不能作为正式测试成绩。")

    print("\n消融实验完成！")


if __name__ == "__main__":
    main()
