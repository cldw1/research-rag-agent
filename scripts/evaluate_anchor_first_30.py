
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.paper_scoped_retriever import (
    PaperScopedRetriever
)

from scripts.evaluate_context_expansion import (
    expand_neighbors
)

from scripts.evaluate_context_expansion_30 import (
    load_questions,
    load_gold_evidence,
    evaluate_context
)

from scripts.diagnose_rag_failures import (
    matches_gold_evidence
)


# ==========================================
# 1. 配置
# ==========================================

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "anchor_first_ablation_30.json"
)

TOP_K = 5
COMPARE_TOP_K = 10
WINDOW = 1


# ==========================================
# 2. 优先保留原始检索结果
# ==========================================

def expand_anchors_first(
    retrieved,
    all_chunks
):

    expanded = expand_neighbors(
        retrieved=retrieved,
        all_chunks=all_chunks,
        window=WINDOW
    )

    anchor_ids = {
        item["chunk_id"]
        for item in retrieved
    }

    # 原始 Top-5 放在最前面
    anchors = list(retrieved)

    # 额外扩展的段落放在后面
    neighbors = [
        item
        for item in expanded
        if item["chunk_id"] not in anchor_ids
    ]

    return anchors + neighbors


# ==========================================
# 3. 汇总单个方法
# ==========================================

def summarize_method(records, key):

    hits = sum(
        item[key]["hit"]
        for item in records
    )

    return {
        "hits": hits,
        "coverage": hits / len(records),
        "avg_chars": mean(
            item[key]["context_chars"]
            for item in records
        ),
        "avg_chunks": mean(
            item[key]["included_chunk_count"]
            for item in records
        )
    }


# ==========================================
# 4. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("RAG 检索优化 - 三组消融实验")
    print("=" * 65)

    print("\n[1] 加载固定的30道问题")

    questions = load_questions()
    gold_map = load_gold_evidence(questions)

    print("问题数量:", len(questions))

    print("\n[2] 初始化检索器")

    retriever = PaperScopedRetriever()

    records = []

    for index, item in enumerate(
        questions,
        start=1
    ):

        paper_id = item["paper_id"]
        question_id = item["question_id"]
        question = item["question"]

        gold = gold_map.get(
            (paper_id, question_id),
            []
        )

        if not gold:
            print(f"[{index:02d}] 缺少标准证据，跳过")
            continue

        paper_data = retriever.get_paper_data(
            paper_id
        )

        all_chunks = paper_data["chunks"]

        chunk_lookup = {
            chunk["chunk_id"]: chunk
            for chunk in all_chunks
        }

        # 确保标准证据存在于语料中
        available_gold = [
            chunk
            for chunk in all_chunks
            if matches_gold_evidence(
                chunk["text"], gold
            )
        ]

        if not available_gold:
            print(f"[{index:02d}] 标准证据不在语料中")
            continue

        # 一次检索 Top-10
        top10 = retriever.search(
            query=question,
            paper_id=paper_id,
            top_k=COMPARE_TOP_K
        )

        top5 = top10[:TOP_K]

        # 保护原始 Top-5 后再扩展
        anchor_first = expand_anchors_first(
            retrieved=top5,
            all_chunks=all_chunks
        )

        # 三组分别评估
        result5 = evaluate_context(
            top5, chunk_lookup, gold
        )

        result10 = evaluate_context(
            top10, chunk_lookup, gold
        )

        result_anchor = evaluate_context(
            anchor_first, chunk_lookup, gold
        )

        # 检查原始证据是否被挤掉
        lost_ids = sorted(
            set(result5["included_chunk_ids"])
            - set(result_anchor["included_chunk_ids"])
        )

        records.append({
            "paper_id": paper_id,
            "question_id": question_id,
            "question": question,
            "top5": result5,
            "top10": result10,
            "anchor_first": result_anchor,
            "lost_anchor_ids": lost_ids
        })

        print(
            f"[{index:02d}/{len(questions)}] "
            f"Top5={int(result5['hit'])} | "
            f"Top10={int(result10['hit'])} | "
            f"AnchorFirst={int(result_anchor['hit'])}"
        )

    if not records:
        raise RuntimeError("没有有效评估数据")

    # ======================================
    # 5. 汇总实验结果
    # ======================================

    summary = {
        "top5": summarize_method(records, "top5"),
        "top10": summarize_method(records, "top10"),
        "anchor_first": summarize_method(
            records, "anchor_first"
        )
    }

    print("\n" + "=" * 65)
    print("三组消融实验结果")
    print("=" * 65)

    for name, values in summary.items():

        print(f"\n方法：{name}")

        print(
            f"证据覆盖：{values['hits']}/{len(records)}"
        )

        print(
            f"证据覆盖率：{values['coverage']:.2%}"
        )

        print(
            f"平均上下文字符数："
            f"{values['avg_chars']:.1f}"
        )

        print(
            f"平均 Chunk 数："
            f"{values['avg_chunks']:.2f}"
        )

    lost_count = sum(
        bool(item["lost_anchor_ids"])
        for item in records
    )

    print(
        "\n原始 Top-5 Chunk 被挤掉的题数：",
        lost_count
    )

    improved_count = sum(
        not item["top5"]["hit"]
        and item["anchor_first"]["hit"]
        for item in records
    )

    regressed_count = sum(
        item["top5"]["hit"]
        and not item["anchor_first"]["hit"]
        for item in records
    )

    print("相对于 Top-5 改善的题数：", improved_count)
    print("相对于 Top-5 退化的题数：", regressed_count)

    # ======================================
    # 6. 保存文件
    # ======================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "experiment": "Anchor First Ablation",
                "split": "train_debug",
                "max_chars": 8000,
                "evaluated_questions": len(records),
                "summary": summary,
                "lost_anchor_count": lost_count,
                "improved_count": improved_count,
                "regressed_count": regressed_count,
                "results": records
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n结果文件：", OUTPUT_PATH)

    print("\n说明：")
    print("这是证据覆盖实验，不是生成答案准确率实验。")
    print("标准证据仅用于评估，不参与检索排序。")

    print("\n实验完成！")


if __name__ == "__main__":
    main()
