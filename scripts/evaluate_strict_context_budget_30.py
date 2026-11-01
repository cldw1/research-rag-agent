
import json
import sys
from pathlib import Path
from statistics import mean


# ==========================================
# 1. 项目根目录
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.validation_paper_scoped_retriever import (
    ValidationPaperScopedRetriever
)

from scripts.evaluate_anchor_first_30 import (
    expand_anchors_first
)

from scripts.diagnose_rag_failures import (
    matches_gold_evidence
)

from src.generation.context_builder import (
    build_context
)

from src.generation.strict_context_builder import (
    build_strict_context
)


# ==========================================
# 2. 实验配置
# ==========================================

MAX_CHARS = 8000

QUESTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "qasper_validation_30_questions.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "strict_context_budget_30.json"
)


# ==========================================
# 3. 检查人工证据是否进入上下文
# ==========================================

def contains_gold(
    sources,
    chunk_lookup,
    gold_evidence
):

    for source in sources:

        chunk_id = source["chunk_id"]

        chunk = chunk_lookup[chunk_id]

        if matches_gold_evidence(
            chunk["text"],
            gold_evidence
        ):
            return True

    return False


# ==========================================
# 4. 运行单个方案
# ==========================================

def evaluate_method(
    candidates,
    chunk_lookup,
    gold_evidence
):

    old_context, old_sources = build_context(
        candidates,
        max_chars=MAX_CHARS
    )

    new_context, new_sources = build_strict_context(
        candidates,
        max_chars=MAX_CHARS
    )

    # 核心要求：严格控制字符数量
    assert len(new_context) <= MAX_CHARS

    # 检查引用编号连续
    for index, source in enumerate(
        new_sources,
        start=1
    ):
        assert int(source["citation"]) == index

        assert source["chunk_id"] in chunk_lookup

    old_hit = contains_gold(
        old_sources,
        chunk_lookup,
        gold_evidence
    )

    new_hit = contains_gold(
        new_sources,
        chunk_lookup,
        gold_evidence
    )

    return {
        "old_chars": len(old_context),
        "new_chars": len(new_context),
        "old_hit": old_hit,
        "new_hit": new_hit,
        "old_chunk_count": len(old_sources),
        "new_chunk_count": len(new_sources),
        "new_chunk_ids": [
            source["chunk_id"]
            for source in new_sources
        ]
    }


# ==========================================
# 5. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("Validation 严格上下文预算检查")
    print("=" * 65)

    with open(
        QUESTION_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        questions = json.load(f)["questions"]

    print("\n验证问题数量:", len(questions))

    print("\n初始化 Validation 检索器...")

    retriever = ValidationPaperScopedRetriever()

    records = []

    # ======================================
    # 6. 逐题检查
    # ======================================

    for index, item in enumerate(
        questions,
        start=1
    ):

        paper_id = item["paper_id"]
        question = item["question"]
        gold_evidence = item["gold_evidence"]

        paper_data = retriever.get_paper_data(
            paper_id
        )

        chunks = paper_data["chunks"]

        chunk_lookup = {
            chunk["chunk_id"]: chunk
            for chunk in chunks
        }

        top10 = retriever.search(
            query=question,
            paper_id=paper_id,
            top_k=10,
            candidate_k=50
        )

        top5 = top10[:5]

        anchor_first = expand_anchors_first(
            retrieved=top5,
            all_chunks=chunks
        )

        baseline = evaluate_method(
            top5,
            chunk_lookup,
            gold_evidence
        )

        top10_result = evaluate_method(
            top10,
            chunk_lookup,
            gold_evidence
        )

        anchor_result = evaluate_method(
            anchor_first,
            chunk_lookup,
            gold_evidence
        )

        # Anchor-First 应保留原始 Top-5
        original_anchor_ids = set(
            baseline["new_chunk_ids"]
        )

        expanded_ids = set(
            anchor_result["new_chunk_ids"]
        )

        lost_anchors = sorted(
            original_anchor_ids - expanded_ids
        )

        record = {
            "paper_id": paper_id,
            "question_id": item["question_id"],
            "top5": baseline,
            "top10": top10_result,
            "anchor_first": anchor_result,
            "lost_anchor_ids": lost_anchors
        }

        records.append(record)

        print(
            f"[{index:02d}/{len(questions)}] "
            f"Top10={top10_result['new_chars']} | "
            f"AnchorFirst={anchor_result['new_chars']} | "
            f"丢失Anchor={len(lost_anchors)}"
        )

    # ======================================
    # 7. 汇总结果
    # ======================================

    methods = [
        "top5",
        "top10",
        "anchor_first"
    ]

    summary = {}

    for method in methods:

        values = [
            record[method]
            for record in records
        ]

        old_over = sum(
            value["old_chars"] > MAX_CHARS
            for value in values
        )

        new_over = sum(
            value["new_chars"] > MAX_CHARS
            for value in values
        )

        old_hits = sum(
            value["old_hit"]
            for value in values
        )

        new_hits = sum(
            value["new_hit"]
            for value in values
        )

        summary[method] = {
            "old_over_budget": old_over,
            "new_over_budget": new_over,
            "old_hits": old_hits,
            "new_hits": new_hits,
            "old_coverage": old_hits / len(values),
            "new_coverage": new_hits / len(values),
            "avg_new_chars": mean(
                value["new_chars"]
                for value in values
            )
        }

    lost_anchor_count = sum(
        bool(record["lost_anchor_ids"])
        for record in records
    )

    print("\n" + "=" * 65)
    print("严格上下文预算检查结果")
    print("=" * 65)

    for method, values in summary.items():

        print(f"\n方法: {method}")

        print(
            "原始超预算题数:",
            values["old_over_budget"]
        )

        print(
            "修复后超预算题数:",
            values["new_over_budget"]
        )

        print(
            "原始证据命中:",
            f"{values['old_hits']}/30"
        )

        print(
            "修复后证据命中:",
            f"{values['new_hits']}/30"
        )

        print(
            "修复后覆盖率:",
            f"{values['new_coverage']:.2%}"
        )

        print(
            "平均上下文字符数:",
            f"{values['avg_new_chars']:.1f}"
        )

    print("\nAnchor-First 丢失原始 Anchor 的题数:")
    print(lost_anchor_count)

    # ======================================
    # 8. 保存报告
    # ======================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "experiment": "Strict Context Budget",
        "split": "validation",
        "sample_size": len(records),
        "max_chars": MAX_CHARS,
        "summary": summary,
        "lost_anchor_count": lost_anchor_count,
        "results": records
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    assert all(
        values["new_over_budget"] == 0
        for values in summary.values()
    )

    print("\n结果已保存:")
    print(OUTPUT_PATH)

    print("\n全部方法均满足严格字符预算！")
    print("第35步完成！")


if __name__ == "__main__":
    main()
