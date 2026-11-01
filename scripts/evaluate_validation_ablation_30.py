
import json
import sys
from pathlib import Path
from statistics import mean


# ==========================================
# 1. 项目目录
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

from scripts.evaluate_context_expansion_30 import (
    evaluate_context
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
    / "qasper_validation_30_questions.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "validation_ablation_30.json"
)

TOP_K = 5
COMPARE_TOP_K = 10
MAX_CHARS = 8000


# ==========================================
# 3. 读取固定验证集
# ==========================================

def load_questions():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    questions = data["questions"]

    if len(questions) != 30:
        raise ValueError(
            f"预期30道问题，实际{len(questions)}道"
        )

    if data.get("split") != "validation":
        raise ValueError(
            "数据集不是 Validation"
        )

    return questions


# ==========================================
# 4. 汇总单个方法的成绩
# ==========================================

def summarize_method(records, method):

    hits = sum(
        item[method]["hit"]
        for item in records
    )

    return {
        "hits": hits,
        "total": len(records),
        "coverage": hits / len(records),
        "avg_chars": mean(
            item[method]["context_chars"]
            for item in records
        ),
        "avg_chunks": mean(
            item[method]["included_chunk_count"]
            for item in records
        )
    }


# ==========================================
# 5. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("QASPER Validation - 三组消融实验")
    print("=" * 65)

    print("\n[1] 读取固定验证集...")

    questions = load_questions()

    print("Validation 问题数量:", len(questions))

    print("\n[2] 初始化 Validation 检索器...")

    retriever = ValidationPaperScopedRetriever()

    records = []

    # ======================================
    # 6. 逐题评估
    # ======================================

    print("\n[3] 开始逐题检索...")

    for index, item in enumerate(
        questions,
        start=1
    ):

        paper_id = str(item["paper_id"])
        question_id = str(item["question_id"])
        question = item["question"]

        gold_evidence = item.get(
            "gold_evidence", []
        )

        paper_data = retriever.get_paper_data(
            paper_id
        )

        chunks = paper_data["chunks"]

        chunk_lookup = {
            chunk["chunk_id"]: chunk
            for chunk in chunks
        }

        # 检查人工证据是否存在于索引语料
        gold_chunk_ids = [
            chunk["chunk_id"]
            for chunk in chunks
            if matches_gold_evidence(
                chunk["text"],
                gold_evidence
            )
        ]

        if not gold_chunk_ids:

            records.append({
                "paper_id": paper_id,
                "question_id": question_id,
                "question": question,
                "status": "gold_not_matched"
            })

            print(
                f"[{index:02d}/30] "
                "人工证据未匹配到索引，单独记录"
            )

            continue

        # ----------------------------------
        # 同一检索结果比较三个方案
        # ----------------------------------

        top10 = retriever.search(
            query=question,
            paper_id=paper_id,
            top_k=COMPARE_TOP_K,
            candidate_k=50
        )

        top5 = top10[:TOP_K]

        anchor_first = expand_anchors_first(
            retrieved=top5,
            all_chunks=chunks
        )

        # ----------------------------------
        # 使用相同的评估方式
        # ----------------------------------

        baseline = evaluate_context(
            top5,
            chunk_lookup,
            gold_evidence
        )

        top10_result = evaluate_context(
            top10,
            chunk_lookup,
            gold_evidence
        )

        improved = evaluate_context(
            anchor_first,
            chunk_lookup,
            gold_evidence
        )

        # 检查有没有挤掉原始 Top-5
        lost_ids = sorted(
            set(baseline["included_chunk_ids"])
            - set(improved["included_chunk_ids"])
        )

        records.append({
            "paper_id": paper_id,
            "question_id": question_id,
            "question": question,
            "status": "evaluated",
            "gold_chunk_ids": gold_chunk_ids,
            "top5": baseline,
            "top10": top10_result,
            "anchor_first": improved,
            "lost_anchor_ids": lost_ids
        })

        print(
            f"[{index:02d}/30] "
            f"Top5={int(baseline['hit'])} | "
            f"Top10={int(top10_result['hit'])} | "
            f"AnchorFirst={int(improved['hit'])}"
        )

    # ======================================
    # 7. 筛选有效样本
    # ======================================

    valid = [
        item
        for item in records
        if item["status"] == "evaluated"
    ]

    if not valid:
        raise RuntimeError(
            "没有有效样本，需检查人工证据匹配"
        )

    # ======================================
    # 8. 汇总三种方法
    # ======================================

    summary = {
        "top5": summarize_method(
            valid, "top5"
        ),
        "top10": summarize_method(
            valid, "top10"
        ),
        "anchor_first": summarize_method(
            valid, "anchor_first"
        )
    }

    improved_cases = [
        item for item in valid
        if (
            not item["top5"]["hit"]
            and item["anchor_first"]["hit"]
        )
    ]

    regressed_cases = [
        item for item in valid
        if (
            item["top5"]["hit"]
            and not item["anchor_first"]["hit"]
        )
    ]

    lost_anchor_cases = [
        item for item in valid
        if item["lost_anchor_ids"]
    ]

    # ======================================
    # 9. 打印真实结果
    # ======================================

    print("\n" + "=" * 65)
    print("Validation 独立验证结果")
    print("=" * 65)

    print("验证集总问题数量:", len(questions))
    print("有效评估问题数量:", len(valid))
    print(
        "未匹配人工证据的数量:",
        len(questions) - len(valid)
    )

    for method, values in summary.items():

        print("\n" + "-" * 55)
        print("方法:", method)

        print(
            "证据命中:",
            f"{values['hits']}/{values['total']}"
        )

        print(
            "证据覆盖率:",
            f"{values['coverage']:.2%}"
        )

        print(
            "平均字符数:",
            f"{values['avg_chars']:.1f}"
        )

        print(
            "平均 Chunk 数:",
            f"{values['avg_chunks']:.2f}"
        )

    print("\n" + "=" * 65)
    print("Anchor-First 与 Top-5 对比")
    print("=" * 65)

    print(
        "改善题数:",
        len(improved_cases)
    )

    print(
        "退化题数:",
        len(regressed_cases)
    )

    print(
        "原始 Chunk 丢失题数:",
        len(lost_anchor_cases)
    )

    change_pp = (
        summary["anchor_first"]["coverage"]
        - summary["top5"]["coverage"]
    ) * 100

    print(
        "证据覆盖率变化:",
        f"{change_pp:+.2f} 个百分点"
    )

    # ======================================
    # 10. 保存结果
    # ======================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "experiment": "Validation Ablation",
        "dataset": "QASPER",
        "split": "validation",
        "sample_size": len(questions),
        "max_chars": MAX_CHARS,
        "top_k": TOP_K,
        "compare_top_k": COMPARE_TOP_K,
        "neighbor_window": 1,
        "summary": summary,
        "improved_count": len(improved_cases),
        "regressed_count": len(regressed_cases),
        "lost_anchor_count": len(lost_anchor_cases),
        "unmatched_gold_count": (
            len(questions) - len(valid)
        ),
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

    print("\n实验结果已保存：")
    print(OUTPUT_PATH)

    print("\n注意：")
    print("本实验不调用 DeepSeek。")
    print("人工证据只用于离线评分。")
    print("未匹配到索引的证据单独统计，不计入覆盖率分母。")
    print("Validation 结果不等于最终独立测试集成绩。")

    print("\n第32步实验完成！")


if __name__ == "__main__":
    main()
