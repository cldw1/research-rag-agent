
import json
import re
import sys
from pathlib import Path


# ==========================================
# 1. 项目路径
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
    build_context,
    build_qa_prompt
)

from src.generation.answer_generator import (
    generate_answer
)


# ==========================================
# 2. 实验配置
# ==========================================

QUESTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "qasper_validation_30_questions.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "validation_generation_compare_5.json"
)

SAMPLE_SIZE = 5
TOP_K = 5
COMPARE_TOP_K = 10
MAX_CHARS = 8000

METHODS = [
    "top10",
    "anchor_first"
]


# ==========================================
# 3. 读取固定验证问题
# ==========================================

def load_questions():

    with open(
        QUESTION_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if data.get("split") != "validation":
        raise ValueError("当前文件不是 Validation 数据")

    questions = data["questions"][:SAMPLE_SIZE]

    if len(questions) != SAMPLE_SIZE:
        raise ValueError("验证问题数量不足")

    return questions


# ==========================================
# 4. 分析模型引用
# ==========================================

def analyze_citations(
    answer,
    sources,
    gold_evidence,
    chunk_lookup
):

    cited_numbers = sorted({
        int(x)
        for x in re.findall(
            r"\[(\d+)\]",
            answer
        )
    })

    valid_numbers = [
        n
        for n in cited_numbers
        if 1 <= n <= len(sources)
    ]

    invalid_numbers = [
        n
        for n in cited_numbers
        if n not in valid_numbers
    ]

    gold_source_ids = []

    for source in sources:

        chunk_id = source["chunk_id"]

        chunk = chunk_lookup.get(chunk_id)

        if chunk is None:
            continue

        if matches_gold_evidence(
            chunk["text"],
            gold_evidence
        ):
            gold_source_ids.append(chunk_id)

    cited_gold_ids = []

    for number in valid_numbers:

        chunk_id = sources[number - 1]["chunk_id"]

        if chunk_id in gold_source_ids:
            cited_gold_ids.append(chunk_id)

    return {
        "cited_numbers": cited_numbers,
        "invalid_citations": invalid_numbers,
        "gold_in_context": bool(gold_source_ids),
        "gold_source_ids": gold_source_ids,
        "cited_gold_ids": cited_gold_ids
    }


# ==========================================
# 5. 保存实验进度
# ==========================================

def save_report(report):

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
            report,
            f,
            ensure_ascii=False,
            indent=2
        )


# ==========================================
# 6. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("QASPER Validation - RAG 回答质量对比")
    print("=" * 65)

    print("\n[1] 读取固定验证问题...")

    questions = load_questions()

    # 如果程序中断，可从上一次保存的位置继续
    if OUTPUT_PATH.exists():

        with open(
            OUTPUT_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            report = json.load(f)

        saved_ids = {
            item["question_id"]
            for item in report["results"]
        }

        expected_ids = {
            str(item["question_id"])
            for item in questions
        }

        if not saved_ids.issubset(expected_ids):
            raise ValueError(
                "已有结果与本次问题样本不匹配，"
                "请检查输出文件。"
            )

        print("发现已有结果，将跳过完成的调用。")

    else:

        report = {
            "experiment": "Validation Generation Comparison",
            "dataset": "QASPER",
            "split": "validation",
            "sample_size": SAMPLE_SIZE,
            "sample_rule": "first 5 of fixed random sample",
            "top_k": TOP_K,
            "compare_top_k": COMPARE_TOP_K,
            "max_chars": MAX_CHARS,
            "methods": METHODS,
            "results": []
        }

    print("\n[2] 初始化验证集检索器...")

    retriever = ValidationPaperScopedRetriever()

    # ======================================
    # 7. 对每道问题分别生成答案
    # ======================================

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

        print("\n" + "=" * 65)
        print(f"问题 {index}/{SAMPLE_SIZE}")
        print("=" * 65)

        print("论文:", paper_id)
        print("问题:", question)

        record = next(
            (
                r
                for r in report["results"]
                if r["question_id"] == question_id
            ),
            None
        )

        if record is None:

            record = {
                "paper_id": paper_id,
                "question_id": question_id,
                "question": question,
                "gold_evidence": gold_evidence,
                "variants": {}
            }

            report["results"].append(record)

        # ----------------------------------
        # 获取同一批初始检索结果
        # ----------------------------------

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
            top_k=COMPARE_TOP_K,
            candidate_k=50
        )

        top5 = top10[:TOP_K]

        anchor_first = expand_anchors_first(
            retrieved=top5,
            all_chunks=chunks
        )

        candidates = {
            "top10": top10,
            "anchor_first": anchor_first
        }

        # ----------------------------------
        # 分别调用 DeepSeek
        # ----------------------------------

        for method in METHODS:

            if method in record["variants"]:

                print(
                    f"\n{method} 已完成，跳过。"
                )

                continue

            print(
                f"\n[3] 当前方法：{method}"
            )

            context, sources = build_context(
                candidates[method],
                max_chars=MAX_CHARS
            )

            print("上下文 Chunk 数:", len(sources))
            print("上下文字符数:", len(context))

            # Prompt 中只有问题和检索上下文
            # 不包含人工标准证据
            prompt = build_qa_prompt(
                question=question,
                context=context
            )

            print("正在调用 DeepSeek...")

            answer = generate_answer(prompt)

            citation_result = analyze_citations(
                answer=answer,
                sources=sources,
                gold_evidence=gold_evidence,
                chunk_lookup=chunk_lookup
            )

            record["variants"][method] = {
                "answer": answer,
                "context_chars": len(context),
                "chunk_count": len(sources),
                "sources": sources,
                **citation_result
            }

            # 每次 API 调用成功后立即保存
            save_report(report)

            print("\n生成答案：")
            print(answer)

            print("\n引用编号：")
            print(
                citation_result["cited_numbers"]
            )

            print("无效引用：")
            print(
                citation_result["invalid_citations"]
            )

            print("标准证据在上下文中：")
            print(
                citation_result["gold_in_context"]
            )

        # ----------------------------------
        # 展示人工证据供人工核对
        # ----------------------------------

        print("\n[4] 人工标准证据（仅评估使用）")

        for evidence_index, evidence in enumerate(
            gold_evidence,
            start=1
        ):

            print(
                f"人工证据 {evidence_index}: "
                f"{evidence[:600]}"
            )

    # ======================================
    # 8. 汇总实验结果
    # ======================================

    print("\n" + "=" * 65)
    print("Validation 回答生成实验汇总")
    print("=" * 65)

    for method in METHODS:

        completed = [
            record["variants"][method]
            for record in report["results"]
            if method in record["variants"]
        ]

        if not completed:
            continue

        evidence_hits = sum(
            result["gold_in_context"]
            for result in completed
        )

        has_citation = sum(
            bool(result["cited_numbers"])
            for result in completed
        )

        invalid_count = sum(
            bool(result["invalid_citations"])
            for result in completed
        )

        print(f"\n方法：{method}")

        print(
            "已生成答案数量:",
            len(completed)
        )

        print(
            "包含标准证据的题数:",
            evidence_hits
        )

        print(
            "包含引用的回答数:",
            has_citation
        )

        print(
            "存在无效引用的回答数:",
            invalid_count
        )

    print("\n实验结果保存位置：")
    print(OUTPUT_PATH)

    print("\n注意：")
    print("以上统计不是答案准确率。")
    print("最终答案是否正确，需要对照人工证据检查。")
    print("标准证据没有发送给 DeepSeek。")

    print("\n第33步完成！")


if __name__ == "__main__":
    main()
