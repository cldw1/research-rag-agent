import json
import re
import sys
import time
from pathlib import Path


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.paper_scoped_retriever import PaperScopedRetriever

from src.generation.context_builder import (
    build_context,
    build_qa_prompt,
)

from src.generation.answer_generator import generate_answer


# ==========================================
# 2. 配置
# ==========================================

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "bm25_train_debug_30.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "rag_answers_debug_5.json"
)

MAX_QUESTIONS = 5
TOP_K = 5


# ==========================================
# 3. 读取真实问题
# ==========================================

def load_questions():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"找不到评估问题文件：{INPUT_PATH}"
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("detailed_results", [])
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("评估数据格式不正确。")

    if not isinstance(records, list):
        raise ValueError("detailed_results 不是列表。")

    questions = []

    for item in records:

        if not isinstance(item, dict):
            continue

        question = item.get("question")
        paper_id = item.get("paper_id")

        if not question or paper_id is None:
            continue

        questions.append({
            "question_id": item.get("question_id"),
            "paper_id": str(paper_id),
            "question": str(question),
        })

    return questions[:MAX_QUESTIONS]


# ==========================================
# 4. 保存结果
# ==========================================

def save_results(records):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "description": "QASPER train debugging: RAG generated answers",
        "top_k": TOP_K,
        "results": records,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


# ==========================================
# 5. 批量评估
# ==========================================

def main():

    print("=" * 65)
    print("QASPER 真实问题 RAG 批量测试")
    print("=" * 65)

    print("\n[1] 加载测试问题...")

    questions = load_questions()

    print(f"本次测试问题数量：{len(questions)}")

    if not questions:
        print("没有找到可用的问题。")
        return

    print("\n[2] 初始化论文检索器...")

    # 整个批量测试只初始化一次
    retriever = PaperScopedRetriever()

    print("\n检索器加载完成！")

    all_results = []

    start_time = time.perf_counter()

    # --------------------------------------
    # 逐题测试
    # --------------------------------------

    for index, item in enumerate(questions, start=1):

        question = item["question"]
        paper_id = item["paper_id"]

        print("\n" + "=" * 65)
        print(f"测试进度：{index}/{len(questions)}")
        print("=" * 65)

        print(f"论文 ID：{paper_id}")
        print(f"问题：{question}")

        record = {
            "question_id": item["question_id"],
            "paper_id": paper_id,
            "question": question,
        }

        try:

            # 1. 检索证据
            results = retriever.search(
                query=question,
                paper_id=paper_id,
                top_k=TOP_K,
            )

            # 2. 构造上下文
            context, sources = build_context(
                results,
                max_chars=8000,
            )

            record["retrieved_count"] = len(results)
            record["sources"] = sources

            if not sources:

                record["answer"] = ""
                record["status"] = "no_evidence"

                print("没有可用证据，跳过模型调用。")

            else:

                # 3. 构造 Prompt
                prompt = build_qa_prompt(
                    question=question,
                    context=context,
                )

                # 4. 生成回答
                answer = generate_answer(prompt)

                # 5. 提取引用编号
                cited_numbers = sorted({
                    int(number)
                    for number in re.findall(
                        r"\[(\d+)\]",
                        answer,
                    )
                })

                invalid_citations = [
                    number
                    for number in cited_numbers
                    if not (1 <= number <= len(sources))
                ]

                record["answer"] = answer
                record["cited_numbers"] = cited_numbers
                record["invalid_citations"] = invalid_citations
                record["status"] = "success"

                print("\n模型回答：")
                print(answer)

                print(f"\n引用编号：{cited_numbers}")

        except Exception as e:

            record["status"] = "error"
            record["error_type"] = type(e).__name__

            print("\n本题运行失败：")
            print(f"{type(e).__name__}: {e}")

        # 每完成一题就保存一次，避免中途报错丢失全部结果
        all_results.append(record)

        save_results(all_results)

        print("\n当前测试结果已保存。")

    # ======================================
    # 6. 汇总
    # ======================================

    elapsed = time.perf_counter() - start_time

    successful = [
        item
        for item in all_results
        if item["status"] == "success"
    ]

    with_citations = [
        item
        for item in successful
        if item.get("cited_numbers")
    ]

    invalid_count = sum(
        len(item.get("invalid_citations", []))
        for item in successful
    )

    print("\n" + "=" * 65)
    print("RAG 批量测试结果")
    print("=" * 65)

    print(f"测试问题数量：{len(all_results)}")
    print(f"成功生成回答：{len(successful)}")
    print(f"包含引用编号的回答：{len(with_citations)}")
    print(f"不存在的引用编号数量：{invalid_count}")

    print(f"总耗时：{elapsed:.2f} 秒")

    print(f"\n结果保存位置：{OUTPUT_PATH}")

    print("\n说明：")
    print("本次只生成并保存回答，尚未评估答案是否正确。")
    print("无引用不一定是错误，可能是合理的证据不足拒答。")

    print("\n" + "=" * 65)
    print("批量测试完成")
    print("=" * 65)


if __name__ == "__main__":
    main()