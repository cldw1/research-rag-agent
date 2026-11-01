
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.paper_scoped_retriever import (
    PaperScopedRetriever
)

from src.generation.context_builder import (
    build_context,
    build_qa_prompt,
)

from src.generation.answer_generator import generate_answer

from scripts.evaluate_context_expansion import (
    expand_neighbors
)


# ==========================================
# 1. 实验配置
# ==========================================

PAPER_ID = "1905.10810"
QUESTION = "How is PIEWi annotated?"

GOLD_CHUNK_ID = "1905.10810_s005_p000_c000"

BASELINE_PATH = (
    PROJECT_ROOT
    / "results"
    / "rag_answers_debug_5.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "rag_expansion_answer_plewi.json"
)


# ==========================================
# 2. 读取之前的 Baseline 回答
# ==========================================

def load_baseline_answer():

    with open(
        BASELINE_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    for item in data["results"]:

        if str(item["paper_id"]) == PAPER_ID:
            return item.get("answer", "")

    raise ValueError("找不到之前保存的 Baseline 回答")


# ==========================================
# 3. 执行改进后的 RAG
# ==========================================

def main():

    print("=" * 65)
    print("RAG 相邻段落扩展 - 回答质量验证")
    print("=" * 65)

    baseline_answer = load_baseline_answer()

    print("\n[1] 读取之前的 Baseline 回答完成")

    print("\n[2] 初始化检索器...")

    retriever = PaperScopedRetriever()

    # 原始 Hybrid Top-5
    retrieved = retriever.search(
        query=QUESTION,
        paper_id=PAPER_ID,
        top_k=5
    )

    paper_data = retriever.get_paper_data(PAPER_ID)

    # 扩展相邻段落
    expanded = expand_neighbors(
        retrieved=retrieved,
        all_chunks=paper_data["chunks"],
        window=1
    )

    # 构建上下文
    context, sources = build_context(
        expanded,
        max_chars=8000
    )

    included_ids = {
        source["chunk_id"]
        for source in sources
    }

    gold_in_context = GOLD_CHUNK_ID in included_ids

    print("\n[3] 上下文检查")
    print("实际纳入 Chunk 数量:", len(sources))
    print("上下文字符数:", len(context))
    print("包含标准证据:", gold_in_context)

    if not gold_in_context:
        print("正确证据未进入上下文，停止生成。")
        return

    # 构造 Prompt，不提供标准答案
    prompt = build_qa_prompt(
        question=QUESTION,
        context=context
    )

    # 只对改进方案调用一次 DeepSeek
    print("\n[4] 调用 DeepSeek...")

    improved_answer = generate_answer(prompt)

    # ======================================
    # 4. 对比回答
    # ======================================

    print("\n" + "=" * 65)
    print("原始 Hybrid 回答")
    print("=" * 65)

    print(baseline_answer)

    print("\n" + "=" * 65)
    print("相邻段落扩展后的回答")
    print("=" * 65)

    print(improved_answer)

    print("\n" + "=" * 65)
    print("人工标准答案（仅用于评估）")
    print("=" * 65)

    print("[error, correction] pairs")

    # ======================================
    # 5. 检查引用编号
    # ======================================

    cited_numbers = sorted({
        int(x)
        for x in re.findall(
            r"\[(\d+)\]",
            improved_answer
        )
    })

    invalid_citations = [
        number
        for number in cited_numbers
        if not (1 <= number <= len(sources))
    ]

    print("\n[5] 引用检查")
    print("引用编号:", cited_numbers)
    print("无效引用:", invalid_citations)

    print("\n实际引用的 Chunk:")

    for number in cited_numbers:

        if 1 <= number <= len(sources):

            source = sources[number - 1]

            print(
                f"[{number}] {source['chunk_id']}"
            )

    # ======================================
    # 6. 保存结果
    # ======================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "experiment": "Neighbor Expansion Answer Test",
        "paper_id": PAPER_ID,
        "question": QUESTION,
        "baseline_answer": baseline_answer,
        "expanded_answer": improved_answer,
        "gold_answer": "[error, correction] pairs",
        "gold_chunk_in_context": gold_in_context,
        "context_chars": len(context),
        "sources": sources,
        "cited_numbers": cited_numbers,
        "invalid_citations": invalid_citations
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

    print("\n结果已保存：", OUTPUT_PATH)

    print("\n注意：")
    print("请人工检查答案是否真正符合标准答案。")
    print("本次实验没有把标准答案提供给 DeepSeek。")

    print("\n实验完成！")


if __name__ == "__main__":
    main()
