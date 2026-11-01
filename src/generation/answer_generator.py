import os
import re
import sys
from pathlib import Path

from openai import OpenAI


# ==========================================
# 1. 设置项目根目录
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# 2. 导入已有模块
# ==========================================

from src.generation.context_builder import (
    build_context,
    build_qa_prompt,
)


# ==========================================
# 3. 调用 DeepSeek 生成回答
# ==========================================

def generate_answer(prompt: str) -> str:

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY，"
            "请在 PyCharm 运行配置的环境变量中设置。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=1,
    )

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=False,
    )

    answer = response.choices[0].message.content

    return answer.strip() if answer else ""


# ==========================================
# 4. 完整 RAG 问答流程
# ==========================================

def rag_answer(
    question: str,
    paper_id: str,
    top_k: int = 5,
):

    from src.retrieval.paper_scoped_retriever import PaperScopedRetriever

    print("[1] 初始化论文检索器...")

    retriever = PaperScopedRetriever()

    print("\n[2] 检索相关证据...")

    results = retriever.search(
        query=question,
        paper_id=paper_id,
        top_k=top_k,
    )

    print(f"检索结果数量: {len(results)}")

    print("\n[3] 构造引用上下文...")

    context, sources = build_context(
        results,
        max_chars=8000,
    )

    print(f"引用来源数量: {len(sources)}")

    if not sources:
        print("未找到可用证据，无法生成有依据的回答。")
        return

    print("\n[4] 构造 Prompt...")

    prompt = build_qa_prompt(
        question=question,
        context=context,
    )

    print(f"Prompt 字符数: {len(prompt)}")

    print("\n[5] 调用 DeepSeek 生成回答...")

    answer = generate_answer(prompt)

    print("\n" + "=" * 65)
    print("RAG 最终回答")
    print("=" * 65)

    print(f"\n问题: {question}")

    print(f"\n回答:\n{answer}")

    print("\n" + "=" * 65)
    print("引用检查")
    print("=" * 65)

    cited_numbers = sorted({
        int(number)
        for number in re.findall(r"\[(\d+)\]", answer)
    })

    if not cited_numbers:
        print("提醒：回答中没有检测到引用编号。")

    for number in cited_numbers:

        if 1 <= number <= len(sources):
            print(f"[{number}] 引用编号有效")
        else:
            print(f"[{number}] 警告：引用编号不存在")

    print("\n" + "=" * 65)
    print("RAG 问答测试结束")
    print("=" * 65)


# ==========================================
# 5. 测试入口
# ==========================================

def main():

    question = "What is the seed lexicon?"

    paper_id = "1909.00694"

    rag_answer(
        question=question,
        paper_id=paper_id,
        top_k=5,
    )


if __name__ == "__main__":
    main()
