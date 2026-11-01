import re
import sys
import textwrap
from pathlib import Path


# ==========================================
# 1. 设置项目根目录
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# 2. 导入已有模块
# ==========================================

from src.retrieval.paper_scoped_retriever import PaperScopedRetriever

from src.generation.context_builder import (
    build_context,
    build_qa_prompt,
)

from src.generation.answer_generator import generate_answer


# ==========================================
# 3. 展示引用来源
# ==========================================

def show_citation_sources(answer, sources, results):

    print("\n" + "=" * 65)
    print("引用来源与原文")
    print("=" * 65)

    cited_numbers = sorted({
        int(number)
        for number in re.findall(r"\[(\d+)\]", answer)
    })

    # 没有引用编号
    if not cited_numbers:

        print("\n回答中没有检测到引用编号。")
        print("如果模型因为证据不足而拒绝回答，这是正常情况。")

        return

    # 建立 Chunk ID 到检索结果的映射
    chunk_map = {
        str(item["chunk_id"]): item
        for item in results
        if item.get("chunk_id")
    }

    # 逐一展示被引用的证据
    for number in cited_numbers:

        print("\n" + "-" * 65)
        print(f"引用编号：[{number}]")

        # 检查编号是否存在
        if not (1 <= number <= len(sources)):

            print("警告：引用编号不存在！")

            continue

        source = sources[number - 1]

        # 兼容不同的来源数据形式
        if not isinstance(source, dict):

            print(f"来源记录：{source}")
            print("当前来源记录不是字典，无法自动展示详细字段。")

            continue

        chunk_id = source.get("chunk_id")

        # 根据 Chunk ID 找回原始检索结果
        matched_result = chunk_map.get(str(chunk_id), {})

        paper_id = (
            source.get("paper_id")
            or matched_result.get("paper_id")
            or "未知"
        )

        title = (
            source.get("title")
            or matched_result.get("title")
            or "未知"
        )

        section_name = (
            source.get("section_name")
            or matched_result.get("section_name")
            or "未知"
        )

        text = (
            matched_result.get("text")
            or source.get("text")
            or source.get("content")
            or ""
        )

        print(f"论文 ID：{paper_id}")
        print(f"论文标题：{title}")
        print(f"章节：{section_name}")
        print(f"Chunk ID：{chunk_id}")

        print("\n引用原文：")

        if text:

            # 自动换行，避免 PyCharm 控制台内容超出屏幕
            formatted_text = textwrap.fill(
                str(text),
                width=95,
                break_long_words=False,
            )

            print(formatted_text)

        else:

            print("未能找到对应的原文内容。")

    print("\n" + "=" * 65)
    print("引用来源展示完成")
    print("=" * 65)


# ==========================================
# 4. 单次论文问答
# ==========================================

def ask_question(
    retriever,
    question,
    paper_id,
    top_k=5,
):

    print("\n[1] 检索相关证据...")

    results = retriever.search(
        query=question,
        paper_id=paper_id,
        top_k=top_k,
    )

    print(f"检索到 {len(results)} 条证据")

    if not results:

        print("未检索到相关证据。")

        return

    print("\n[2] 构造引用上下文...")

    context, sources = build_context(
        results,
        max_chars=8000,
    )

    print(f"可用引用数量：{len(sources)}")

    if not sources:

        print("没有可用的证据内容。")

        return

    print("\n[3] 构造 Prompt...")

    prompt = build_qa_prompt(
        question=question,
        context=context,
    )

    print("\n[4] 调用 DeepSeek 生成回答...")

    answer = generate_answer(prompt)

    # --------------------------------------
    # 展示模型回答
    # --------------------------------------

    print("\n" + "=" * 65)
    print("RAG 最终回答")
    print("=" * 65)

    print(f"\n问题：{question}")

    print("\n回答：")

    print(
        textwrap.fill(
            answer,
            width=95,
            break_long_words=False,
            replace_whitespace=False,
        )
    )

    # --------------------------------------
    # 展示引用来源
    # --------------------------------------

    show_citation_sources(
        answer=answer,
        sources=sources,
        results=results,
    )


# ==========================================
# 5. 交互式问答主程序
# ==========================================

def main():

    print("=" * 65)
    print("科研论文 RAG 交互式问答系统")
    print("=" * 65)

    print("\n[初始化] 加载论文检索器...")

    # 整个程序只初始化一次
    retriever = PaperScopedRetriever()

    print("\n检索器加载完成！")

    print("\n使用说明：")
    print("1. 直接输入问题")
    print("2. 输入 :paper 论文ID 切换论文")
    print("3. 输入 q 退出程序")

    paper_id = "1909.00694"

    print(f"\n当前论文 ID：{paper_id}")

    print("\n" + "=" * 65)
    print("交互式问答已启动")
    print("=" * 65)

    while True:

        try:

            user_input = input(
                "\n请输入问题（q 退出）："
            ).strip()

        except (KeyboardInterrupt, EOFError):

            print("\n用户结束问答。")

            break

        # ----------------------------------
        # 退出程序
        # ----------------------------------

        if user_input.lower() in ["q", "quit", "exit"]:

            print("\n正在退出问答系统...")

            break

        # ----------------------------------
        # 空输入
        # ----------------------------------

        if not user_input:

            print("请输入有效问题。")

            continue

        # ----------------------------------
        # 切换论文
        # ----------------------------------

        if user_input.lower().startswith(":paper "):

            new_paper_id = user_input.split(
                maxsplit=1
            )[1].strip()

            if not new_paper_id:

                print("请输入有效论文 ID。")

                continue

            paper_id = new_paper_id

            print(f"\n已切换到论文：{paper_id}")

            continue

        # ----------------------------------
        # 执行问答
        # ----------------------------------

        try:

            print(f"\n当前论文：{paper_id}")
            print(f"当前问题：{user_input}")

            ask_question(
                retriever=retriever,
                question=user_input,
                paper_id=paper_id,
                top_k=5,
            )

        except Exception as e:

            print("\n问答过程中出现错误：")
            print(f"{type(e).__name__}: {e}")

            print("\n你可以继续输入其他问题。")

    print("\n程序结束。")


# ==========================================
# 6. 程序入口
# ==========================================

if __name__ == "__main__":
    main()