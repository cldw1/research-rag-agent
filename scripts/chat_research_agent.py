
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# 1. 设置项目根目录
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from scripts.demo_agent_live_dedup import (
    ResearchAgent,
    MODEL,
    MAX_TOTAL_TOOL_CALLS
)

from src.retrieval.validation_paper_scoped_retriever import (
    ValidationPaperScopedRetriever
)


# ==========================================
# 2. 配置
# ==========================================

QUESTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "qasper_validation_30_questions.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "research_agent_chat_history.json"
)


# ==========================================
# 3. 创建 Agent
# ==========================================

def create_agent(client, retriever, paper_id):

    return ResearchAgent(
        client=client,
        retriever=retriever,
        paper_id=paper_id
    )


# ==========================================
# 4. 保存聊天记录
# ==========================================

def save_history(history):

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
                "experiment": "Interactive Research Agent",
                "model": MODEL,
                "records": history
            },
            f,
            ensure_ascii=False,
            indent=2
        )


# ==========================================
# 5. 查看引用原文
# ==========================================

def show_source(
    retriever,
    agent,
    number
):

    source = agent.registry.get_source(
        number
    )

    if source is None:

        print("没有找到这个引用编号。")
        return

    chunks = retriever.get_paper_data(
        agent.paper_id
    )["chunks"]

    for chunk in chunks:

        if chunk["chunk_id"] == source["chunk_id"]:

            print("\n" + "=" * 60)
            print(f"引用 [{number}]")
            print("=" * 60)

            print("论文:", chunk["title"])
            print("章节:", chunk["section_name"])
            print("Chunk ID:", chunk["chunk_id"])

            print("\n论文原文：")
            print(chunk["text"])

            return

    print("找到引用元数据，但没有找到原文。")


# ==========================================
# 6. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("Research RAG Agent - 交互式科研论文助手")
    print("=" * 65)

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "请检查项目根目录的 .env 文件"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    print("\n正在加载验证集检索器...")

    retriever = ValidationPaperScopedRetriever()

    with open(
        QUESTION_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    default_paper_id = str(
        data["questions"][0]["paper_id"]
    )

    current_paper_id = default_paper_id

    agent = create_agent(
        client,
        retriever,
        current_paper_id
    )

    history = []
    session_id = 1

    print("\nAgent 初始化成功！")
    print("当前论文:", current_paper_id)

    print("\n可用命令：")
    print("/papers         查看可用论文")
    print("/paper 论文ID   切换论文")
    print("/source 编号    查看引用原文")
    print("/reset          重置当前对话")
    print("/exit           退出程序")

    print("\n直接输入自然语言问题即可。\n")

    # ======================================
    # 7. 交互循环
    # ======================================

    while True:

        try:
            user_input = input(
                "\n你："
            ).strip()

        except (KeyboardInterrupt, EOFError):
            print("\n已退出。")
            break

        if not user_input:
            continue

        # ----------------------------------
        # 退出
        # ----------------------------------

        if user_input == "/exit":

            print("再见！")
            break

        # ----------------------------------
        # 查看论文列表
        # ----------------------------------

        if user_input == "/papers":

            print("\n当前索引中的论文：")

            for paper_id in sorted(
                retriever.paper_rows
            ):

                paper_data = (
                    retriever.get_paper_data(
                        paper_id
                    )
                )

                title = paper_data["chunks"][0].get(
                    "title", ""
                )

                print(
                    f"{paper_id} | {title}"
                )

            continue

        # ----------------------------------
        # 切换论文
        # ----------------------------------

        if user_input.startswith("/paper "):

            new_paper_id = (
                user_input.split(
                    maxsplit=1
                )[1].strip()
            )

            if new_paper_id not in retriever.paper_rows:

                print(
                    "当前验证集索引中没有这篇论文。"
                )

                continue

            current_paper_id = new_paper_id

            session_id += 1

            agent = create_agent(
                client,
                retriever,
                current_paper_id
            )

            print(
                "已切换论文：",
                current_paper_id
            )

            print(
                "对话历史和引用编号已重新开始。"
            )

            continue

        # ----------------------------------
        # 重置对话
        # ----------------------------------

        if user_input == "/reset":

            session_id += 1

            agent = create_agent(
                client,
                retriever,
                current_paper_id
            )

            print(
                "当前对话已重置，引用从[1]开始。"
            )

            continue

        # ----------------------------------
        # 查看引用原文
        # ----------------------------------

        if user_input.startswith("/source "):

            value = user_input.split(
                maxsplit=1
            )[1].strip()

            if not value.isdigit():

                print(
                    "请输入数字，例如：/source 3"
                )

                continue

            show_source(
                retriever,
                agent,
                int(value)
            )

            continue

        # ----------------------------------
        # 到达会话上限，自动重置
        # ----------------------------------

        if (
            agent.total_tool_calls
            >= MAX_TOTAL_TOOL_CALLS
        ):

            print(
                "\n当前会话已达到工具调用上限，"
                "将开始新的会话。"
            )

            session_id += 1

            agent = create_agent(
                client,
                retriever,
                current_paper_id
            )

            print(
                "新会话引用编号从[1]开始。"
            )

        # ==================================
        # 8. 正式调用 Research Agent
        # ==================================

        try:

            result = agent.ask(
                user_input
            )

        except Exception as exc:

            print("\n本次请求未完成：")
            print(
                type(exc).__name__,
                str(exc)
            )

            print(
                "为避免保留不完整的工具消息，"
                "当前对话将重置。"
            )

            session_id += 1

            agent = create_agent(
                client,
                retriever,
                current_paper_id
            )

            continue

        # ----------------------------------
        # 展示答案
        # ----------------------------------

        print("\n" + "=" * 65)
        print("Research Agent 回答")
        print("=" * 65)

        print(result["answer"])

        print("\n工具调用次数:",
              result["executed_tool_count"])

        print("引用编号:",
              result["cited_numbers"])

        print("无效引用:",
              result["invalid_citations"])

        # ----------------------------------
        # 保存本轮记录
        # ----------------------------------

        history.append({
            "session_id": session_id,
            "paper_id": current_paper_id,
            "question": user_input,
            "answer": result["answer"],
            "executed_tool_count": (
                result["executed_tool_count"]
            ),
            "tool_names": [
                call["tool_name"]
                for call in result["tool_calls"]
            ],
            "cited_numbers": result["cited_numbers"],
            "cited_sources": result["cited_sources"],
            "invalid_citations": result["invalid_citations"],
            "placeholder_found": result["placeholder_found"]
        })

        save_history(history)

        print(
            "\n聊天记录已保存。"
        )


if __name__ == "__main__":
    main()
