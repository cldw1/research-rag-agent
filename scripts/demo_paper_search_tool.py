
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.validation_paper_scoped_retriever import (
    ValidationPaperScopedRetriever
)

from src.agent.paper_search_tool import (
    PaperSearchTool
)


QUESTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "qasper_validation_30_questions.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "paper_search_tool_demo.json"
)


def main():

    print("=" * 60)
    print("Research Agent - 论文检索工具测试")
    print("=" * 60)

    with open(
        QUESTION_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        questions = json.load(f)["questions"]

    sample = questions[0]

    retriever = ValidationPaperScopedRetriever()

    tool = PaperSearchTool(
        retriever=retriever,
        max_chars=8000
    )

    print("\n工具名称:", tool.name)
    print("工具描述:", tool.description)

    print("\n[1] Agent 模拟发起工具调用...")

    result = tool.run(
        question=sample["question"],
        paper_id=sample["paper_id"]
    )

    print("\n[2] 工具执行完成")

    print("论文 ID:", result["paper_id"])
    print("问题:", result["question"])
    print("方法:", result["method"])
    print("上下文字符数:", result["context_chars"])
    print("证据 Chunk 数:", result["chunk_count"])

    print("\n[3] 引用来源")

    for source in result["sources"]:

        print(
            f"[{source['citation']}] "
            f"{source['section']}"
        )

    print("\n[4] 证据内容预览")

    print(result["evidence"][:500])

    # ======================================
    # 检查工具结果
    # ======================================

    assert result["context_chars"] <= 8000

    assert result["chunk_count"] > 0

    assert all(
        str(source["paper_id"])
        == str(sample["paper_id"])
        for source in result["sources"]
    )

    # ======================================
    # 保存结果
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
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n结果已保存:")
    print(OUTPUT_PATH)

    print("\n论文检索工具测试通过！")
    print("本次没有调用 DeepSeek API。")


if __name__ == "__main__":
    main()
