
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.validation_paper_scoped_retriever import (
    ValidationPaperScopedRetriever
)

from src.generation.rag_pipeline import (
    ResearchRAGPipeline
)


QUESTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "qasper_validation_30_questions.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "rag_pipeline_demo.json"
)


def main():

    print("=" * 60)
    print("Research RAG Pipeline 端到端测试")
    print("=" * 60)

    with open(
        QUESTION_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        questions = json.load(f)["questions"]

    sample = questions[0]

    question = sample["question"]
    paper_id = sample["paper_id"]

    print("\n论文:", paper_id)
    print("问题:", question)

    print("\n[1] 初始化检索器...")

    retriever = ValidationPaperScopedRetriever()

    pipeline = ResearchRAGPipeline(
        retriever=retriever,
        max_chars=8000
    )

    print("\n[2] 检索并构造上下文...")

    prepared = pipeline.prepare(
        question=question,
        paper_id=paper_id,
        method="anchor_first"
    )

    print(
        "最终上下文字符数:",
        prepared["context_chars"]
    )

    print(
        "实际 Chunk 数量:",
        prepared["chunk_count"]
    )

    assert prepared["context_chars"] <= 8000

    print("\n引用来源:")

    for source in prepared["sources"]:

        print(
            f"[{source['citation']}] "
            f"{source['section']} "
            f"({source['chunk_id']})"
        )

    print("\n[3] DeepSeek 生成答案...")

    result = pipeline.generate(prepared)

    print("\n" + "=" * 60)
    print("最终回答")
    print("=" * 60)

    print(result["answer"])

    print("\n有效引用数量:")
    print(len(result["valid_citations"]))

    print("\n无效引用:")
    print(result["invalid_citations"])

    # 不重复保存长篇 Prompt 和 Context
    output = {
        key: value
        for key, value in result.items()
        if key not in ("prompt", "context")
    }

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
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n结果文件:")
    print(OUTPUT_PATH)

    print("\nRAG Pipeline 端到端测试完成！")


if __name__ == "__main__":
    main()
