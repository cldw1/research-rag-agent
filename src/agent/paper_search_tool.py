
from src.generation.rag_pipeline import (
    ResearchRAGPipeline
)


class PaperSearchTool:
    """
    提供给 Research Agent 使用的论文检索工具。
    """

    name = "search_research_paper"

    description = (
        "Search a specified research paper for evidence "
        "relevant to a question. "
        "Requires a paper ID and a question."
    )

    def __init__(
        self,
        retriever,
        max_chars=8000
    ):
        self.pipeline = ResearchRAGPipeline(
            retriever=retriever,
            max_chars=max_chars
        )

    def run(
        self,
        question,
        paper_id,
        method="anchor_first"
    ):
        # 复用已完成的 RAG Pipeline
        prepared = self.pipeline.prepare(
            question=question,
            paper_id=paper_id,
            method=method
        )

        # 仅返回 Agent 需要的信息
        return {
            "tool_name": self.name,
            "paper_id": prepared["paper_id"],
            "question": prepared["question"],
            "method": prepared["method"],
            "evidence": prepared["context"],
            "sources": prepared["sources"],
            "context_chars": prepared["context_chars"],
            "chunk_count": prepared["chunk_count"]
        }
