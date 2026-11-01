"""Inject a retrieval strategy without modifying ResearchRAGPipeline or PaperSearchTool."""

from src.retrieval.retrieval_strategy import retrieve_with_strategy


class RetrievalStrategyProxy:
    """Delegate metadata; intercept only search() to apply the selected strategy."""

    def __init__(self, retriever, strategy="hybrid"):
        if strategy not in {"hybrid", "section_aware"}:
            raise ValueError(f"不支持的检索策略：{strategy}")
        if strategy == "section_aware" and not getattr(
            retriever, "has_verified_sections", False
        ):
            raise ValueError("当前索引没有经过验证的章节信息，不能启用 Section-Aware")
        self._retriever = retriever
        self.strategy = strategy
        self.search_call_count = 0

    def __getattr__(self, name):
        return getattr(self._retriever, name)

    def search(self, query, paper_id, top_k=5, candidate_k=50):
        results = retrieve_with_strategy(
            retriever=self._retriever,
            query=query,
            paper_id=paper_id,
            strategy=self.strategy,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        self.search_call_count += 1
        return results
