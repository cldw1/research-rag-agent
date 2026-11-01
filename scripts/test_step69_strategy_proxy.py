"""Run from project root: python scripts/test_step69_strategy_proxy.py"""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.strategy_proxy import RetrievalStrategyProxy


def chunk(chunk_id, section, score):
    return {
        "chunk_id": chunk_id,
        "paper_id": "paper_test",
        "section_name": section,
        "rrf_score": score,
        "page_number": 1,
        "text": f"Passage {chunk_id}",
    }


class FakeRetriever:
    has_verified_sections = True

    def __init__(self):
        self.chunks = [
            chunk("A", "5.4.2 Tuning HMC", 0.03132),
            chunk("B", "5.5.6 Short-Cut Trajectories", 0.03009),
            chunk("C", "5.5.6 Short-Cut Trajectories", 0.02900),
            chunk("D", "5.5.6 Short-Cut Trajectories", 0.02800),
            chunk("E", "5.4.1 Linear Transformations", 0.02700),
            chunk("F", "References", 0.02600),
            chunk("G", "5.3.2 HMC Algorithm", 0.02500),
        ]
        self.paper_rows = {"paper_test": list(range(len(self.chunks)))}

    def get_paper_data(self, paper_id):
        assert paper_id == "paper_test"
        return {"chunks": self.chunks}

    def search(self, query, paper_id, top_k=5, candidate_k=50):
        assert paper_id == "paper_test"
        return [
            {**item, "rank": i}
            for i, item in enumerate(self.chunks[:top_k], 1)
        ]


def main():
    query = "How are the step size and trajectory length tuned in HMC?"
    base = FakeRetriever()
    original = RetrievalStrategyProxy(base, "hybrid")
    aware = RetrievalStrategyProxy(base, "section_aware")
    assert original.paper_rows == base.paper_rows
    assert aware.get_paper_data("paper_test")["chunks"] == base.chunks

    a = original.search(query, "paper_test", top_k=5)
    b = aware.search(query, "paper_test", top_k=5)
    assert [x["chunk_id"] for x in a] == list("ABCDE")
    assert [x["chunk_id"] for x in b] == ["A", "B", "C", "E", "G"]
    assert original.search_call_count == 1
    assert aware.search_call_count == 1
    assert max(Counter(x["section_name"] for x in b).values()) == 2

    class PageOnlyRetriever(FakeRetriever):
        has_verified_sections = False

    try:
        RetrievalStrategyProxy(PageOnlyRetriever(), "section_aware")
    except ValueError:
        pass
    else:
        raise AssertionError("页面分块索引不得启用 section_aware")

    print("第69步：策略代理测试通过！")
    print("Hybrid:", [x["chunk_id"] for x in a])
    print("Section-Aware:", [x["chunk_id"] for x in b])
    print("普通 PDF 的章节模式已正确禁止")


if __name__ == "__main__":
    main()
