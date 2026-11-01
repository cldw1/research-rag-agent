
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.retrieval.retrieval_strategy import (
    retrieve_with_strategy
)


# ==========================================
# 1. 模拟 Chunk
# ==========================================

def make_chunk(
    chunk_id,
    section,
    score
):

    return {
        "chunk_id": chunk_id,
        "paper_id": "paper_test",
        "section_name": section,
        "rrf_score": score,
        "page_number": 1,
        "text": f"Test passage {chunk_id}"
    }


# ==========================================
# 2. 模拟已有 V3 Retriever
# ==========================================

class FakeV3Retriever:

    def __init__(self):

        self.chunks = [
            make_chunk(
                "A",
                "5.4.2 Tuning HMC",
                0.03132
            ),
            make_chunk(
                "B",
                "5.5.6 Short-Cut Trajectories",
                0.03009
            ),
            make_chunk(
                "C",
                "5.5.6 Short-Cut Trajectories",
                0.02900
            ),
            make_chunk(
                "D",
                "5.5.6 Short-Cut Trajectories",
                0.02800
            ),
            make_chunk(
                "E",
                "5.4.1 Linear Transformations",
                0.02700
            ),
            make_chunk(
                "F",
                "References",
                0.02600
            ),
            make_chunk(
                "G",
                "5.3.2 HMC Algorithm",
                0.02500
            )
        ]

        self.paper_rows = {
            "paper_test": list(
                range(len(self.chunks))
            )
        }

        self.calls = []

    def search(
        self,
        query,
        paper_id,
        top_k,
        candidate_k
    ):

        if paper_id != "paper_test":
            raise ValueError(
                "paper_id 不一致"
            )

        self.calls.append({
            "query": query,
            "top_k": top_k,
            "candidate_k": candidate_k
        })

        ranked = sorted(
            self.chunks,
            key=lambda x: -x["rrf_score"]
        )

        return [
            {
                **item,
                "rank": rank
            }
            for rank, item in enumerate(
                ranked[:top_k],
                start=1
            )
        ]


# ==========================================
# 3. 测试两种策略
# ==========================================

def main():

    retriever = FakeV3Retriever()

    question = (
        "How are the step size and "
        "trajectory length tuned in HMC?"
    )

    # A. 原始 Hybrid
    hybrid = retrieve_with_strategy(
        retriever=retriever,
        query=question,
        paper_id="paper_test",
        strategy="hybrid",
        top_k=5
    )

    # B. 章节感知
    section_aware = retrieve_with_strategy(
        retriever=retriever,
        query=question,
        paper_id="paper_test",
        strategy="section_aware",
        top_k=5
    )

    print("=" * 55)
    print("第67步：检索策略切换测试")
    print("=" * 55)

    print("\n原始 Hybrid：")

    for item in hybrid:

        print(
            f"Top {item['rank']} | "
            f"{item['chunk_id']} | "
            f"{item['section_name']}"
        )

    print("\n章节感知检索：")

    for item in section_aware:

        print(
            f"Top {item['rank']} | "
            f"{item['chunk_id']} | "
            f"{item['section_name']}"
        )

    # ======================================
    # 4. 自动断言
    # ======================================

    hybrid_ids = [
        item["chunk_id"]
        for item in hybrid
    ]

    section_ids = [
        item["chunk_id"]
        for item in section_aware
    ]

    # 默认模式不应改变原始顺序
    assert hybrid_ids == [
        "A", "B", "C", "D", "E"
    ]

    # 章节模式排除第三个重复章节
    assert section_ids == [
        "A", "B", "C", "E", "G"
    ]

    # 两种模式都应返回5条
    assert len(hybrid) == 5
    assert len(section_aware) == 5

    # 章节模式保留原始Top-1
    assert hybrid_ids[0] in section_ids

    # 每章节最多2条
    counts = Counter(
        item["section_name"]
        for item in section_aware
    )

    assert max(counts.values()) <= 2

    # 重新编号
    assert [
        item["rank"]
        for item in section_aware
    ] == [1, 2, 3, 4, 5]

    # 检索策略字段正确
    assert all(
        item["retrieval_strategy"] == "hybrid"
        for item in hybrid
    )

    assert all(
        item["retrieval_strategy"] == "section_aware"
        for item in section_aware
    )

    # 原始Hybrid调用一次，
    # 章节感知调用两次。
    assert len(retriever.calls) == 3

    # 检查非法策略
    try:

        retrieve_with_strategy(
            retriever=retriever,
            query=question,
            paper_id="paper_test",
            strategy="unknown"
        )

        raise AssertionError(
            "非法策略应该触发ValueError"
        )

    except ValueError:
        pass

    print("\n所有检索策略断言通过！")


if __name__ == "__main__":
    main()
