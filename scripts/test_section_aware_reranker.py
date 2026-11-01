
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.retrieval.section_aware_reranker import (
    constrained_section_rerank
)


def make_item(chunk_id, section, score):

    return {
        "chunk_id": chunk_id,
        "section_name": section,
        "rrf_score": score,
        "page_number": 1,
        "text": f"Test passage: {chunk_id}"
    }


def main():

    query = (
        "How are the step size "
        "and trajectory length tuned in HMC?"
    )

    candidates = [
        make_item(
            "A",
            "5.4.2 Tuning HMC",
            0.03132
        ),
        make_item(
            "B",
            "5.5.6 Short-Cut Trajectories",
            0.03009
        ),
        make_item(
            "C",
            "5.5.6 Short-Cut Trajectories",
            0.02900
        ),
        make_item(
            "D",
            "5.5.6 Short-Cut Trajectories",
            0.02800
        ),
        make_item(
            "E",
            "5.4.1 Linear Transformations",
            0.02700
        ),
        make_item(
            "F",
            "References",
            0.02600
        ),
        make_item(
            "G",
            "5.3.2 HMC Algorithm",
            0.02500
        )
    ]

    baseline = candidates[:5]

    results = constrained_section_rerank(
        query=query,
        baseline_results=baseline,
        candidates=candidates,
        top_k=5
    )

    assert len(results) == 5

    # 原始Top-1仍在
    result_ids = {
        item["chunk_id"]
        for item in results
    }

    assert "A" in result_ids

    # 每个章节最多2个
    counts = Counter(
        item["section_name"]
        for item in results
    )

    assert max(counts.values()) <= 2

    # rank必须重新编号
    assert [
        item["rank"]
        for item in results
    ] == [1, 2, 3, 4, 5]

    # 没有重复Chunk
    assert len(result_ids) == 5

    print("=" * 50)
    print("章节重排序模块测试")
    print("=" * 50)

    for item in results:

        print(
            f"Top {item['rank']} | "
            f"{item['chunk_id']} | "
            f"{item['section_name']} | "
            f"{item['constrained_score']:.5f}"
        )

    print("\n所有断言通过！")


if __name__ == "__main__":
    main()
