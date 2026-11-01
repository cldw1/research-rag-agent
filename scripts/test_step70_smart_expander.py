
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.retrieval.smart_section_expander import (
    rank_section_neighbors
)


def make_chunk(
    chunk_id,
    page,
    section,
    text
):

    return {
        "chunk_id": chunk_id,
        "paper_id": "paper_test",
        "title": "Test",
        "section_name": section,
        "section_index": page - 1,
        "paragraph_index": 0,
        "piece_index": 0,
        "page_number": page,
        "text": text,
        "char_count": len(text)
    }


def main():

    leapfrog_section = (
        "5.2.3.3 The Leapfrog Method"
    )

    chunks = [
        make_chunk(
            "intro",
            8,
            "5.2.3.2 Other Methods",
            "Unrelated introduction"
        ),
        make_chunk(
            "formula",
            9,
            leapfrog_section,
            (
                "Leapfrog momentum update "
                "p = p - epsilon * gradient "
                "(5.18). Position update "
                "q = q + epsilon * p "
                "(5.19). Momentum update "
                "p = p - epsilon * gradient "
                "(5.20)."
            )
        ),
        make_chunk(
            "background",
            9,
            leapfrog_section,
            (
                "Trajectories may become "
                "unstable with large stepsizes."
            )
        ),
        make_chunk(
            "anchor",
            10,
            leapfrog_section,
            (
                "The leapfrog integrator "
                "combines half-step momentum "
                "and full-step position updates."
            )
        ),
        make_chunk(
            "other_section",
            11,
            "5.3 MCMC from Hamiltonian Dynamics",
            (
                "Other algorithm equations "
                "(5.30) x = y."
            )
        )
    ]

    anchors = [chunks[3]]

    question = (
        "How does the leapfrog integrator "
        "work? Explain the momentum and "
        "position update steps."
    )

    ranked = rank_section_neighbors(
        question=question,
        anchors=anchors,
        all_chunks=chunks,
        max_distance=10
    )

    assert ranked

    # 第9页公式必须排第一
    assert (
        ranked[0]["chunk_id"]
        == "formula"
    )

    # 不得跨真实章节
    assert all(
        item["section_name"]
        == leapfrog_section
        for item in ranked
    )

    # 不得重复 Anchor
    assert all(
        item["chunk_id"] != "anchor"
        for item in ranked
    )

    # 三个核心公式编号均被识别
    assert {
        "5.18", "5.19", "5.20"
    }.issubset(
        set(ranked[0]["equation_numbers"])
    )

    print("=" * 55)
    print("第70步：智能章节邻居测试")
    print("=" * 55)

    for item in ranked:

        print(
            item["chunk_id"],
            "| PDF第",
            item["page_number"],
            "页 | 得分",
            item["neighbor_score"]
        )

        print(
            " 公式：",
            item["equation_numbers"]
        )

    print("\n智能邻居扩展基础测试通过！")


if __name__ == "__main__":
    main()
