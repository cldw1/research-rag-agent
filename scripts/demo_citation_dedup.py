
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.agent.citation_registry import (
    GlobalCitationRegistry
)


INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "research_agent_unique_citations_demo.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "agent_citation_dedup_demo.json"
)


def main():

    print("=" * 60)
    print("Research Agent 跨轮次 Chunk 去重测试")
    print("=" * 60)

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    registry = GlobalCitationRegistry()

    records = []
    total_original = 0

    tool_calls = data["tool_calls"]

    print(
        "\n历史工具调用次数:",
        len(tool_calls)
    )

    for call_index, call in enumerate(
        tool_calls,
        start=1
    ):

        sources = call["sources"]

        total_original += len(sources)

        result = registry.register(
            sources
        )

        print(
            f"\n第 {call_index} 次检索"
        )

        print(
            "原始 Chunk 数:",
            len(sources)
        )

        print(
            "新增 Chunk 数:",
            len(result["new_sources"])
        )

        print(
            "重复 Chunk 数:",
            len(result["reused_sources"])
        )

        print("引用编号映射：")

        for old_number, new_number in (
            result["citation_map"].items()
        ):

            print(
                f"[{old_number}] -> [{new_number}]"
            )

        records.append({
            "call_index": call_index,
            "original_count": len(sources),
            "new_count": len(
                result["new_sources"]
            ),
            "duplicate_count": len(
                result["reused_sources"]
            ),
            "citation_map": (
                result["citation_map"]
            ),
            "new_sources": (
                result["new_sources"]
            ),
            "reused_sources": (
                result["reused_sources"]
            )
        })

    # ======================================
    # 检查去重正确性
    # ======================================

    all_sources = registry.get_all_sources()

    unique_keys = {
        (
            source["paper_id"],
            source["chunk_id"]
        )
        for call in tool_calls
        for source in call["sources"]
    }

    assert len(all_sources) == len(unique_keys)

    all_numbers = [
        source["citation"]
        for source in all_sources
    ]

    assert all_numbers == list(
        range(
            1,
            len(all_sources) + 1
        )
    )

    assert (
        total_original
        == len(all_sources)
        + registry.duplicate_count
    )

    # ======================================
    # 输出结果
    # ======================================

    print("\n" + "=" * 60)
    print("跨轮次去重结果")
    print("=" * 60)

    print(
        "原始检索 Chunk 总数:",
        total_original
    )

    print(
        "不同 Chunk 数:",
        registry.get_unique_count()
    )

    print(
        "重复 Chunk 数:",
        registry.duplicate_count
    )

    print(
        "引用编号连续唯一:",
        True
    )

    # ======================================
    # 保存报告
    # ======================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "experiment": "Agent Citation Deduplication",
        "paper_id": data["paper_id"],
        "tool_call_count": len(tool_calls),
        "original_count": total_original,
        "unique_count": (
            registry.get_unique_count()
        ),
        "duplicate_count": (
            registry.duplicate_count
        ),
        "calls": records,
        "all_sources": all_sources
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n结果已保存：")
    print(OUTPUT_PATH)

    print("\n跨轮次引用去重测试通过！")


if __name__ == "__main__":
    main()
