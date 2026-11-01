
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.paper_scoped_retriever import (
    PaperScopedRetriever
)
from src.generation.context_builder import build_context


# ==========================================
# 实验配置
# ==========================================

PAPER_ID = "1905.10810"
QUESTION = "How is PIEWi annotated?"

TOP_K = 5
WINDOW = 1
MAX_CHARS = 8000

DIAGNOSIS_PATH = (
    PROJECT_ROOT
    / "results"
    / "rag_failure_diagnosis_2.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "context_expansion_comparison.json"
)


# ==========================================
# 1. 加载人工证据对应的 Chunk ID
# ==========================================

def load_gold_chunk_ids():

    with open(
        DIAGNOSIS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    for case in data["results"]:

        if str(case["paper_id"]) == PAPER_ID:

            return set(
                case["matching_gold_chunk_ids"]
            )

    raise ValueError("没有找到目标论文的诊断数据")


# ==========================================
# 2. 扩展相邻段落
# ==========================================

def expand_neighbors(
    retrieved,
    all_chunks,
    window=1
):

    # 根据 Chunk ID 找到完整元数据
    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in all_chunks
    }

    # 按章节和自然段落组织
    paragraph_map = defaultdict(list)

    for chunk in all_chunks:

        key = (
            chunk["section_index"],
            chunk["paragraph_index"]
        )

        paragraph_map[key].append(chunk)

    # 确保一个段落被拆分后仍按顺序排列
    for parts in paragraph_map.values():

        parts.sort(
            key=lambda x: x["piece_index"]
        )

    expanded = []
    seen_ids = set()

    def add_chunk(chunk, role):

        chunk_id = chunk["chunk_id"]

        if chunk_id in seen_ids:
            return

        seen_ids.add(chunk_id)

        record = chunk.copy()
        record["source_role"] = role

        expanded.append(record)

    for result in retrieved:

        chunk_id = result["chunk_id"]

        if chunk_id not in chunk_lookup:
            raise ValueError(
                f"找不到 Chunk：{chunk_id}"
            )

        anchor = chunk_lookup[chunk_id]

        section_index = anchor["section_index"]
        paragraph_index = anchor["paragraph_index"]

        # 先保留原始检索结果
        add_chunk(anchor, "retrieved")

        # 再补充前后相邻段落
        for offset in range(-window, window + 1):

            if offset == 0:
                continue

            neighbor_key = (
                section_index,
                paragraph_index + offset
            )

            neighbors = paragraph_map.get(
                neighbor_key,
                []
            )

            for neighbor in neighbors:

                add_chunk(
                    neighbor,
                    "neighbor"
                )

    return expanded


# ==========================================
# 3. 评价上下文中的证据覆盖
# ==========================================

def evaluate_context(
    chunks,
    gold_chunk_ids
):

    context, sources = build_context(
        chunks,
        max_chars=MAX_CHARS
    )

    included_ids = {
        source["chunk_id"]
        for source in sources
    }

    matched_ids = sorted(
        included_ids & gold_chunk_ids
    )

    return {
        "included_count": len(sources),
        "context_chars": len(context),
        "gold_hit": bool(matched_ids),
        "matched_gold_ids": matched_ids,
        "included_chunk_ids": [
            source["chunk_id"]
            for source in sources
        ]
    }


# ==========================================
# 4. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("科研 RAG Agent - 相邻段落扩展实验")
    print("=" * 65)

    gold_chunk_ids = load_gold_chunk_ids()

    print("\n[1] 人工证据对应 Chunk")
    print(sorted(gold_chunk_ids))

    print("\n[2] 初始化检索器...")

    retriever = PaperScopedRetriever()

    paper_data = retriever.get_paper_data(
        PAPER_ID
    )

    print(
        "当前论文 Chunk 数量:",
        len(paper_data["chunks"])
    )

    # --------------------------------------
    # Baseline：原始 Hybrid Top-5
    # --------------------------------------

    print("\n[3] 原始 Hybrid 检索...")

    retrieved = retriever.search(
        query=QUESTION,
        paper_id=PAPER_ID,
        top_k=TOP_K
    )

    baseline = evaluate_context(
        retrieved,
        gold_chunk_ids
    )

    # --------------------------------------
    # 改进：相邻段落扩展
    # --------------------------------------

    print("\n[4] 相邻段落扩展...")

    expanded = expand_neighbors(
        retrieved=retrieved,
        all_chunks=paper_data["chunks"],
        window=WINDOW
    )

    improved = evaluate_context(
        expanded,
        gold_chunk_ids
    )

    # --------------------------------------
    # 输出对比
    # --------------------------------------

    print("\n[5] 实验结果")

    print("\nBaseline：Hybrid Top-5")
    print("上下文 Chunk 数:", baseline["included_count"])
    print("上下文字符数:", baseline["context_chars"])
    print("是否包含标准证据:", baseline["gold_hit"])

    print("\n改进：Hybrid + Neighbor Expansion")
    print("候选 Chunk 数:", len(expanded))
    print("实际纳入 Chunk 数:", improved["included_count"])
    print("上下文字符数:", improved["context_chars"])
    print("是否包含标准证据:", improved["gold_hit"])

    print(
        "命中的标准证据:",
        improved["matched_gold_ids"]
    )

    print("\n[6] 新增段落检查")

    original_ids = {
        item["chunk_id"]
        for item in retrieved
    }

    included_ids = set(
        improved["included_chunk_ids"]
    )

    for item in expanded:

        chunk_id = item["chunk_id"]

        if (
            chunk_id in included_ids
            and chunk_id not in original_ids
        ):

            print(
                "新增:",
                chunk_id
            )

            if chunk_id in gold_chunk_ids:

                print("  *** 包含人工标注证据 ***")

                print(
                    "  正文:",
                    item["text"][:400]
                )

    # --------------------------------------
    # 保存结果
    # --------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "experiment": "Neighbor Context Expansion",
        "dataset_split": "train",
        "paper_id": PAPER_ID,
        "question": QUESTION,
        "top_k": TOP_K,
        "window": WINDOW,
        "max_chars": MAX_CHARS,
        "gold_chunk_ids": sorted(gold_chunk_ids),
        "baseline": baseline,
        "expanded": improved
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

    print("\n结果保存位置:", OUTPUT_PATH)

    print("\n注意：")
    print("人工标准答案只用于离线评测。")
    print("段落扩展本身不使用标准答案。")
    print("这只是单题诊断，不代表总体效果提升。")

    print("\n实验完成！")


if __name__ == "__main__":
    main()
