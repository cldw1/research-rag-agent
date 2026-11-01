
import json
import sys
from pathlib import Path


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# 2. 文件路径
# ==========================================

ANSWER_PATH = (
    PROJECT_ROOT
    / "results"
    / "validation_generation_compare_5.json"
)

CHUNK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "qasper_validation_30_index_chunks.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "validation_manual_review_5.md"
)

METHODS = [
    "top10",
    "anchor_first"
]


# ==========================================
# 3. 读取实验结果
# ==========================================

def load_results():

    with open(
        ANSWER_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if data.get("split") != "validation":
        raise ValueError("不是 Validation 实验结果")

    records = data["results"]

    if len(records) != 5:
        raise ValueError(
            f"预期5道问题，实际{len(records)}道"
        )

    return records


# ==========================================
# 4. 读取完整论文 Chunk
# ==========================================

def load_chunks():

    chunk_lookup = {}

    with open(
        CHUNK_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            chunk = json.loads(line)

            chunk_id = chunk["chunk_id"]

            chunk_lookup[chunk_id] = chunk

    return chunk_lookup


# ==========================================
# 5. Markdown 引用格式
# ==========================================

def quote_text(text):

    lines = str(text).strip().splitlines()

    if not lines:
        return "> （空）"

    return "\n".join(
        "> " + line
        for line in lines
    )


# ==========================================
# 6. 整理一个模型回答
# ==========================================

def add_variant(
    lines,
    method,
    variant,
    chunk_lookup
):

    lines.append(f"### 方法：{method}\n")

    lines.append(
        f"- 上下文字符数：{variant['context_chars']}"
    )

    lines.append(
        f"- 上下文 Chunk 数：{variant['chunk_count']}"
    )

    lines.append(
        "- 人工证据进入上下文："
        + str(variant["gold_in_context"])
    )

    lines.append(
        "- 存在无效引用编号："
        + str(bool(variant["invalid_citations"]))
    )

    lines.append("")

    # --------------------------------------
    # 模型完整回答
    # --------------------------------------

    lines.append("**模型完整回答**\n")

    lines.append(
        quote_text(variant["answer"])
    )

    lines.append("")

    # --------------------------------------
    # 查看实际引用的原文
    # --------------------------------------

    lines.append("**模型引用的真实论文原文**\n")

    source_map = {
        int(source["citation"]): source
        for source in variant["sources"]
    }

    cited_numbers = variant["cited_numbers"]

    gold_source_ids = set(
        variant["gold_source_ids"]
    )

    for number in cited_numbers:

        source = source_map.get(number)

        if source is None:

            lines.append(
                f"- 引用 [{number}]：无效编号\n"
            )

            continue

        chunk_id = source["chunk_id"]

        chunk = chunk_lookup.get(chunk_id)

        if chunk is None:
            raise ValueError(
                f"找不到引用原文：{chunk_id}"
            )

        lines.append(f"**引用 [{number}]**")

        lines.append(
            f"- Chunk ID：`{chunk_id}`"
        )

        lines.append(
            f"- 章节：{chunk.get('section_name', '')}"
        )

        lines.append(
            "- 与人工证据文本匹配："
            + str(chunk_id in gold_source_ids)
        )

        lines.append("")

        lines.append(
            quote_text(chunk["text"])
        )

        lines.append("")

    # --------------------------------------
    # 查看未被引用的人工证据
    # --------------------------------------

    cited_ids = {
        source_map[number]["chunk_id"]
        for number in cited_numbers
        if number in source_map
    }

    uncited_gold_ids = (
        gold_source_ids - cited_ids
    )

    if uncited_gold_ids:

        lines.append(
            "**已进入上下文、但模型没有引用的人工证据**\n"
        )

        for chunk_id in sorted(uncited_gold_ids):

            chunk = chunk_lookup.get(chunk_id)

            if chunk is None:
                raise ValueError(
                    f"找不到人工证据 Chunk：{chunk_id}"
                )

            lines.append(
                f"Chunk ID：`{chunk_id}`\n"
            )

            lines.append(
                quote_text(chunk["text"])
            )

            lines.append("")

    # --------------------------------------
    # 人工评价栏
    # --------------------------------------

    lines.append("**人工评价（待填写）**\n")

    lines.append(
        "- 答案正确性：正确 / 部分正确 / 错误 / 无法判断"
    )

    lines.append(
        "- 证据充分性：充分 / 不充分 / 无法判断"
    )

    lines.append(
        "- 答案忠实性：支持 / 部分支持 / 不支持 / 无法判断"
    )

    lines.append(
        "- 引用准确性：准确 / 部分准确 / 不准确 / 无法判断"
    )

    lines.append(
        "- 拒答质量：合理 / 不合理 / 未拒答 / 无法判断"
    )

    lines.append("- 评价理由：待填写\n")


# ==========================================
# 7. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("RAG 人工评价材料整理")
    print("=" * 65)

    print("\n[1] 读取回答结果...")

    records = load_results()

    print("问题数量:", len(records))

    print("\n[2] 读取完整论文 Chunk...")

    chunk_lookup = load_chunks()

    print("Chunk 数量:", len(chunk_lookup))

    lines = [
        "# QASPER Validation RAG 人工质量评估",
        "",
        "## 实验说明",
        "",
        "- 数据集：QASPER Validation",
        "- 问题数量：5",
        "- 模型：DeepSeek",
        "- 比较方法：Hybrid Top-10 与 Anchor-First",
        "- 本文件用于人工审查，不是自动评分结果",
        "",
        "## 评估原则",
        "",
        "1. 正确性：重点检查是否真正回答问题。",
        "2. 证据充分性：检查原文是否足以得出答案。",
        "3. 忠实性：检查模型是否添加无依据的事实。",
        "4. 引用准确性：检查引用内容是否支持对应结论。",
        "5. 拒答质量：区分合理拒答与错误拒答。",
        "",
        "注意：人工证据的文本匹配不等于答案正确。",
        "",
        "---",
        ""
    ]

    print("\n[3] 整理5道问题及10份模型回答...")

    for index, record in enumerate(
        records,
        start=1
    ):

        paper_id = record["paper_id"]

        lines.append(
            f"## 问题 {index}"
        )

        lines.append("")

        lines.append(
            f"**论文 ID：** `{paper_id}`"
        )

        lines.append(
            f"**Question：** {record['question']}"
        )

        lines.append("")

        # ----------------------------------
        # QASPER 人工证据
        # ----------------------------------

        lines.append("### QASPER 人工标注证据\n")

        for evidence_index, evidence in enumerate(
            record["gold_evidence"],
            start=1
        ):

            lines.append(
                f"**人工证据 {evidence_index}**\n"
            )

            lines.append(
                quote_text(evidence)
            )

            lines.append("")

        # ----------------------------------
        # 两组回答
        # ----------------------------------

        for method in METHODS:

            variant = record["variants"].get(
                method
            )

            if variant is None:
                raise ValueError(
                    f"问题 {index} 缺少 {method} 的结果"
                )

            add_variant(
                lines=lines,
                method=method,
                variant=variant,
                chunk_lookup=chunk_lookup
            )

        lines.append("---\n")

        print(
            f"问题 {index}/5 整理完成"
        )

    # ======================================
    # 8. 保存 Markdown 报告
    # ======================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("人工评价材料整理完成！")
    print("=" * 65)

    print("\n输出文件：")
    print(OUTPUT_PATH)

    print("\n问题数量：", len(records))

    print(
        "模型回答数量：",
        len(records) * len(METHODS)
    )

    print("\n说明：")
    print("文件包含完整回答、人工证据和引用原文。")
    print("没有自动填写正确性或忠实性评分。")
    print("没有调用任何大模型 API。")


if __name__ == "__main__":
    main()
