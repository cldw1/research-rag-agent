
import json
from pathlib import Path
from collections import Counter


# ==========================================
# P2-02：论文分块质量检查
# ==========================================

MAX_CHARS = 1200


def main():
    project_root = Path(__file__).resolve().parents[1]

    input_path = (
        project_root
        / "data"
        / "processed"
        / "qasper_sample_chunks.jsonl"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到分块文件：{input_path}"
        )

    # 1. 读取 JSONL 文件
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    print("=" * 60)
    print("QASPER 论文分块质量检查")
    print("=" * 60)

    print("\n[1] 基础统计")
    print("Chunk 总数:", len(chunks))

    if not chunks:
        raise ValueError("分块文件为空")

    lengths = [len(chunk["text"]) for chunk in chunks]

    print("平均长度:", round(sum(lengths) / len(lengths), 2))
    print("最短长度:", min(lengths))
    print("最长长度:", max(lengths))

    # 2. 检查空文本
    empty_chunks = [
        chunk for chunk in chunks
        if not chunk["text"].strip()
    ]

    # 3. 检查超长文本
    overlong_chunks = [
        chunk for chunk in chunks
        if len(chunk["text"]) > MAX_CHARS
    ]

    # 4. 检查重复 ID
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    duplicate_count = (
        len(chunk_ids) - len(set(chunk_ids))
    )

    # 5. 检查元数据是否完整
    required_fields = {
        "chunk_id",
        "paper_id",
        "title",
        "section_name",
        "paragraph_index",
        "text",
        "char_count"
    }

    missing_metadata = [
        chunk for chunk in chunks
        if not required_fields.issubset(chunk.keys())
    ]

    # 6. 检查记录的字符数是否正确
    incorrect_lengths = [
        chunk for chunk in chunks
        if chunk["char_count"] != len(chunk["text"])
    ]

    print("\n[2] 数据质量检查")

    print("空文本数量:", len(empty_chunks))
    print("超长文本数量:", len(overlong_chunks))
    print("重复 Chunk ID 数量:", duplicate_count)
    print("元数据缺失数量:", len(missing_metadata))
    print("字符数记录错误数量:", len(incorrect_lengths))

    # 7. 分析短文本块
    short_chunks = [
        chunk for chunk in chunks
        if len(chunk["text"]) < 100
    ]

    short_ratio = len(short_chunks) / len(chunks)

    print("\n[3] 短文本分析")
    print("长度小于 100 的 Chunk 数量:", len(short_chunks))
    print("短文本比例:", f"{short_ratio:.2%}")

    # 8. 统计各章节的 Chunk 数量
    section_counts = Counter(
        chunk["section_name"]
        for chunk in chunks
    )

    print("\n[4] 各章节 Chunk 数量")

    for section, count in section_counts.items():
        print(f"{section}: {count}")

    # 9. 最终检查
    passed = (
        len(empty_chunks) == 0
        and len(overlong_chunks) == 0
        and duplicate_count == 0
        and len(missing_metadata) == 0
        and len(incorrect_lengths) == 0
    )

    print("\n[5] 最终结果")

    if passed:
        print("基础数据质量检查：通过")
    else:
        print("基础数据质量检查：未通过，请检查上述问题")

    print("\n" + "=" * 60)
    print("质量检查完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
