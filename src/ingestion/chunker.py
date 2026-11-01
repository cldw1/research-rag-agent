
import json
import textwrap
from pathlib import Path

from datasets import load_dataset


# ==========================================
# 项目二：科研 RAG Agent
# P2-02：论文段落分块
# ==========================================

MAX_CHARS = 1200


def build_chunks(paper, max_chars=MAX_CHARS):
    """
    将一篇 QASPER 论文划分为多个文本块。

    参数：
        paper: QASPER 中的一篇论文
        max_chars: 每个文本块允许的最大字符数

    返回：
        chunks: 文本块列表
    """

    chunks = []

    paper_id = str(paper["id"])
    title = paper["title"]

    full_text = paper["full_text"]

    section_names = full_text["section_name"]
    section_paragraphs = full_text["paragraphs"]

    # 遍历每一个章节
    for section_index, (section_name, paragraphs) in enumerate(
        zip(section_names, section_paragraphs)
    ):

        # 遍历章节中的每一个自然段
        for paragraph_index, paragraph in enumerate(paragraphs):

            if not paragraph:
                continue

            # 清理多余空白
            clean_text = " ".join(paragraph.split())

            if not clean_text:
                continue

            # 将过长段落切分成多个片段
            pieces = textwrap.wrap(
                clean_text,
                width=max_chars,
                break_long_words=True,
                break_on_hyphens=False
            )

            for piece_index, piece in enumerate(pieces):

                # 为每个文本块生成唯一编号
                chunk_id = (
                    f"{paper_id}_"
                    f"s{section_index:03d}_"
                    f"p{paragraph_index:03d}_"
                    f"c{piece_index:03d}"
                )

                chunk = {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "title": title,
                    "section_index": section_index,
                    "section_name": section_name,
                    "paragraph_index": paragraph_index,
                    "piece_index": piece_index,
                    "text": piece,
                    "char_count": len(piece)
                }

                chunks.append(chunk)

    return chunks


def main():
    # 1. 定位项目目录
    project_root = Path(__file__).resolve().parents[2]

    cache_dir = project_root / "data" / "raw" / "hf_cache"
    output_dir = project_root / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 加载已经下载的 QASPER 数据
    base_url = (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/refs%2Fconvert%2Fparquet/qasper"
    )

    print("=" * 60)
    print("QASPER 论文文本分块")
    print("=" * 60)

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{base_url}/train/0000.parquet"
        },
        split="train",
        cache_dir=str(cache_dir)
    )

    # 3. 先选择第一篇论文进行实验
    paper = dataset[0]

    print("\n[1] 论文基本信息")
    print("论文 ID:", paper["id"])
    print("论文标题:", paper["title"])

    # 4. 执行文本分块
    chunks = build_chunks(
        paper,
        max_chars=MAX_CHARS
    )

    print("\n[2] 分块结果")
    print("文本块总数:", len(chunks))

    if chunks:
        lengths = [chunk["char_count"] for chunk in chunks]

        print("最大文本块长度:", max(lengths))
        print("最小文本块长度:", min(lengths))
        print("平均文本块长度:", round(sum(lengths) / len(lengths), 2))

        # 验证长度约束
        assert all(length <= MAX_CHARS for length in lengths)
        print("长度检查：通过")

    # 5. 展示前 3 个文本块
    print("\n[3] 前 3 个文本块")

    for chunk in chunks[:3]:
        print("\n" + "-" * 60)
        print("Chunk ID:", chunk["chunk_id"])
        print("章节:", chunk["section_name"])
        print("段落编号:", chunk["paragraph_index"])
        print("字符数:", chunk["char_count"])
        print("文本内容:")
        print(chunk["text"][:300])

    # 6. 保存结果
    output_path = output_dir / "qasper_sample_chunks.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    print("\n[4] 数据保存")
    print("保存路径:", output_path)

    print("\n" + "=" * 60)
    print("论文分块完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
