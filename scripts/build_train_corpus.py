
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


# ==========================================
# P2-02：批量构建 QASPER 训练集知识库
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 保证可以导入项目中的 src 模块
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunker import build_chunks


def main():
    cache_dir = PROJECT_ROOT / "data" / "raw" / "hf_cache"
    output_dir = PROJECT_ROOT / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 QASPER 训练集
    base_url = (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/refs%2Fconvert%2Fparquet/qasper"
    )

    print("=" * 60)
    print("QASPER 训练集知识库构建")
    print("=" * 60)

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{base_url}/train/0000.parquet"
        },
        split="train",
        cache_dir=str(cache_dir)
    )

    print("\n训练集论文数量:", len(dataset))

    # 2. 设置输出文件
    output_path = output_dir / "qasper_train_chunks.jsonl"

    total_chunks = 0
    total_chars = 0
    short_chunks = 0
    empty_papers = 0
    max_length = 0

    # 3. 批量处理论文
    with open(output_path, "w", encoding="utf-8") as f:

        for paper in tqdm(dataset, desc="处理论文", unit="篇"):

            chunks = build_chunks(paper)

            if not chunks:
                empty_papers += 1
                continue

            for chunk in chunks:
                text_length = chunk["char_count"]

                # 保存一个文本块
                f.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False
                    ) + "\n"
                )

                # 统计数据
                total_chunks += 1
                total_chars += text_length

                max_length = max(max_length, text_length)

                if text_length < 100:
                    short_chunks += 1

    # 4. 输出统计结果
    print("\n[1] 知识库规模")

    print("处理论文数量:", len(dataset))
    print("无有效正文的论文:", empty_papers)
    print("生成 Chunk 总数:", total_chunks)

    print("\n[2] 分块统计")

    if total_chunks > 0:
        avg_length = total_chars / total_chunks
        short_ratio = short_chunks / total_chunks

        print("平均 Chunk 长度:", round(avg_length, 2))
        print("最大 Chunk 长度:", max_length)
        print("短文本 Chunk 数量:", short_chunks)
        print("短文本比例:", f"{short_ratio:.2%}")

    print("\n[3] 文件保存")

    print("输出文件:", output_path)

    print("\n" + "=" * 60)
    print("QASPER 训练集知识库构建完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
