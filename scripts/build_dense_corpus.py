import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import tokenize


MODEL_NAME = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 16

CHUNKS_PER_SHARD = 512


def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            block = f.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


def load_valid_chunks(corpus_path):

    """
    使用与BM25一致的文本过滤规则。
    """

    texts = []

    with open(corpus_path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            chunk = json.loads(line)

            text = chunk["text"]

            if tokenize(text):
                texts.append(text)

    return texts


def main():

    print("=" * 60)
    print("科研 RAG Agent - 完整训练集向量化")
    print("=" * 60)

    corpus_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "qasper_train_chunks.jsonl"
    )

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "dense_train_shards"
    )

    cache_dir = (
        PROJECT_ROOT
        / "models"
        / "embedding_cache"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==================================
    # 1. 加载有效Chunk
    # ==================================

    print("\n[1] 读取有效 Chunk...")

    texts = load_valid_chunks(corpus_path)

    total_chunks = len(texts)

    total_shards = math.ceil(
        total_chunks / CHUNKS_PER_SHARD
    )

    print("有效 Chunk 数量:", total_chunks)
    print("每个分片 Chunk 数量:", CHUNKS_PER_SHARD)
    print("预计分片数量:", total_shards)

    # ==================================
    # 2. 创建实验清单
    # ==================================

    print("\n[2] 检查数据一致性...")

    manifest_path = output_dir / "manifest.json"

    manifest = {
        "model": MODEL_NAME,
        "source_file": corpus_path.name,
        "source_sha256": calculate_file_hash(corpus_path),
        "filter": "bm25_tokenize",
        "total_chunks": total_chunks,
        "chunks_per_shard": CHUNKS_PER_SHARD,
        "total_shards": total_shards
    }

    if manifest_path.exists():

        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as f:

            previous_manifest = json.load(f)

        if previous_manifest != manifest:

            raise ValueError(
                "数据或配置发生变化，"
                "不能直接复用旧向量分片。"
            )

        print("检测到已有实验清单，可以继续之前的任务。")

    else:

        existing_shards = list(
            output_dir.glob("part_*.npy")
        )

        if existing_shards:

            raise RuntimeError(
                "发现已有向量分片，但缺少实验清单，"
                "为防止数据错位，已停止运行。"
            )

        with open(
            manifest_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                manifest,
                f,
                ensure_ascii=False,
                indent=2
            )

        print("实验清单已创建。")

    # ==================================
    # 3. 查找未完成的分片
    # ==================================

    pending_shards = []

    completed_shards = 0

    for shard_id in range(total_shards):

        shard_path = (
            output_dir / f"part_{shard_id:05d}.npy"
        )

        start = shard_id * CHUNKS_PER_SHARD

        end = min(
            start + CHUNKS_PER_SHARD,
            total_chunks
        )

        expected_rows = end - start

        if shard_path.exists():

            saved = np.load(
                shard_path,
                mmap_mode="r"
            )

            if saved.shape != (expected_rows, 384):

                raise ValueError(
                    f"分片 {shard_path.name} 形状异常"
                )

            completed_shards += 1

        else:

            pending_shards.append(shard_id)

    print("\n[3] 检查已有进度")

    print("已完成分片:", completed_shards)

    print("待处理分片:", len(pending_shards))

    if not pending_shards:

        print("\n所有分片已经生成完成！")
        return

    # ==================================
    # 4. 加载Embedding模型
    # ==================================

    print("\n[4] 加载 Embedding 模型...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(cache_dir)
    )

    print("模型加载成功！")

    # ==================================
    # 5. 分片生成向量
    # ==================================

    print("\n[5] 开始批量生成向量...")

    start_time = time.perf_counter()

    for number, shard_id in enumerate(
        pending_shards,
        start=1
    ):

        start = shard_id * CHUNKS_PER_SHARD

        end = min(
            start + CHUNKS_PER_SHARD,
            total_chunks
        )

        batch_texts = texts[start:end]

        embeddings = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype(np.float32)

        shard_path = (
            output_dir / f"part_{shard_id:05d}.npy"
        )

        # 先写临时文件，完成后再替换为正式文件
        temp_path = output_dir / (
            f"part_{shard_id:05d}.npy.tmp"
        )

        with open(temp_path, "wb") as f:

            np.save(f, embeddings)

        os.replace(
            temp_path,
            shard_path
        )

        print(
            f"分片 {shard_id + 1}/{total_shards} "
            f"完成 | Chunk {start}-{end - 1}"
        )

        if number % 10 == 0:

            elapsed = time.perf_counter() - start_time

            print(
                f"本次运行已耗时: "
                f"{elapsed / 60:.2f} 分钟"
            )

    # ==================================
    # 6. 最终检查
    # ==================================

    print("\n[6] 检查向量分片...")

    saved_files = sorted(
        output_dir.glob("part_*.npy")
    )

    saved_rows = 0

    for file_path in saved_files:

        array = np.load(
            file_path,
            mmap_mode="r"
        )

        saved_rows += len(array)

    print("保存分片数量:", len(saved_files))
    print("保存向量数量:", saved_rows)

    if saved_rows == total_chunks:

        print("\n完整训练集向量化成功！")

    else:

        print("\n向量数量不一致，请检查。")

    elapsed = time.perf_counter() - start_time

    print(
        "本次运行耗时:",
        f"{elapsed / 60:.2f} 分钟"
    )

    print("\n" + "=" * 60)
    print("完整训练集向量化任务结束")
    print("=" * 60)


if __name__ == "__main__":
    main()