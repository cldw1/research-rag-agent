import hashlib
import json
import os
from pathlib import Path

import faiss
import numpy as np


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SHARD_DIR = PROCESSED_DIR / "dense_train_shards"

MANIFEST_PATH = SHARD_DIR / "manifest.json"

CORPUS_PATH = PROCESSED_DIR / "qasper_train_chunks.jsonl"

INDEX_PATH = PROCESSED_DIR / "qasper_train.faiss"

# 根据上一阶段的实际结果
EXPECTED_VECTORS = 48700
EXPECTED_SHARDS = 96
VECTOR_DIM = 384


# ==========================================
# 2. 计算文件 SHA256
# ==========================================

def calculate_sha256(file_path):
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            data = f.read(1024 * 1024)

            if not data:
                break

            hasher.update(data)

    return hasher.hexdigest()


# ==========================================
# 3. 构建 FAISS 索引
# ==========================================

def main():

    print("=" * 65)
    print("科研 RAG Agent - 完整训练集 FAISS 索引构建")
    print("=" * 65)

    # --------------------------------------
    # [1] 读取向量分片清单
    # --------------------------------------

    print("\n[1] 读取向量分片清单...")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"找不到 manifest.json：{MANIFEST_PATH}"
        )

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("分片清单读取成功！")

    # 如果清单记录了原始数据指纹，验证数据未改变
    source_sha256 = manifest.get("source_sha256")

    if source_sha256 is not None:

        if not CORPUS_PATH.exists():
            raise FileNotFoundError(
                f"找不到原始 Chunk 文件：{CORPUS_PATH}"
            )

        current_sha256 = calculate_sha256(CORPUS_PATH)

        if current_sha256 != source_sha256:
            raise ValueError(
                "原始 Chunk 文件发生变化，"
                "与已保存向量不一致！"
            )

        print("原始 Chunk 文件一致性检查通过！")

    # --------------------------------------
    # [2] 检查全部分片
    # --------------------------------------

    print("\n[2] 检查向量分片...")

    shard_files = sorted(SHARD_DIR.glob("part_*.npy"))

    print(f"发现分片数量：{len(shard_files)}")

    if len(shard_files) != EXPECTED_SHARDS:
        raise ValueError(
            f"分片数量错误："
            f"预期 {EXPECTED_SHARDS}，"
            f"实际 {len(shard_files)}"
        )

    # --------------------------------------
    # [3] 创建 FAISS 索引
    # --------------------------------------

    print("\n[3] 创建 FAISS 索引...")

    index = faiss.IndexFlatIP(VECTOR_DIM)

    print("索引类型：IndexFlatIP")
    print(f"向量维度：{VECTOR_DIM}")

    # --------------------------------------
    # [4] 逐个加载分片
    # --------------------------------------

    print("\n[4] 合并全部向量分片...")

    total_added = 0

    for i in range(EXPECTED_SHARDS):

        shard_path = SHARD_DIR / f"part_{i:05d}.npy"

        if not shard_path.exists():
            raise FileNotFoundError(
                f"缺少向量分片：{shard_path.name}"
            )

        vectors = np.load(
            shard_path,
            allow_pickle=False
        )

        if vectors.ndim != 2 or vectors.shape[1] != VECTOR_DIM:
            raise ValueError(
                f"向量维度错误：{shard_path.name}，"
                f"实际形状 {vectors.shape}"
            )

        vectors = np.ascontiguousarray(
            vectors,
            dtype=np.float32
        )

        if not np.isfinite(vectors).all():
            raise ValueError(
                f"发现 NaN 或 Inf：{shard_path.name}"
            )

        # 归一化后，内积对应余弦相似度
        faiss.normalize_L2(vectors)

        # 添加到 FAISS 索引
        index.add(vectors)

        total_added += len(vectors)

        if (i + 1) % 10 == 0 or i + 1 == EXPECTED_SHARDS:

            print(
                f"已合并 {i + 1}/{EXPECTED_SHARDS} 个分片"
                f" | 累计向量：{total_added}"
            )

    # --------------------------------------
    # [5] 验证索引规模
    # --------------------------------------

    print("\n[5] 验证索引...")

    print(f"索引中的向量数量：{index.ntotal}")
    print(f"索引向量维度：{index.d}")

    if index.ntotal != EXPECTED_VECTORS:
        raise ValueError(
            f"向量数量不正确："
            f"预期 {EXPECTED_VECTORS}，"
            f"实际 {index.ntotal}"
        )

    print("索引规模验证通过！")

    # --------------------------------------
    # [6] 保存索引
    # --------------------------------------

    print("\n[6] 保存完整 FAISS 索引...")

    temporary_path = INDEX_PATH.with_suffix(".faiss.tmp")

    faiss.write_index(index, str(temporary_path))

    # 重新读取，验证文件是否完整
    loaded_index = faiss.read_index(str(temporary_path))

    if (
        loaded_index.ntotal != EXPECTED_VECTORS
        or loaded_index.d != VECTOR_DIM
    ):
        raise ValueError("保存后的 FAISS 索引验证失败！")

    # 验证成功后替换为正式文件
    os.replace(temporary_path, INDEX_PATH)

    file_size_mb = INDEX_PATH.stat().st_size / (1024 ** 2)

    print(f"保存位置：{INDEX_PATH}")
    print(f"文件大小：{file_size_mb:.2f} MiB")

    # --------------------------------------
    # [7] 最终结果
    # --------------------------------------

    print("\n" + "=" * 65)
    print("完整训练集 FAISS 索引构建成功！")
    print(f"有效向量数量：{index.ntotal}")
    print(f"向量维度：{index.d}")
    print("索引类型：IndexFlatIP")
    print("=" * 65)


if __name__ == "__main__":
    main()