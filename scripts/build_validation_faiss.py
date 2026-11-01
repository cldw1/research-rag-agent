
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 设置项目根目录
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import tokenize


# ==========================================
# 2. 实验配置
# ==========================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"

MODEL_CACHE = (
    PROJECT_ROOT / "models" / "embedding_cache"
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "qasper_validation_30_chunks.jsonl"
)

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "qasper_validation_30.faiss"
)

EMBEDDING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "qasper_validation_30_embeddings.npy"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "qasper_validation_30_index_chunks.jsonl"
)

BATCH_SIZE = 64


# ==========================================
# 3. 读取验证集 Chunk
# ==========================================

def load_chunks():

    chunks = []

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            chunk = json.loads(line)

            text = chunk.get("text", "")

            # 与训练集索引保持相同的过滤规则
            if not tokenize(text):
                continue

            chunks.append(chunk)

    if not chunks:
        raise ValueError("没有可用的验证集 Chunk")

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("发现重复的 Chunk ID")

    return chunks


# ==========================================
# 4. 生成文本向量
# ==========================================

def build_embeddings(chunks):

    print("\n[2] 加载 BGE 模型...")

    MODEL_CACHE.mkdir(
        parents=True,
        exist_ok=True
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(MODEL_CACHE)
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\n[3] 生成验证集 Embedding...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    print("Embedding 形状:", embeddings.shape)

    return embeddings


# ==========================================
# 5. 构建 FAISS 索引
# ==========================================

def build_faiss_index(embeddings):

    print("\n[4] 构建 FAISS 索引...")

    dimension = embeddings.shape[1]

    # 归一化向量 + 内积 = 余弦相似度
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("索引向量维度:", index.d)
    print("索引向量数量:", index.ntotal)

    return index


# ==========================================
# 6. 保存索引和元数据
# ==========================================

def save_files(chunks, embeddings, index):

    print("\n[5] 保存验证集索引...")

    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 保存 FAISS 索引
    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    # 保存 NumPy 向量，方便后续实验
    np.save(
        EMBEDDING_PATH,
        embeddings
    )

    # 保存与向量位置一一对应的 Chunk
    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                ) + "\n"
            )


# ==========================================
# 7. 检查保存结果
# ==========================================

def verify_index(expected_count):

    print("\n[6] 重新加载 FAISS 索引进行检查...")

    loaded_index = faiss.read_index(
        str(INDEX_PATH)
    )

    loaded_embeddings = np.load(
        EMBEDDING_PATH
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata_count = sum(
            1
            for line in f
            if line.strip()
        )

    print("FAISS 向量数:", loaded_index.ntotal)
    print("NumPy 向量数:", len(loaded_embeddings))
    print("Chunk 元数据数:", metadata_count)

    assert loaded_index.ntotal == expected_count
    assert len(loaded_embeddings) == expected_count
    assert metadata_count == expected_count

    assert loaded_index.d == loaded_embeddings.shape[1]

    # 用第一条向量检索自身
    scores, indices = loaded_index.search(
        loaded_embeddings[:1],
        1
    )

    print("\n索引自检：")
    print("最相似向量 ID:", int(indices[0][0]))
    print("相似度:", round(float(scores[0][0]), 4))

    assert float(scores[0][0]) > 0.99

    print("\n索引检查通过！")


# ==========================================
# 8. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("QASPER Validation FAISS 索引构建")
    print("=" * 65)

    print("\n[1] 读取 Validation Chunk...")

    chunks = load_chunks()

    print("可索引 Chunk 数量:", len(chunks))

    paper_ids = {
        chunk["paper_id"]
        for chunk in chunks
    }

    print("覆盖论文数量:", len(paper_ids))

    embeddings = build_embeddings(chunks)

    index = build_faiss_index(embeddings)

    save_files(
        chunks,
        embeddings,
        index
    )

    verify_index(len(chunks))

    print("\n" + "=" * 65)
    print("Validation 索引构建完成！")
    print("=" * 65)

    print("\nFAISS 索引:")
    print(INDEX_PATH)

    print("\nEmbedding 文件:")
    print(EMBEDDING_PATH)

    print("\nChunk 元数据:")
    print(METADATA_PATH)

    print("\n注意：")
    print("该索引仅包含验证集样本论文。")
    print("不会覆盖原来的训练集 FAISS 索引。")
    print("后续验证检索必须使用这里对应的索引和元数据。")


if __name__ == "__main__":
    main()
