import json
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


def main():

    print("=" * 60)
    print("科研 RAG Agent - FAISS 向量检索实验")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[2]

    # ==================================
    # 1. 设置文件路径
    # ==================================

    chunks_path = (
        project_root
        / "data"
        / "processed"
        / "qasper_sample_chunks.jsonl"
    )

    embeddings_path = (
        project_root
        / "data"
        / "processed"
        / "qasper_sample_embeddings.npy"
    )

    index_path = (
        project_root
        / "data"
        / "processed"
        / "qasper_sample.faiss"
    )

    cache_dir = (
        project_root
        / "models"
        / "embedding_cache"
    )

    # ==================================
    # 2. 读取论文Chunk
    # ==================================

    print("\n[1] 读取论文 Chunk...")

    chunks = []

    with open(chunks_path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                chunks.append(json.loads(line))

    print("Chunk 数量:", len(chunks))

    # ==================================
    # 3. 读取之前保存的向量
    # ==================================

    print("\n[2] 读取 Embedding 向量...")

    embeddings = np.load(embeddings_path)

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32
    )

    print("向量矩阵形状:", embeddings.shape)

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Chunk 数量与向量数量不一致，"
            "请重新运行 dense_retriever.py"
        )

    # ==================================
    # 4. 构建 FAISS 索引
    # ==================================

    print("\n[3] 构建 FAISS 索引...")

    dimension = embeddings.shape[1]

    # 再次确保向量归一化
    faiss.normalize_L2(embeddings)

    # 创建内积检索索引
    index = faiss.IndexFlatIP(dimension)

    # 将向量加入索引
    index.add(embeddings)

    print("向量维度:", dimension)
    print("索引中向量数量:", index.ntotal)

    # ==================================
    # 5. 保存并重新加载索引
    # ==================================

    print("\n[4] 保存 FAISS 索引...")

    faiss.write_index(
        index,
        str(index_path)
    )

    print("FAISS 索引保存成功！")
    print("保存位置:", index_path)

    index = faiss.read_index(str(index_path))

    print("重新加载索引成功！")
    print("索引向量数量:", index.ntotal)

    # ==================================
    # 6. 加载 Embedding 模型
    # ==================================

    print("\n[5] 加载 Embedding 模型...")

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        device="cpu",
        cache_folder=str(cache_dir)
    )

    # ==================================
    # 7. 生成查询向量
    # ==================================

    query = "What is the seed lexicon?"

    instruction = (
        "Represent this sentence for searching relevant passages: "
    )

    query_embedding = model.encode(
        [instruction + query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32
    )

    faiss.normalize_L2(query_embedding)

    # ==================================
    # 8. FAISS 检索
    # ==================================

    print("\n[6] 执行 FAISS 检索...")

    top_k = 5

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    print("用户问题:", query)

    for rank, (score, chunk_index) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if chunk_index < 0:
            continue

        chunk = chunks[int(chunk_index)]

        print("\n" + "-" * 60)

        print("排名:", rank)

        print(
            "相似度:",
            f"{score:.4f}"
        )

        print("论文 ID:", chunk["paper_id"])

        print("章节:", chunk["section_name"])

        print("Chunk ID:", chunk["chunk_id"])

        print("\n文本内容:")

        print(chunk["text"][:300])

    # ==================================
    # 9. 与 NumPy 结果进行一致性检查
    # ==================================

    print("\n[7] 检查 FAISS 与 NumPy 结果是否一致...")

    numpy_scores = (
        embeddings @ query_embedding[0]
    )

    numpy_top_indices = np.argsort(
        numpy_scores
    )[::-1][:top_k]

    same_ranking = np.array_equal(
        indices[0],
        numpy_top_indices
    )

    same_scores = np.allclose(
        scores[0],
        numpy_scores[numpy_top_indices],
        atol=1e-5
    )

    print("Top-5 排名一致:", same_ranking)
    print("Top-5 分数一致:", same_scores)

    print("\n" + "=" * 60)
    print("FAISS 向量检索实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()