import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:

    def __init__(self, model_name, cache_dir):

        print("正在加载 Embedding 模型...")

        self.model = SentenceTransformer(
            model_name,
            device="cpu",
            cache_folder=str(cache_dir)
        )

        self.chunks = []
        self.embeddings = None

        print("Embedding 模型加载成功！")

    def build_index(self, chunks):

        # 过滤空文本
        self.chunks = [
            chunk
            for chunk in chunks
            if chunk.get("text", "").strip()
        ]

        if not self.chunks:
            raise ValueError("没有可用于检索的文本")

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        print("\n正在生成论文 Chunk 向量...")

        self.embeddings = self.model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype(np.float32)

        print("Chunk 向量生成完成！")
        print("向量矩阵形状:", self.embeddings.shape)

    def search(self, query, top_k=5):

        if self.embeddings is None:
            raise ValueError("请先构建向量索引")

        # 为查询添加检索指令
        instruction = (
            "Represent this sentence for searching relevant passages: "
        )

        query_embedding = self.model.encode(
            [instruction + query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        # 计算余弦相似度
        scores = self.embeddings @ query_embedding

        # 获取得分最高的Chunk
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:

            chunk = self.chunks[int(index)]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "section_name": chunk["section_name"],
                "score": float(scores[index]),
                "text": chunk["text"]
            })

        return results


def main():

    print("=" * 60)
    print("科研 RAG Agent - 真实论文 Dense Retrieval")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[2]

    # 1. 数据路径
    corpus_path = (
        project_root
        / "data"
        / "processed"
        / "qasper_sample_chunks.jsonl"
    )

    cache_dir = (
        project_root
        / "models"
        / "embedding_cache"
    )

    # 2. 读取论文Chunk
    print("\n[1] 读取论文 Chunk...")

    chunks = []

    with open(corpus_path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                chunks.append(json.loads(line))

    print("读取 Chunk 数量:", len(chunks))

    # 3. 初始化Dense检索器
    print("\n[2] 初始化 Dense Retriever...")

    retriever = DenseRetriever(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir=cache_dir
    )

    # 4. 构建向量索引
    print("\n[3] 构建向量索引...")

    retriever.build_index(chunks)

    # 5. 保存向量
    vector_path = (
        project_root
        / "data"
        / "processed"
        / "qasper_sample_embeddings.npy"
    )

    np.save(
        vector_path,
        retriever.embeddings
    )

    print("\n向量保存成功！")
    print("保存路径:", vector_path)

    # 6. 执行检索
    print("\n[4] 执行语义检索...")

    query = "What is the seed lexicon?"

    print("用户问题:", query)

    results = retriever.search(
        query,
        top_k=5
    )

    # 7. 输出结果
    print("\n[5] Top-5 检索结果")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 60)

        print("排名:", rank)

        print(
            "语义相似度:",
            f"{result['score']:.4f}"
        )

        print("论文 ID:", result["paper_id"])

        print("章节:", result["section_name"])

        print("Chunk ID:", result["chunk_id"])

        print("\n文本内容:")

        print(result["text"][:400])

    print("\n" + "=" * 60)
    print("真实论文 Dense Retrieval 实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()