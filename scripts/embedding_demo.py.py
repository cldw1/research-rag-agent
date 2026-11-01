import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def main():

    print("=" * 60)
    print("科研 RAG Agent - Embedding 语义检索实验")
    print("=" * 60)

    # 1. 设置模型缓存位置
    project_root = Path(__file__).resolve().parents[1]

    cache_dir = (
        project_root
        / "models"
        / "embedding_cache"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 2. 加载 Embedding 模型
    print("\n[1] 正在加载 Embedding 模型...")
    print("首次运行需要下载模型，请耐心等待。")

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        device="cpu",
        cache_folder=str(cache_dir)
    )

    print("模型加载成功！")
    print("运行设备: CPU")

    # 3. 准备测试问题
    query = "What is the seed lexicon?"

    # 4. 准备测试文本
    documents = [
        "The seed lexicon consists of positive and negative predicates.",

        "A seed lexicon is a collection of positive and negative predicates.",

        "The paper evaluates a neural network on several public datasets.",

        "The weather is sunny and the sky is blue."
    ]

    print("\n[2] 测试问题:")
    print(query)

    print("\n测试文本数量:", len(documents))

    # 5. 将问题转换成向量
    print("\n[3] 生成问题向量...")

    query_instruction = (
        "Represent this sentence for searching relevant passages: "
    )

    query_embedding = model.encode(
        [query_instruction + query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    print("问题向量形状:", query_embedding.shape)

    # 6. 将文本转换成向量
    print("\n[4] 生成文本向量...")

    document_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    print("文本向量形状:", document_embeddings.shape)

    # 7. 计算相似度
    print("\n[5] 计算语义相似度...")

    # 向量已经归一化
    # 此时点积等于余弦相似度
    scores = document_embeddings @ query_embedding[0]

    # 按相似度从高到低排序
    sorted_indices = np.argsort(scores)[::-1]

    # 8. 输出检索结果
    print("\n[6] 语义检索结果")

    for rank, index in enumerate(
        sorted_indices,
        start=1
    ):

        print("\n" + "-" * 60)

        print("排名:", rank)

        print(
            "相似度:",
            f"{scores[index]:.4f}"
        )

        print("文本:")
        print(documents[index])

    print("\n" + "=" * 60)
    print("Embedding 语义检索实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()