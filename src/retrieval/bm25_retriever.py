
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


# ==========================================
# 项目二：科研 RAG Agent
# P2-03：BM25 关键词检索 Baseline
# ==========================================


def tokenize(text):
    """
    对英文文本进行简单分词。

    主要操作：
    1. 统一转换为小写
    2. 提取英文单词和数字
    3. 去除大部分标点符号
    """

    text = text.lower()

    tokens = re.findall(
        r"[a-z0-9]+(?:[-'][a-z0-9]+)*",
        text
    )

    return tokens


class BM25Retriever:
    """
    BM25 关键词检索器。
    """

    def __init__(self, corpus_path):

        self.chunks = []
        tokenized_corpus = []

        print("正在读取论文知识库...")

        with open(corpus_path, "r", encoding="utf-8") as f:

            for line in f:
                if not line.strip():
                    continue

                chunk = json.loads(line)
                tokens = tokenize(chunk["text"])

                if not tokens:
                    continue

                self.chunks.append(chunk)
                tokenized_corpus.append(tokens)

        if not self.chunks:
            raise ValueError("论文知识库为空")

        print("有效 Chunk 数量:", len(self.chunks))

        # 构建 BM25 索引
        print("正在构建 BM25 索引...")

        self.bm25 = BM25Okapi(tokenized_corpus)

        print("BM25 索引构建完成！")

    def search(self, query, top_k=5):
        """
        根据用户问题检索最相关的文本块。
        """

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # 计算每个文本块的 BM25 分数
        scores = self.bm25.get_scores(query_tokens)

        # 按分数从高到低排序
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:

            score = float(scores[index])

            if score <= 0:
                continue

            chunk = self.chunks[index]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "section_name": chunk["section_name"],
                "score": score,
                "text": chunk["text"]
            })

        return results


def main():

    project_root = Path(__file__).resolve().parents[2]

    corpus_path = (
        project_root
        / "data"
        / "processed"
        / "qasper_train_chunks.jsonl"
    )

    print("=" * 60)
    print("科研 RAG Agent - BM25 检索实验")
    print("=" * 60)

    # 1. 初始化检索器
    retriever = BM25Retriever(corpus_path)

    # 2. 设置测试问题
    query = "What is the seed lexicon?"

    print("\n用户问题:")
    print(query)

    # 3. 执行检索
    results = retriever.search(query, top_k=5)

    print("\n检索结果数量:", len(results))

    # 4. 展示检索结果
    for rank, result in enumerate(results, start=1):

        print("\n" + "-" * 60)

        print(f"排名: {rank}")
        print(f"BM25 分数: {result['score']:.4f}")
        print(f"论文 ID: {result['paper_id']}")
        print(f"论文标题: {result['title']}")
        print(f"章节: {result['section_name']}")
        print(f"Chunk ID: {result['chunk_id']}")

        print("\n文本内容:")
        print(result["text"][:300])

    print("\n" + "=" * 60)
    print("BM25 检索实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
