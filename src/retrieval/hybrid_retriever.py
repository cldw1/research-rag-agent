import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.full_dense_retriever import FullDenseRetriever


CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "qasper_train_chunks.jsonl"
)


class HybridRetriever:

    def __init__(self, rrf_k=60):

        self.rrf_k = rrf_k

        print("[1] 初始化 BM25 检索器...")
        self.bm25 = BM25Retriever(CORPUS_PATH)

        print("\n[2] 初始化 Dense 检索器...")
        self.dense = FullDenseRetriever()

        print("\n混合检索器初始化完成！")

    @staticmethod
    def normalize_result(item):

        """
        统一两种检索器返回的数据格式
        """

        if not isinstance(item, dict):
            raise TypeError("检索结果必须是字典")

        result = {}

        # 兼容嵌套形式
        for key in ("chunk", "metadata"):

            nested = item.get(key)

            if isinstance(nested, dict):
                result.update(nested)

        result.update(item)

        return result

    def search(self, query, top_k=10, candidate_k=50):

        if not query.strip():
            return []

        # ----------------------------------
        # 1. BM25 关键词检索
        # ----------------------------------

        bm25_results = self.bm25.search(
            query,
            top_k=candidate_k
        )

        # ----------------------------------
        # 2. Dense 语义检索
        # ----------------------------------

        dense_results = self.dense.search(
            query,
            top_k=candidate_k
        )

        # ----------------------------------
        # 3. RRF 排名融合
        # ----------------------------------

        fused = {}

        retrieval_lists = [
            ("bm25", bm25_results),
            ("dense", dense_results)
        ]

        for source, items in retrieval_lists:

            for rank, raw_item in enumerate(items, start=1):

                item = self.normalize_result(raw_item)

                chunk_id = item.get("chunk_id")

                if not chunk_id:
                    raise ValueError(
                        f"{source} 检索结果缺少 chunk_id"
                    )

                if chunk_id not in fused:

                    fused[chunk_id] = {
                        "chunk_id": chunk_id,
                        "paper_id": item.get("paper_id", ""),
                        "title": item.get("title", ""),
                        "section_name": item.get("section_name", ""),
                        "text": item.get("text", ""),
                        "rrf_score": 0.0,
                        "bm25_rank": None,
                        "dense_rank": None
                    }

                record = fused[chunk_id]

                # 补充可能缺失的元数据
                for field in (
                    "paper_id",
                    "title",
                    "section_name",
                    "text"
                ):

                    if not record[field] and item.get(field):
                        record[field] = item[field]

                # RRF 融合公式
                record["rrf_score"] += (
                    1.0 / (self.rrf_k + rank)
                )

                record[f"{source}_rank"] = rank

        # ----------------------------------
        # 4. 根据融合分数重新排序
        # ----------------------------------

        ranked_results = sorted(
            fused.values(),
            key=lambda x: (
                -x["rrf_score"],
                x["chunk_id"]
            )
        )

        final_results = ranked_results[:top_k]

        for rank, item in enumerate(final_results, start=1):
            item["rank"] = rank

        return final_results


def main():

    print("=" * 65)
    print("科研 RAG Agent - BM25 + Dense 混合检索实验")
    print("=" * 65)

    retriever = HybridRetriever(rrf_k=60)

    question = "What is the seed lexicon?"

    target_paper_id = "1909.00694"

    print("\n[3] 开始混合检索...")

    print(f"用户问题：{question}")

    results = retriever.search(
        query=question,
        top_k=10,
        candidate_k=50
    )

    print("\n[4] RRF 融合结果")

    for item in results:

        print("\n" + "-" * 65)

        print(f"排名：{item['rank']}")

        print(f"RRF 分数：{item['rrf_score']:.6f}")

        print(f"BM25 排名：{item['bm25_rank']}")

        print(f"Dense 排名：{item['dense_rank']}")

        print(f"论文 ID：{item['paper_id']}")

        print(f"章节：{item['section_name']}")

        print(f"Chunk ID：{item['chunk_id']}")

        print("\n文本内容：")

        print(
            item["text"].replace("\n", " ")[:250]
        )

    # --------------------------------------
    # 5. 简单检查目标论文
    # --------------------------------------

    print("\n[5] 检查目标论文...")

    matching = [
        item
        for item in results
        if str(item["paper_id"]) == target_paper_id
    ]

    if matching:

        print("目标论文进入 Top-10：是")
        print(f"首次出现排名：{matching[0]['rank']}")

    else:

        print("目标论文进入 Top-10：否")

    print("\n说明：")
    print("本次是单问题功能测试。")
    print("暂不能据此判断混合检索优于单路检索。")

    print("\n" + "=" * 65)
    print("RRF 混合检索实验完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()