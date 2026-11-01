
import json
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import tokenize


DATA_DIR = PROJECT_ROOT / "data" / "processed"

INDEX_PATH = DATA_DIR / "qasper_validation_30.faiss"
EMBEDDING_PATH = DATA_DIR / "qasper_validation_30_embeddings.npy"
METADATA_PATH = DATA_DIR / "qasper_validation_30_index_chunks.jsonl"

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "qasper_validation_30_questions.json"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RRF_K = 60


# ==========================================
# 2. Validation 检索器
# ==========================================

class ValidationPaperScopedRetriever:

    def __init__(self):

        print("正在加载 Validation 检索数据...")

        missing = [p.name for p in (INDEX_PATH, EMBEDDING_PATH, METADATA_PATH) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "缺少 QASPER Validation 索引文件：" + ", ".join(missing)
                + "。请先运行 scripts/prepare_validation_sample.py，"
                "再运行 scripts/build_validation_faiss.py。"
            )

        # 读取验证集索引
        self.global_index = faiss.read_index(
            str(INDEX_PATH)
        )

        # 读取验证集向量
        self.embeddings = np.load(
            EMBEDDING_PATH
        )

        # 读取 Chunk 元数据
        self.chunks = []

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                if line.strip():
                    self.chunks.append(
                        json.loads(line)
                    )

        # 检查向量与元数据数量
        n = len(self.chunks)

        if not (
            self.global_index.ntotal == n
            and len(self.embeddings) == n
            and self.embeddings.ndim == 2
            and self.global_index.d == self.embeddings.shape[1]
            and n > 0
        ):
            raise ValueError(
                "Validation 向量和元数据数量不一致"
            )

        # 按论文建立行号映射
        self.paper_rows = defaultdict(list)

        for index, chunk in enumerate(self.chunks):

            paper_id = str(chunk["paper_id"])

            self.paper_rows[paper_id].append(
                index
            )

        self.paper_cache = {}

        print("Validation Chunk 数量:", n)

        print(
            "Validation 论文数量:",
            len(self.paper_rows)
        )

        # 只加载现有 Embedding 模型
        print("正在加载 BGE 模型...")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu",
            cache_folder=str(
                PROJECT_ROOT
                / "models"
                / "embedding_cache"
            )
        )

        print("Validation 检索器初始化完成！")

    # ======================================
    # 3. 获得单篇论文的检索数据
    # ======================================

    def get_paper_data(self, paper_id):

        paper_id = str(paper_id)

        if paper_id in self.paper_cache:
            return self.paper_cache[paper_id]

        rows = self.paper_rows.get(paper_id)

        if not rows:
            raise ValueError(
                f"Validation 中不存在论文：{paper_id}"
            )

        chunks = [
            self.chunks[i]
            for i in rows
        ]

        vectors = np.ascontiguousarray(
            self.embeddings[rows],
            dtype=np.float32
        )

        # 为这篇论文创建局部 FAISS 索引
        local_index = faiss.IndexFlatIP(
            vectors.shape[1]
        )

        local_index.add(vectors)

        # 为这篇论文创建局部 BM25
        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        bm25 = BM25Okapi(
            tokenized_corpus
        )

        paper_data = {
            "chunks": chunks,
            "vectors": vectors,
            "index": local_index,
            "bm25": bm25
        }

        self.paper_cache[paper_id] = paper_data

        return paper_data

    # ======================================
    # 4. 混合检索
    # ======================================

    def search(
        self,
        query,
        paper_id,
        top_k=5,
        candidate_k=50
    ):

        paper_data = self.get_paper_data(
            paper_id
        )

        chunks = paper_data["chunks"]
        n = len(chunks)

        candidate_k = min(
            n,
            max(candidate_k, top_k)
        )

        # ----------------------------------
        # BM25 检索
        # ----------------------------------

        bm25_scores = paper_data["bm25"].get_scores(
            tokenize(query)
        )

        bm25_order = np.argsort(
            -np.asarray(bm25_scores),
            kind="stable"
        )[:candidate_k]

        bm25_ranks = {
            int(index): rank
            for rank, index in enumerate(
                bm25_order,
                start=1
            )
        }

        # ----------------------------------
        # Dense 检索
        # ----------------------------------

        instruction = (
            "Represent this sentence for searching "
            "relevant passages: "
        )

        query_vector = self.model.encode(
            [instruction + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        query_vector = np.ascontiguousarray(
            query_vector,
            dtype=np.float32
        )

        scores, indices = (
            paper_data["index"].search(
                query_vector,
                candidate_k
            )
        )

        dense_ranks = {
            int(index): rank
            for rank, index in enumerate(
                indices[0],
                start=1
            )
            if index >= 0
        }

        # ----------------------------------
        # RRF 排名融合
        # ----------------------------------

        candidate_ids = (
            set(bm25_ranks)
            | set(dense_ranks)
        )

        fused = []

        for index in candidate_ids:

            bm25_rank = bm25_ranks.get(index)
            dense_rank = dense_ranks.get(index)

            rrf_score = 0.0

            if bm25_rank is not None:
                rrf_score += (
                    1.0 / (RRF_K + bm25_rank)
                )

            if dense_rank is not None:
                rrf_score += (
                    1.0 / (RRF_K + dense_rank)
                )

            fused.append({
                "index": index,
                "rrf_score": rrf_score,
                "bm25_rank": bm25_rank,
                "dense_rank": dense_rank
            })

        fused.sort(
            key=lambda item: (
                -item["rrf_score"],
                item["index"]
            )
        )

        # ----------------------------------
        # 整理结果
        # ----------------------------------

        results = []

        for rank, item in enumerate(
            fused[:top_k],
            start=1
        ):

            chunk = chunks[item["index"]]

            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "title": chunk.get("title", ""),
                "section_name": chunk.get(
                    "section_name", ""
                ),
                "text": chunk["text"],
                "rrf_score": item["rrf_score"],
                "bm25_rank": item["bm25_rank"],
                "dense_rank": item["dense_rank"]
            })

        return results


# ==========================================
# 5. 运行测试
# ==========================================

def main():

    print("=" * 65)
    print("QASPER Validation Paper-Scoped 检索测试")
    print("=" * 65)

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    # 选择固定验证集中的第一道问题
    sample = data["questions"][0]

    paper_id = sample["paper_id"]
    question = sample["question"]

    print("\n测试论文 ID:", paper_id)
    print("测试问题:", question)

    retriever = ValidationPaperScopedRetriever()

    paper_data = retriever.get_paper_data(
        paper_id
    )

    print(
        "\n当前论文 Chunk 数量:",
        len(paper_data["chunks"])
    )

    print("\n执行 Hybrid Top-5 检索...")

    results = retriever.search(
        query=question,
        paper_id=paper_id,
        top_k=5
    )

    print("\n" + "=" * 65)
    print("Validation 检索结果")
    print("=" * 65)

    for item in results:

        if str(item["paper_id"]) != str(paper_id):
            raise ValueError(
                "发现其他论文的 Chunk，检索范围错误"
            )

        print(
            f"\n排名 {item['rank']}"
        )

        print(
            "RRF 分数:",
            round(item["rrf_score"], 6)
        )

        print(
            "BM25 排名:",
            item["bm25_rank"]
        )

        print(
            "Dense 排名:",
            item["dense_rank"]
        )

        print(
            "Chunk ID:",
            item["chunk_id"]
        )

        print(
            "章节:",
            item["section_name"]
        )

        print(
            "正文预览:",
            item["text"][:180]
        )

    assert len(results) > 0

    print("\n" + "=" * 65)
    print("Validation Paper-Scoped 检索测试通过！")
    print("=" * 65)

    print("\n说明：")
    print("本次仅使用验证集论文与验证集向量。")
    print("没有读取训练集 FAISS 索引。")
    print("人工标准证据未参与检索。")


if __name__ == "__main__":
    main()
