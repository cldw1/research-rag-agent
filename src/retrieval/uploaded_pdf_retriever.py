
import hashlib
import re
from pathlib import Path

import faiss
import numpy as np
import pymupdf
from rank_bm25 import BM25Okapi

from src.retrieval.bm25_retriever import tokenize


# ==========================================
# 1. 配置
# ==========================================

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PAGES = 80
MAX_CHUNKS = 1500

CHUNK_SIZE = 1200
RRF_K = 60

QUERY_PREFIX = (
    "Represent this sentence for searching "
    "relevant passages: "
)


# ==========================================
# 2. 将长段落切分为 Chunk
# ==========================================

def split_text(text, max_chars=CHUNK_SIZE):

    text = re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()

    pieces = []

    while len(text) > max_chars:

        # 尽量在空格处断开
        cut = text.rfind(
            " ",
            0,
            max_chars + 1
        )

        if cut < max_chars // 2:
            cut = max_chars

        piece = text[:cut].strip()

        if piece:
            pieces.append(piece)

        text = text[cut:].strip()

    if text:
        pieces.append(text)

    return pieces


# ==========================================
# 3. 上传 PDF 的检索器
# ==========================================

class UploadedPDFRetriever:

    def __init__(
        self,
        pdf_bytes,
        filename,
        model
    ):

        if not isinstance(
            pdf_bytes, bytes
        ):
            raise ValueError(
                "PDF 内容必须是 bytes"
            )

        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise ValueError(
                "PDF 文件不能超过 20 MB"
            )

        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError(
                "文件不是有效的 PDF 格式"
            )

        self.model = model

        self.title = Path(
            filename
        ).stem

        # 根据 PDF 内容生成稳定的论文 ID
        digest = hashlib.sha256(
            pdf_bytes
        ).hexdigest()[:12]

        self.paper_id = (
            "upload_" + digest
        )

        print("\n正在解析上传的 PDF...")

        self.chunks = []
        self.page_count = 0

        # ----------------------------------
        # 4. 读取 PDF 页面
        # ----------------------------------

        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as doc:

            if doc.needs_pass:
                raise ValueError(
                    "暂不支持需要密码的 PDF"
                )

            self.page_count = len(doc)

            if self.page_count > MAX_PAGES:
                raise ValueError(
                    f"当前最多支持 {MAX_PAGES} 页 PDF"
                )

            for page_index, page in enumerate(doc):

                # sort=True 尽量按阅读顺序排列
                blocks = page.get_text(
                    "blocks",
                    sort=True
                )

                paragraph_index = 0

                for block in blocks:

                    # 只处理文字块，跳过图片块
                    if len(block) < 7 or block[6] != 0:
                        continue

                    text = str(block[4]).strip()

                    pieces = split_text(
                        text,
                        max_chars=CHUNK_SIZE
                    )

                    for piece_index, piece in enumerate(
                        pieces
                    ):

                        # 保持与原有 BM25 索引一致
                        if not tokenize(piece):
                            continue

                        chunk_id = (
                            f"{self.paper_id}"
                            f"_s{page_index:03d}"
                            f"_p{paragraph_index:03d}"
                            f"_c{piece_index:03d}"
                        )

                        self.chunks.append({
                            "chunk_id": chunk_id,
                            "paper_id": self.paper_id,
                            "title": self.title,
                            "section_name": (
                                f"Page {page_index + 1}"
                            ),
                            "section_index": page_index,
                            "paragraph_index": paragraph_index,
                            "piece_index": piece_index,
                            "page_number": page_index + 1,
                            "text": piece,
                            "char_count": len(piece)
                        })

                        if len(self.chunks) > MAX_CHUNKS:
                            raise ValueError(
                                "论文分块超过 1500 个，"
                                "当前版本暂不支持这么大的文件"
                            )

                    paragraph_index += 1

        if not self.chunks:
            raise ValueError(
                "没有提取到可用文本。"
                "可能是扫描版 PDF，"
                "当前版本暂不支持 OCR。"
            )

        print("PDF 解析完成")
        print("论文页数:", self.page_count)
        print("Chunk 数量:", len(self.chunks))

        # ----------------------------------
        # 5. 构建 BM25
        # ----------------------------------

        print("\n正在构建 BM25 索引...")

        tokenized = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized
        )

        # ----------------------------------
        # 6. 生成 BGE 向量
        # ----------------------------------

        print("\n正在生成 BGE 向量...")

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        self.vectors = np.ascontiguousarray(
            vectors,
            dtype=np.float32
        )

        # ----------------------------------
        # 7. 构建 FAISS
        # ----------------------------------

        dimension = self.vectors.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            self.vectors
        )

        # 兼容原有检索器接口
        self.paper_rows = {
            self.paper_id: list(
                range(len(self.chunks))
            )
        }

        print("\n上传论文索引构建成功！")
        print("向量维度:", dimension)
        print("向量数量:", self.index.ntotal)

    # ======================================
    # 8. 获取指定论文数据
    # ======================================

    def get_paper_data(self, paper_id):

        if str(paper_id) != self.paper_id:
            raise ValueError(
                f"不存在论文：{paper_id}"
            )

        return {
            "chunks": self.chunks,
            "bm25": self.bm25,
            "vectors": self.vectors,
            "index": self.index
        }

    # ======================================
    # 9. Hybrid 检索
    # ======================================

    def search(
        self,
        query,
        paper_id,
        top_k=5,
        candidate_k=50
    ):

        self.get_paper_data(
            paper_id
        )

        if not isinstance(query, str):
            raise ValueError(
                "检索问题必须是字符串"
            )

        if not query.strip():
            raise ValueError(
                "检索问题不能为空"
            )

        if top_k < 1:
            raise ValueError(
                "top_k 必须大于 0"
            )

        n = len(self.chunks)

        candidate_k = min(
            n,
            max(candidate_k, top_k)
        )

        # ----------------------------------
        # BM25 排名
        # ----------------------------------

        bm25_scores = np.asarray(
            self.bm25.get_scores(
                tokenize(query)
            )
        )

        bm25_order = np.argsort(
            -bm25_scores,
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
        # Dense 排名
        # ----------------------------------

        query_vector = self.model.encode(
            [QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        query_vector = np.ascontiguousarray(
            query_vector,
            dtype=np.float32
        )

        _, indices = self.index.search(
            query_vector,
            candidate_k
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

        merged = []

        for index in candidate_ids:

            bm25_rank = bm25_ranks.get(
                index
            )

            dense_rank = dense_ranks.get(
                index
            )

            score = 0.0

            if bm25_rank is not None:
                score += (
                    1.0 / (RRF_K + bm25_rank)
                )

            if dense_rank is not None:
                score += (
                    1.0 / (RRF_K + dense_rank)
                )

            merged.append({
                "index": index,
                "rrf_score": score,
                "bm25_rank": bm25_rank,
                "dense_rank": dense_rank
            })

        merged.sort(
            key=lambda item: (
                -item["rrf_score"],
                item["index"]
            )
        )

        # ----------------------------------
        # 返回检索结果
        # ----------------------------------

        results = []

        for rank, item in enumerate(
            merged[:top_k],
            start=1
        ):

            chunk = self.chunks[
                item["index"]
            ]

            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "section_name": chunk["section_name"],
                "section_index": chunk["section_index"],
                "paragraph_index": chunk["paragraph_index"],
                "piece_index": chunk["piece_index"],
                "page_number": chunk["page_number"],
                "text": chunk["text"],
                "rrf_score": item["rrf_score"],
                "bm25_rank": item["bm25_rank"],
                "dense_rank": item["dense_rank"]
            })

        return results
