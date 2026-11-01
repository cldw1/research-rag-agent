
import hashlib
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pymupdf
import streamlit as st

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.bm25_retriever import tokenize

from src.retrieval.uploaded_pdf_retriever import (
    UploadedPDFRetriever
)

from src.generation.strict_context_builder import (
    build_strict_context
)


# ==========================================
# 2. 实验配置
# ==========================================

CHUNK_SIZE = 1200
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PAGES = 80
MAX_CHUNKS = 1500

QUESTIONS = [
    "What is Hamiltonian Monte Carlo and why is it useful?",
    "How does the leapfrog integrator work in HMC?",
    "How are the step size and trajectory length tuned in HMC?",
    "How do mass matrices and correlations affect HMC?",
    "How does HMC handle multimodal distributions and tempering?"
]


# ==========================================
# 3. 加载相同的 Embedding 模型
# ==========================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        device="cpu",
        cache_folder=str(
            PROJECT_ROOT
            / "models"
            / "embedding_cache"
        )
    )


# ==========================================
# 4. 文本清洗
# ==========================================

def normalize_text(text):

    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


def compact(text):
    """
    忽略空白字符。
    用于检查文本内容是否被保留。
    """

    return re.sub(
        r"\s+",
        "",
        str(text)
    )


# ==========================================
# 5. 从 PDF 提取统一的原始文本
# ==========================================

def extract_reference_pages(pdf_bytes):

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError(
            "PDF 不能超过 20 MB"
        )

    pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as doc:

        if doc.needs_pass:
            raise ValueError(
                "暂不支持加密 PDF"
            )

        if len(doc) > MAX_PAGES:
            raise ValueError(
                "PDF 不能超过 80 页"
            )

        for page in doc:

            blocks = page.get_text(
                "blocks",
                sort=True
            )

            texts = []

            for block in blocks:

                if len(block) < 7 or block[6] != 0:
                    continue

                text = normalize_text(
                    block[4]
                )

                if text:
                    texts.append(text)

            pages.append(texts)

    return pages


# ==========================================
# 6. 安全切分超长文本
# ==========================================

def split_long_text(text, limit=CHUNK_SIZE):

    words = normalize_text(text).split()

    pieces = []
    current = ""

    for word in words:

        if len(word) > limit:

            if current:
                pieces.append(current)
                current = ""

            for start in range(
                0,
                len(word),
                limit
            ):
                pieces.append(
                    word[start:start + limit]
                )

            continue

        candidate = (
            f"{current} {word}"
            if current
            else word
        )

        if len(candidate) > limit:

            if current:
                pieces.append(current)

            current = word

        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


# ==========================================
# 7. 新策略：合并相邻文本块
# ==========================================

def build_merged_chunks(
    pages,
    paper_id,
    title
):

    chunks = []

    for page_index, blocks in enumerate(
        pages
    ):

        current = ""
        paragraph_index = 0

        # ----------------------------------
        # 保存一个合并后的 Chunk
        # ----------------------------------

        def flush():

            nonlocal current
            nonlocal paragraph_index

            if current and tokenize(current):

                chunk_id = (
                    f"{paper_id}"
                    f"_s{page_index:03d}"
                    f"_p{paragraph_index:03d}"
                    f"_c000"
                )

                chunks.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "title": title,
                    "section_name": (
                        f"Page {page_index + 1}"
                    ),
                    "section_index": page_index,
                    "paragraph_index": paragraph_index,
                    "piece_index": 0,
                    "page_number": page_index + 1,
                    "text": current,
                    "char_count": len(current)
                })

                paragraph_index += 1

            current = ""

        # ----------------------------------
        # 合并同一页的相邻文本块
        # ----------------------------------

        for block in blocks:

            pieces = split_long_text(
                block,
                limit=CHUNK_SIZE
            )

            for piece in pieces:

                candidate = (
                    f"{current} {piece}"
                    if current
                    else piece
                )

                if len(candidate) <= CHUNK_SIZE:

                    current = candidate

                else:

                    flush()
                    current = piece

        # 每页结束保存剩余内容
        # 不跨页合并
        flush()

    if len(chunks) > MAX_CHUNKS:
        raise ValueError(
            "合并后的 Chunk 超过限制"
        )

    return chunks


# ==========================================
# 8. 合并版检索器
# ==========================================

class MergedPDFRetriever(
    UploadedPDFRetriever
):

    def __init__(
        self,
        pages,
        original,
        model
    ):

        self.model = model
        self.title = original.title
        self.paper_id = original.paper_id
        self.page_count = original.page_count

        # ----------------------------------
        # 使用新分块策略
        # ----------------------------------

        self.chunks = build_merged_chunks(
            pages=pages,
            paper_id=self.paper_id,
            title=self.title
        )

        if not self.chunks:
            raise ValueError(
                "合并分块结果为空"
            )

        # ----------------------------------
        # 建立 BM25
        # ----------------------------------

        tokenized = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized
        )

        # ----------------------------------
        # 生成 BGE 向量
        # ----------------------------------

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        vectors = model.encode(
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
        # 建立 FAISS
        # ----------------------------------

        self.index = faiss.IndexFlatIP(
            self.vectors.shape[1]
        )

        self.index.add(
            self.vectors
        )

        self.paper_rows = {
            self.paper_id: list(
                range(len(self.chunks))
            )
        }


# ==========================================
# 9. 修复：检查文本完整性
# ==========================================

def verify_text_integrity(
    pages,
    original_chunks,
    merged_chunks
):
    """
    以 PDF 原始提取文本作为基准，
    不直接要求两套 Chunk 完全一致。
    """

    # A. PDF 原始文本
    reference_text = compact(
        "".join(
            block
            for page in pages
            for block in page
        )
    )

    # B. 原始检索器的 Chunk 文本
    old_text = compact(
        "".join(
            chunk["text"]
            for chunk in original_chunks
        )
    )

    # C. 合并检索器的 Chunk 文本
    merged_text = compact(
        "".join(
            chunk["text"]
            for chunk in merged_chunks
        )
    )

    result = {
        "reference_chars": len(reference_text),
        "old_chars": len(old_text),
        "merged_chars": len(merged_text),
        "old_matches_reference": (
            old_text == reference_text
        ),
        "merged_matches_reference": (
            merged_text == reference_text
        ),
        "old_matches_merged": (
            old_text == merged_text
        )
    }

    return result


# ==========================================
# 10. 检索一个问题
# ==========================================

def evaluate_one(
    retriever,
    question
):

    results = retriever.search(
        query=question,
        paper_id=retriever.paper_id,
        top_k=5,
        candidate_k=50
    )

    context, sources = build_strict_context(
        results,
        max_chars=8000
    )

    return {
        "context_chars": len(context),
        "source_count": len(sources),
        "included_chunk_ids": [
            source["chunk_id"]
            for source in sources
        ],
        "results": [
            {
                "rank": int(item["rank"]),
                "page": int(item["page_number"]),
                "rrf_score": float(
                    item["rrf_score"]
                ),
                "chunk_id": item["chunk_id"],
                "chars": len(item["text"]),
                "text": item["text"]
            }
            for item in results
        ]
    }


# ==========================================
# 11. Streamlit 页面
# ==========================================

st.set_page_config(
    page_title="PDF Retrieval Ablation",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 PDF 检索效果对比实验")

st.caption(
    "原始文本块分块 vs 相邻文本块合并"
)

st.info(
    "使用相同的 BGE、BM25、RRF 和固定测试问题。"
    "本次只比较检索，不调用 DeepSeek。"
)


# ==========================================
# 12. PDF 上传
# ==========================================

uploaded = st.file_uploader(
    "上传 PDF 论文",
    type=["pdf"]
)

if uploaded is None:
    st.stop()

pdf_bytes = uploaded.getvalue()

pdf_hash = hashlib.sha256(
    pdf_bytes
).hexdigest()


# ==========================================
# 13. 开始检索对比
# ==========================================

if st.button(
    "开始检索对比",
    type="primary"
):

    try:

        with st.spinner(
            "正在解析 PDF 并建立两套索引..."
        ):

            # ----------------------------------
            # A. 加载同一个 BGE 模型
            # ----------------------------------

            model = load_model()

            # ----------------------------------
            # B. 读取 PDF 原始文本
            # ----------------------------------

            pages = extract_reference_pages(
                pdf_bytes
            )

            # ----------------------------------
            # C. 建立原始索引
            # ----------------------------------

            original = UploadedPDFRetriever(
                pdf_bytes=pdf_bytes,
                filename=uploaded.name,
                model=model
            )

            # ----------------------------------
            # D. 建立合并索引
            # ----------------------------------

            merged = MergedPDFRetriever(
                pages=pages,
                original=original,
                model=model
            )

            # ----------------------------------
            # E. 文本完整性检查
            # ----------------------------------

            integrity = verify_text_integrity(
                pages=pages,
                original_chunks=original.chunks,
                merged_chunks=merged.chunks
            )

            # 新策略必须保留 PDF 提取文本
            if not integrity[
                "merged_matches_reference"
            ]:

                raise ValueError(
                    "合并分块没有完整保留 "
                    "PDF 提取文本。"
                    "需要进一步检查，不继续评估。"
                )

            # 注意：
            # 旧策略与原始 PDF 可能不完全相等，
            # 但不直接让实验失败。
            # 两者差异将写入报告。

            # ----------------------------------
            # F. 比较固定问题
            # ----------------------------------

            records = []

            for index, question in enumerate(
                QUESTIONS,
                start=1
            ):

                print(
                    f"\n检索问题 {index}/"
                    f"{len(QUESTIONS)}"
                )

                records.append({
                    "question": question,
                    "original": evaluate_one(
                        original,
                        question
                    ),
                    "merged": evaluate_one(
                        merged,
                        question
                    )
                })

            # ----------------------------------
            # G. 保存到 Streamlit 会话
            # ----------------------------------

            st.session_state[
                "retrieval_comparison"
            ] = {
                "pdf_hash": pdf_hash,
                "filename": uploaded.name,
                "old_count": len(
                    original.chunks
                ),
                "new_count": len(
                    merged.chunks
                ),
                "text_integrity": integrity,
                "records": records
            }

    except Exception as exc:

        st.error(
            f"实验失败：{exc}"
        )

        st.stop()


# ==========================================
# 14. 读取实验结果
# ==========================================

report = st.session_state.get(
    "retrieval_comparison"
)

if report is None:
    st.stop()

if report["pdf_hash"] != pdf_hash:

    st.warning(
        "上传文件已更换，请重新运行实验。"
    )

    st.stop()


# ==========================================
# 15. 显示分块统计
# ==========================================

st.success(
    "两套检索索引构建完成，"
    "固定问题检索成功！"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "原始 Chunk 数",
        report["old_count"]
    )

with col2:

    st.metric(
        "合并 Chunk 数",
        report["new_count"]
    )


# ==========================================
# 16. 显示文本完整性检查
# ==========================================

st.divider()

st.subheader("📄 文本完整性检查")

integrity = report[
    "text_integrity"
]

if integrity["merged_matches_reference"]:

    st.success(
        "合并分块完整保留了 PDF 提取文本。"
    )

else:

    st.error(
        "合并分块文本完整性检查失败。"
    )

if not integrity["old_matches_reference"]:

    st.warning(
        "原始检索器的 Chunk 文本与 PDF "
        "原始提取文本不完全一致。"
        "这可能是文本过滤造成的，"
        "需要结合差异进一步判断。"
    )

st.write(
    "PDF 原始提取字符数：",
    integrity["reference_chars"]
)

st.write(
    "原始 Chunk 字符数：",
    integrity["old_chars"]
)

st.write(
    "合并 Chunk 字符数：",
    integrity["merged_chars"]
)


# ==========================================
# 17. 显示每道题的检索结果
# ==========================================

st.divider()

st.subheader("🔎 五道固定问题检索对比")

for index, record in enumerate(
    report["records"],
    start=1
):

    st.markdown(
        f"### 问题 {index}"
    )

    st.write(
        record["question"]
    )

    left, right = st.columns(2)

    methods = [
        (
            left,
            "original",
            "原始分块"
        ),
        (
            right,
            "merged",
            "合并分块"
        )
    ]

    for column, method, title in methods:

        data = record[method]

        with column:

            st.markdown(
                f"**{title}**"
            )

            st.caption(
                f"上下文字符数："
                f"{data['context_chars']}"
            )

            st.caption(
                f"实际进入上下文的 Chunk："
                f"{data['source_count']}"
            )

            for item in data["results"]:

                with st.expander(
                    f"排名 {item['rank']} | "
                    f"PDF 第 {item['page']} 页 | "
                    f"{item['chars']} 字符"
                ):

                    st.caption(
                        f"RRF："
                        f"{item['rrf_score']:.5f}"
                    )

                    st.caption(
                        f"Chunk ID："
                        f"{item['chunk_id']}"
                    )

                    st.write(
                        item["text"]
                    )

    st.divider()


# ==========================================
# 18. 导出对比结果
# ==========================================

st.download_button(
    label="📥 下载检索对比 JSON",
    data=json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ).encode("utf-8"),
    file_name="pdf_retrieval_ablation.json",
    mime="application/json"
)

st.caption(
    "当前实验只展示检索结果。"
    "没有人工标准答案，因此不能直接"
    "计算检索准确率。"
)
