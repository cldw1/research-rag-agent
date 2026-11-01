
import hashlib
import sys
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.uploaded_pdf_retriever import (
    UploadedPDFRetriever
)


# ==========================================
# 2. 页面设置
# ==========================================

st.set_page_config(
    page_title="PDF Upload Test",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF 论文上传与检索测试")

st.caption(
    "上传 PDF → 提取文本 → 自动分块 "
    "→ BGE Embedding → BM25 + FAISS + RRF"
)


# ==========================================
# 3. 加载 BGE 模型
# ==========================================

@st.cache_resource(
    show_spinner="正在加载 BGE Embedding 模型..."
)
def load_bge_model():

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
# 4. PDF 上传区域
# ==========================================

uploaded_file = st.file_uploader(
    "选择一篇 PDF 论文",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded_file is None:

    st.info(
        "请先上传一篇包含可复制文本的 PDF 论文。"
    )

    st.stop()


pdf_bytes = uploaded_file.getvalue()

pdf_hash = hashlib.sha256(
    pdf_bytes
).hexdigest()


st.write(
    "文件名称：",
    uploaded_file.name
)

st.write(
    "文件大小：",
    f"{len(pdf_bytes) / 1024 / 1024:.2f} MB"
)


# ==========================================
# 5. 解析并建立索引
# ==========================================

if st.button(
    "开始解析并建立索引",
    type="primary"
):

    try:

        with st.spinner(
            "正在解析 PDF 并构建检索索引..."
        ):

            model = load_bge_model()

            retriever = UploadedPDFRetriever(
                pdf_bytes=pdf_bytes,
                filename=uploaded_file.name,
                model=model
            )

        st.session_state["pdf_retriever"] = (
            retriever
        )

        st.session_state["pdf_hash"] = (
            pdf_hash
        )

        st.success(
            "PDF 论文解析与检索索引构建完成！"
        )

    except Exception as exc:

        st.error(
            f"处理失败：{exc}"
        )

        st.stop()


# ==========================================
# 6. 检查是否已建立当前文件索引
# ==========================================

retriever = st.session_state.get(
    "pdf_retriever"
)

indexed_hash = st.session_state.get(
    "pdf_hash"
)

if retriever is None or indexed_hash != pdf_hash:

    st.warning(
        "请点击「开始解析并建立索引」。"
    )

    st.stop()


# ==========================================
# 7. 展示索引信息
# ==========================================

st.divider()

st.subheader("📊 PDF 解析结果")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "PDF 页数",
        retriever.page_count
    )

with col2:

    st.metric(
        "Chunk 数量",
        len(retriever.chunks)
    )

with col3:

    st.metric(
        "FAISS 向量数量",
        retriever.index.ntotal
    )

st.write(
    "论文名称：",
    retriever.title
)

st.write(
    "自动生成的论文 ID：",
    retriever.paper_id
)


# ==========================================
# 8. 检索测试
# ==========================================

st.divider()

st.subheader("🔎 检索论文证据")

question = st.text_input(
    "输入英文问题",
    value=(
        "What is the main contribution "
        "of this paper?"
    )
)


if st.button("执行 Hybrid Top-5 检索"):

    try:

        results = retriever.search(
            query=question,
            paper_id=retriever.paper_id,
            top_k=5,
            candidate_k=50
        )

        st.success(
            f"成功检索到 {len(results)} 个相关 Chunk"
        )

        for item in results:

            with st.expander(
                f"排名 {item['rank']} | "
                f"第 {item['page_number']} 页 | "
                f"RRF={item['rrf_score']:.5f}"
            ):

                st.write(
                    "Chunk ID：",
                    item["chunk_id"]
                )

                st.write(
                    "BM25 排名：",
                    item["bm25_rank"]
                )

                st.write(
                    "Dense 排名：",
                    item["dense_rank"]
                )

                st.markdown(
                    "**论文原文：**"
                )

                st.write(
                    item["text"]
                )

    except Exception as exc:

        st.error(
            f"检索失败：{exc}"
        )


st.caption(
    "当前是 PDF 上传和检索测试。"
    "下一步再接入 DeepSeek Agent，"
    "实现基于上传论文的完整问答。"
)
