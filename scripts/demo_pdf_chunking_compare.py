
import hashlib
import re
import sys
from pathlib import Path
from statistics import mean

import pymupdf
import streamlit as st


# ==========================================
# 1. 项目根目录
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25_retriever import tokenize

from src.retrieval.uploaded_pdf_retriever import (
    split_text
)


# ==========================================
# 2. 参数
# ==========================================

MAX_CHARS = 1200
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PAGES = 80


# ==========================================
# 3. 文本清理
# ==========================================

def normalize_text(text):

    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


# ==========================================
# 4. 读取 PDF 的文本块
# ==========================================

def extract_pages(pdf_bytes):

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF 不能超过 20 MB")

    pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as doc:

        if doc.needs_pass:
            raise ValueError("暂不支持加密 PDF")

        if len(doc) > MAX_PAGES:
            raise ValueError("PDF 不能超过 80 页")

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
# 5. 原始分块方法
# ==========================================

def build_old_chunks(pages):

    chunks = []

    for page_index, blocks in enumerate(
        pages
    ):

        for block in blocks:

            pieces = split_text(
                block,
                max_chars=MAX_CHARS
            )

            for piece in pieces:

                if not tokenize(piece):
                    continue

                chunks.append({
                    "page": page_index + 1,
                    "text": piece
                })

    return chunks


# ==========================================
# 6. 对超长文本进行安全切分
# ==========================================

def split_long_text(text, limit):

    words = normalize_text(text).split()

    pieces = []
    current = ""

    for word in words:

        # 处理极少见的超长连续字符串
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

            pieces.append(current)
            current = word

        else:

            current = candidate

    if current:
        pieces.append(current)

    return pieces


# ==========================================
# 7. 新分块方法：合并相邻文本块
# ==========================================

def build_merged_chunks(pages):

    chunks = []

    for page_index, blocks in enumerate(
        pages
    ):

        current = ""

        # 将当前缓存保存成 Chunk
        def flush():

            nonlocal current

            if current and tokenize(current):

                chunks.append({
                    "page": page_index + 1,
                    "text": current,
                    "char_count": len(current)
                })

            current = ""

        for block in blocks:

            pieces = split_long_text(
                block,
                MAX_CHARS
            )

            for piece in pieces:

                candidate = (
                    f"{current} {piece}"
                    if current
                    else piece
                )

                if len(candidate) <= MAX_CHARS:

                    current = candidate

                else:

                    flush()
                    current = piece

        # 不跨 PDF 页面合并
        flush()

    return chunks


# ==========================================
# 8. 计算分块统计指标
# ==========================================

def summarize(chunks):

    if not chunks:
        return {
            "count": 0,
            "avg_chars": 0,
            "small_count": 0,
            "small_ratio": 0
        }

    lengths = [
        len(chunk["text"])
        for chunk in chunks
    ]

    small_count = sum(
        length < 200
        for length in lengths
    )

    return {
        "count": len(chunks),
        "avg_chars": mean(lengths),
        "small_count": small_count,
        "small_ratio": (
            small_count / len(chunks)
        )
    }


def compact(text):

    return re.sub(
        r"\s+",
        "",
        text
    )


# ==========================================
# 9. Streamlit 页面
# ==========================================

st.set_page_config(
    page_title="PDF Chunking Comparison",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF 分块策略对比")

st.caption(
    "原始文本块切分 vs 相邻文本块合并"
)

st.info(
    "本次只比较分块效果。"
    "不生成 Embedding，不调用 DeepSeek，"
    "也不修改正式项目中的检索索引。"
)

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

if st.button(
    "开始比较两种分块方法",
    type="primary"
):

    try:

        with st.spinner(
            "正在解析 PDF 并比较 Chunk..."
        ):

            pages = extract_pages(
                pdf_bytes
            )

            old_chunks = build_old_chunks(
                pages
            )

            new_chunks = build_merged_chunks(
                pages
            )

            old_stats = summarize(
                old_chunks
            )

            new_stats = summarize(
                new_chunks
            )

            original_text = "".join(
                block
                for page in pages
                for block in page
            )

            rebuilt_text = "".join(
                chunk["text"]
                for chunk in new_chunks
            )

            text_preserved = (
                compact(original_text)
                == compact(rebuilt_text)
            )

            st.session_state[
                "chunk_comparison"
            ] = {
                "hash": pdf_hash,
                "pages": len(pages),
                "old_stats": old_stats,
                "new_stats": new_stats,
                "new_chunks": new_chunks,
                "text_preserved": text_preserved
            }

    except Exception as exc:

        st.error(
            f"比较失败：{exc}"
        )

        st.stop()


# ==========================================
# 10. 展示比较结果
# ==========================================

result = st.session_state.get(
    "chunk_comparison"
)

if result is None or result["hash"] != pdf_hash:
    st.stop()

st.divider()

st.subheader("📊 分块结果对比")

old = result["old_stats"]
new = result["new_stats"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "PDF 页数",
        result["pages"]
    )

with col2:
    st.metric(
        "原始 Chunk 数",
        old["count"]
    )

with col3:
    st.metric(
        "合并后 Chunk 数",
        new["count"]
    )

st.table({
    "指标": [
        "Chunk 数量",
        "平均字符数",
        "短 Chunk 数（小于200字符）",
        "短 Chunk 占比"
    ],
    "原始方式": [
        old["count"],
        round(old["avg_chars"], 1),
        old["small_count"],
        f"{old['small_ratio']:.2%}"
    ],
    "合并方式": [
        new["count"],
        round(new["avg_chars"], 1),
        new["small_count"],
        f"{new['small_ratio']:.2%}"
    ]
})

if result["text_preserved"]:

    st.success(
        "文本完整性检查通过："
        "忽略空白字符后，合并前后的文本一致。"
    )

else:

    st.warning(
        "文本完整性检查未通过。"
        "需要检查是否存在未保留的字符，"
        "暂时不要替换正式分块逻辑。"
    )


# ==========================================
# 11. 查看合并后的 Chunk
# ==========================================

st.divider()

st.subheader("🔎 查看新的 Chunk")

new_chunks = result["new_chunks"]

preview_count = min(
    10,
    len(new_chunks)
)

for i, chunk in enumerate(
    new_chunks[:preview_count],
    start=1
):

    with st.expander(
        f"Chunk {i} | "
        f"PDF 第 {chunk['page']} 页 | "
        f"{chunk['char_count']} 字符"
    ):

        st.write(
            chunk["text"]
        )

st.caption(
    "这里只预览前10个 Chunk。"
    "下一步将评估检索质量，"
    "再决定是否替换正式分块策略。"
)
