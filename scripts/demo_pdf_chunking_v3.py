
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

import streamlit as st


# ==========================================
# 1. 设置路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.ingestion.pdf_chunker_v3 import (
    extract_pdf_pages,
    build_v3_chunks,
    split_long_text,
    compact,
    normalize
)

from src.retrieval.uploaded_pdf_retriever import (
    split_text
)

from src.retrieval.bm25_retriever import (
    tokenize
)


# ==========================================
# 2. 原始分块
# ==========================================

def build_original_chunks(pages):

    chunks = []

    for page_index, blocks in enumerate(pages):

        for block in blocks:

            pieces = split_text(
                block,
                max_chars=1200
            )

            for piece in pieces:

                if tokenize(piece):

                    chunks.append({
                        "page_number": page_index + 1,
                        "text": piece
                    })

    return chunks


# ==========================================
# 3. V2：简单合并分块
# ==========================================

def build_merged_chunks(pages):

    chunks = []

    for page_index, blocks in enumerate(pages):

        current = ""

        def flush():

            nonlocal current

            if current and tokenize(current):

                chunks.append({
                    "page_number": page_index + 1,
                    "text": current
                })

            current = ""

        for block in blocks:

            pieces = split_long_text(
                block,
                limit=1200
            )

            for piece in pieces:

                candidate = (
                    current + " " + piece
                    if current
                    else piece
                )

                if len(candidate) <= 1200:
                    current = candidate

                else:
                    flush()
                    current = piece

        flush()

    return chunks


# ==========================================
# 4. 统计
# ==========================================

def calculate_stats(chunks):

    lengths = [
        len(item["text"])
        for item in chunks
    ]

    if not lengths:
        return {
            "count": 0,
            "avg_chars": 0,
            "short_count": 0,
            "short_ratio": 0,
            "target_ratio": 0
        }

    short_count = sum(
        x < 200
        for x in lengths
    )

    target_count = sum(
        600 <= x <= 900
        for x in lengths
    )

    return {
        "count": len(lengths),
        "avg_chars": round(mean(lengths), 1),
        "short_count": short_count,
        "short_ratio": round(
            short_count / len(lengths) * 100,
            2
        ),
        "target_ratio": round(
            target_count / len(lengths) * 100,
            2
        )
    }


# ==========================================
# 5. 检查 V3 是否保留 PDF 原文
# ==========================================

def check_integrity(pages, chunks):

    problem_pages = []

    for page_index, blocks in enumerate(pages):

        original = compact(
            "".join(blocks)
        )

        rebuilt = compact(
            "".join(
                item["text"]
                for item in chunks
                if item["page_number"] == page_index + 1
            )
        )

        if original != rebuilt:

            problem_pages.append(
                page_index + 1
            )

    return problem_pages


# ==========================================
# 6. Streamlit 页面
# ==========================================

st.set_page_config(
    page_title="PDF Chunking V3",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF 三种分块策略对比")

st.caption(
    "原始分块 vs 相邻合并 vs "
    "标题感知自适应分块"
)

st.info(
    "本次只分析 PDF 分块结构。"
    "不加载 BGE，不生成 FAISS 索引，"
    "也不调用 DeepSeek。"
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

paper_id = "upload_" + pdf_hash[:12]

title = Path(uploaded.name).stem


# ==========================================
# 7. 执行三种策略
# ==========================================

if st.button(
    "开始 V3 分块对比",
    type="primary"
):

    try:

        with st.spinner(
            "正在比较三种分块策略..."
        ):

            pages = extract_pdf_pages(
                pdf_bytes
            )

            original = build_original_chunks(
                pages
            )

            merged = build_merged_chunks(
                pages
            )

            v3 = build_v3_chunks(
                pages=pages,
                paper_id=paper_id,
                title=title
            )

            # 检查是否出现文本丢失
            problem_pages = check_integrity(
                pages,
                v3
            )

            # 收集识别到的章节
            headings = []

            for chunk in v3:

                section = chunk["section_name"]

                if section.startswith("Page "):
                    continue

                if section not in headings:
                    headings.append(section)

            report = {
                "filename": uploaded.name,
                "pdf_hash": pdf_hash,
                "page_count": len(pages),
                "original": calculate_stats(
                    original
                ),
                "merged": calculate_stats(
                    merged
                ),
                "v3": calculate_stats(v3),
                "v3_integrity_passed": (
                    len(problem_pages) == 0
                ),
                "problem_pages": problem_pages,
                "recognized_headings": headings,
                "preview_chunks": v3[:12]
            }

            st.session_state[
                "v3_comparison"
            ] = report

    except Exception as exc:

        st.error(
            f"实验失败：{exc}"
        )

        st.stop()


# ==========================================
# 8. 读取结果
# ==========================================

report = st.session_state.get(
    "v3_comparison"
)

if report is None:
    st.stop()

if report["pdf_hash"] != pdf_hash:

    st.warning(
        "上传文件已变化，请重新比较。"
    )

    st.stop()


# ==========================================
# 9. 展示三种策略统计
# ==========================================

st.divider()

st.subheader("📊 三种分块策略对比")

columns = st.columns(3)

for column, key, name in [
    (columns[0], "original", "原始分块"),
    (columns[1], "merged", "相邻合并"),
    (columns[2], "v3", "V3 结构感知")
]:

    with column:

        st.metric(
            name,
            report[key]["count"]
        )

st.table({
    "指标": [
        "Chunk 数量",
        "平均字符数",
        "短 Chunk 数",
        "短 Chunk 占比",
        "600-900 字符占比"
    ],
    "原始分块": [
        report["original"]["count"],
        report["original"]["avg_chars"],
        report["original"]["short_count"],
        f"{report['original']['short_ratio']}%",
        f"{report['original']['target_ratio']}%"
    ],
    "相邻合并": [
        report["merged"]["count"],
        report["merged"]["avg_chars"],
        report["merged"]["short_count"],
        f"{report['merged']['short_ratio']}%",
        f"{report['merged']['target_ratio']}%"
    ],
    "V3 结构感知": [
        report["v3"]["count"],
        report["v3"]["avg_chars"],
        report["v3"]["short_count"],
        f"{report['v3']['short_ratio']}%",
        f"{report['v3']['target_ratio']}%"
    ]
})


# ==========================================
# 10. 文本完整性
# ==========================================

st.divider()

st.subheader("🔍 文本完整性")

if report["v3_integrity_passed"]:

    st.success(
        "V3 文本完整性检查通过！"
        "逐页忽略空白字符后，"
        "新 Chunk 与 PDF 提取文本一致。"
    )

else:

    st.error(
        "V3 存在文本不一致的页面："
        f"{report['problem_pages']}"
    )

    st.warning(
        "文本完整性未通过，"
        "暂时不要用于正式检索。"
    )


# ==========================================
# 11. 查看章节识别结果
# ==========================================

st.divider()

st.subheader("📑 自动识别的章节标题")

headings = report["recognized_headings"]

st.caption(
    f"识别到 {len(headings)} 个不同的标题。"
    "这些是启发式识别结果，需要人工核对。"
)

if headings:

    for heading in headings:
        st.write("•", heading)

else:

    st.warning(
        "没有识别到明显的章节标题。"
        "V3 仍可以按长度进行分块。"
    )


# ==========================================
# 12. 预览 V3 Chunk
# ==========================================

st.divider()

st.subheader("📖 V3 Chunk 预览")

for index, chunk in enumerate(
    report["preview_chunks"],
    start=1
):

    with st.expander(
        f"Chunk {index} | "
        f"PDF 第 {chunk['page_number']} 页 | "
        f"{chunk['char_count']} 字符"
    ):

        st.write(
            "章节标签：",
            chunk["section_name"]
        )

        st.write(
            "Chunk ID：",
            chunk["chunk_id"]
        )

        st.write(
            chunk["text"]
        )


# ==========================================
# 13. 导出报告
# ==========================================

st.divider()

st.download_button(
    "📥 下载 V3 分块对比报告",
    data=json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ).encode("utf-8"),
    file_name="pdf_chunking_v3_comparison.json",
    mime="application/json"
)
