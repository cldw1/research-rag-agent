
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import streamlit as st


# ==========================================
# 1. 项目路径
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.pdf_chunker_v3 import (
    extract_pdf_pages,
    split_long_text,
    compact
)


HEADING_FILE = (
    ROOT / "results" / "pdf_headings_filtered.json"
)

OUTPUT_FILE = (
    ROOT / "results" / "pdf_v3_section_mapping_report.json"
)

MIN_CHARS = 450
TARGET_CHARS = 800
MAX_CHARS = 1000


# ==========================================
# 2. 文字匹配辅助函数
# ==========================================

def normalize(text):
    return " ".join(str(text).split())


def match_key(text):
    """
    仅用于匹配，原文不会被修改。
    NFKC 可以统一部分 PDF 连字形式。
    """
    text = unicodedata.normalize(
        "NFKC", str(text)
    )
    return normalize(text).casefold()


# ==========================================
# 3. 读取章节筛选结果
# ==========================================

def load_headings():

    if not HEADING_FILE.exists():
        raise FileNotFoundError(
            f"找不到文件：{HEADING_FILE}"
        )

    data = json.loads(
        HEADING_FILE.read_text(encoding="utf-8")
    )

    return data


# ==========================================
# 4. 展开 PDF 的文本行
# ==========================================

def flatten_page(blocks):

    lines = []

    for block in blocks:
        for raw_line in block.splitlines():

            line = normalize(raw_line)

            if line:
                lines.append(line)

    return lines


# ==========================================
# 5. 判断某个位置是否为章节标题
# ==========================================

def heading_span(lines, index, heading):

    number = heading.get("section_number")
    title = heading["section_title"]

    # 例如 References
    if number is None:
        if match_key(lines[index]) == match_key(title):
            return 1
        return 0

    complete = f"{number} {title}"

    # 情况 A：编号与标题在同一行
    if match_key(lines[index]) == match_key(complete):
        return 1

    # 情况 B：编号独占一行
    number_text = lines[index].rstrip(".")

    if match_key(number_text) != match_key(number):
        return 0

    # 标题可能占一到三行
    for count in (1, 2, 3):

        end = index + 1 + count

        if end > len(lines):
            break

        title_lines = lines[index + 1:end]

        if match_key(
            " ".join(title_lines)
        ) == match_key(title):

            return count + 1

    return 0


# ==========================================
# 6. 将章节匹配到 PDF 的真实位置
# ==========================================

def locate_headings(pages, headings):

    by_page = {}

    for heading in headings:
        page = int(heading["page"])
        by_page.setdefault(page, []).append(heading)

    markers = {}
    matched = []
    unmatched = []

    for page_number, blocks in enumerate(
        pages, start=1
    ):

        lines = flatten_page(blocks)

        page_headings = sorted(
            by_page.get(page_number, []),
            key=lambda x: x["y_position"]
        )

        events = {}
        search_start = 0

        for heading in page_headings:

            found = False

            for index in range(
                search_start, len(lines)
            ):

                span = heading_span(
                    lines, index, heading
                )

                if span:

                    events[index] = heading["candidate"]

                    matched.append({
                        "page": page_number,
                        "line_index": index,
                        "section": heading["candidate"]
                    })

                    search_start = index + span
                    found = True
                    break

            if not found:
                unmatched.append({
                    "page": page_number,
                    "section": heading["candidate"]
                })

        markers[page_number] = events

    return markers, matched, unmatched


# ==========================================
# 7. 生成带正确章节标签的 V3 Chunk
# ==========================================

def build_labeled_chunks(pages, markers, paper_id, title):

    chunks = []

    active_section = None

    for page_index, blocks in enumerate(pages):

        page_number = page_index + 1
        lines = flatten_page(blocks)
        events = markers.get(page_number, {})

        buffer = []
        chunk_index = 0

        current_section = active_section

        def flush():

            nonlocal buffer, chunk_index

            if not buffer:
                return

            text = " ".join(buffer)

            section_name = (
                current_section
                if current_section
                else f"Page {page_number}"
            )

            chunks.append({
                "chunk_id": (
                    f"{paper_id}_v3_"
                    f"s{page_index:03d}_"
                    f"p{chunk_index:03d}_c000"
                ),
                "paper_id": paper_id,
                "title": title,
                "page_number": page_number,
                "section_name": section_name,
                "section_index": page_index,
                "paragraph_index": chunk_index,
                "piece_index": 0,
                "text": text,
                "char_count": len(text),
                "is_reference": (
                    section_name.casefold() == "references"
                )
            })

            chunk_index += 1
            buffer = []

        for line_index, line in enumerate(lines):

            # 遇到章节标题：
            # 先结束上一章节的 Chunk，
            # 再给后续内容设置新的章节标签。
            if line_index in events:

                flush()

                active_section = events[line_index]
                current_section = active_section

            pieces = split_long_text(
                line, limit=MAX_CHARS
            )

            for piece in pieces:

                candidate = " ".join(
                    buffer + [piece]
                )

                current_length = len(
                    " ".join(buffer)
                )

                should_flush = (
                    bool(buffer)
                    and (
                        len(candidate) > MAX_CHARS
                        or (
                            current_length >= MIN_CHARS
                            and len(candidate) > TARGET_CHARS
                        )
                    )
                )

                if should_flush:
                    flush()

                buffer.append(piece)

        # 每页结束后保存，不跨页拼接
        flush()

    return chunks


# ==========================================
# 8. 文本完整性检查
# ==========================================

def verify_integrity(pages, chunks):

    problems = []

    for page_number, blocks in enumerate(
        pages, start=1
    ):

        original = compact(
            "".join(blocks)
        )

        rebuilt = compact(
            "".join(
                chunk["text"]
                for chunk in chunks
                if chunk["page_number"] == page_number
            )
        )

        if original != rebuilt:
            problems.append(page_number)

    return problems


# ==========================================
# 9. Streamlit 页面
# ==========================================

st.set_page_config(
    page_title="V3 Section Mapping",
    page_icon="📑",
    layout="wide"
)

st.title("📑 V3 章节标签映射验证")

st.caption(
    "将筛选后的 PDF 章节标题映射到实际 Chunk"
)

st.info(
    "本次只验证分块、章节标签和原文完整性。"
    "不调用 DeepSeek，也不修改正式 Agent。"
)

uploaded = st.file_uploader(
    "上传 HandbookChapter5.pdf",
    type=["pdf"]
)

if uploaded is None:
    st.stop()


# ==========================================
# 10. 执行
# ==========================================

if st.button(
    "开始映射章节标签",
    type="primary"
):

    try:

        heading_data = load_headings()

        if uploaded.name != heading_data["filename"]:
            raise ValueError(
                "上传的 PDF 文件名与章节诊断文件不一致。"
                "请使用生成该诊断文件的原始 PDF。"
            )

        with st.spinner(
            "正在映射章节并生成新的 Chunk..."
        ):

            pages = extract_pdf_pages(
                uploaded.getvalue()
            )

            markers, matched, unmatched = locate_headings(
                pages,
                heading_data["headings"]
            )

            chunks = build_labeled_chunks(
                pages,
                markers,
                paper_id="v3_test",
                title=Path(uploaded.name).stem
            )

            problem_pages = verify_integrity(
                pages, chunks
            )

            section_counts = dict(
                Counter(
                    chunk["section_name"]
                    for chunk in chunks
                )
            )

            important_sections = [
                "5.2.3.3 The Leapfrog Method",
                "5.4.2 Tuning HMC",
                "5.5.7 Tempering during a Trajectory",
                "References"
            ]

            previews = []

            for section in important_sections:

                found = next(
                    (
                        chunk for chunk in chunks
                        if chunk["section_name"] == section
                    ),
                    None
                )

                if found:
                    previews.append(found)

            report = {
                "filename": uploaded.name,
                "expected_headings": len(
                    heading_data["headings"]
                ),
                "matched_headings": len(matched),
                "unmatched_headings": unmatched,
                "matched_events": matched,
                "chunk_count": len(chunks),
                "section_chunk_counts": section_counts,
                "integrity_passed": (
                    not problem_pages
                ),
                "problem_pages": problem_pages,
                "preview_chunks": previews
            }

            OUTPUT_FILE.parent.mkdir(
                parents=True, exist_ok=True
            )

            OUTPUT_FILE.write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2
                ),
                encoding="utf-8"
            )

            st.session_state["mapping_report"] = report

    except Exception as exc:
        st.error(f"映射失败：{exc}")
        st.stop()


# ==========================================
# 11. 结果展示
# ==========================================

report = st.session_state.get("mapping_report")

if report is None:
    st.stop()

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "待匹配标题",
        report["expected_headings"]
    )

with col2:
    st.metric(
        "成功匹配标题",
        report["matched_headings"]
    )

with col3:
    st.metric(
        "生成 Chunk",
        report["chunk_count"]
    )

if report["integrity_passed"]:
    st.success("逐页文本完整性检查通过。")
else:
    st.error(
        "以下页面文本检查未通过："
        f"{report['problem_pages']}"
    )

if report["unmatched_headings"]:

    st.warning(
        f"仍有 {len(report['unmatched_headings'])} "
        "个标题未定位成功。"
    )

    st.dataframe(
        report["unmatched_headings"],
        hide_index=True
    )

else:
    st.success("50 个章节候选全部完成位置匹配。")

st.divider()

st.subheader("📖 重点章节的 Chunk 预览")

for chunk in report["preview_chunks"]:

    with st.expander(
        f"{chunk['section_name']} | "
        f"第 {chunk['page_number']} 页"
    ):

        st.write(
            "Chunk ID：", chunk["chunk_id"]
        )

        st.write(chunk["text"])

st.divider()

st.caption(
    "报告已保存至 results/"
    "pdf_v3_section_mapping_report.json"
)
