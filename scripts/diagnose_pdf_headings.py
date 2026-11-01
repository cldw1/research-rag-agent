
import hashlib
import json
import re
import statistics

import pymupdf
import streamlit as st


# ==========================================
# 1. 章节标题识别规则
# ==========================================

# 优先匹配 5.2 / 5.2.1 / 5.2.1.1
# 不把 1. / 2. 这样的列表序号当章节
SECTION_RE = re.compile(
    r"(?<![\d.])"
    r"(?P<number>[1-9]\d?(?:\.\d+){1,3})"
    r"\.?\s+"
    r"(?P<title>[A-Z][A-Za-z]"
    r"[A-Za-z0-9\s,'’()\-/:]{2,80})"
)

SPECIAL_TITLES = {
    "abstract",
    "introduction",
    "references",
    "bibliography",
    "conclusion",
    "conclusions",
    "related work",
    "acknowledgments",
    "acknowledgements"
}


# ==========================================
# 2. 从 PDF 提取带字体信息的行
# ==========================================

def extract_lines(pdf_bytes):

    records = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as doc:

        if doc.needs_pass:
            raise ValueError("暂不支持加密 PDF")

        if len(doc) > 80:
            raise ValueError("PDF 超过 80 页")

        for page_num, page in enumerate(
            doc,
            start=1
        ):

            page_lines = []

            layout = page.get_text(
                "dict",
                sort=True
            )

            for block in layout["blocks"]:

                if block.get("type") != 0:
                    continue

                for line in block.get("lines", []):

                    spans = line.get("spans", [])

                    text = "".join(
                        span.get("text", "")
                        for span in spans
                    ).strip()

                    if not text:
                        continue

                    sizes = [
                        float(span.get("size", 0))
                        for span in spans
                    ]

                    bold = any(
                        "bold" in str(
                            span.get("font", "")
                        ).lower()
                        or (
                            int(span.get("flags", 0))
                            & 16
                        ) != 0
                        for span in spans
                    )

                    rect = line.get(
                        "bbox",
                        (0, 0, 0, 0)
                    )

                    page_lines.append({
                        "page": page_num,
                        "text": text,
                        "font_size": round(
                            max(sizes, default=0),
                            2
                        ),
                        "bold": bold,
                        "y_position": round(
                            float(rect[1]),
                            1
                        )
                    })

            records.extend(page_lines)

    return records


# ==========================================
# 3. 识别章节候选
# ==========================================

def find_candidates(lines):

    body_sizes = [
        item["font_size"]
        for item in lines
        if len(item["text"]) >= 65
        and item["font_size"] > 0
    ]

    body_size = (
        statistics.median(body_sizes)
        if body_sizes
        else 10.0
    )

    candidates = []

    for index, line in enumerate(lines):

        text = " ".join(
            line["text"].split()
        )

        kind = None
        candidate = None

        # ----------------------------------
        # A. Introduction / References 等
        # ----------------------------------

        if text.casefold() in SPECIAL_TITLES:

            kind = "独立通用标题"
            candidate = text

        else:

            # ----------------------------------
            # B. 数字编号标题
            # ----------------------------------

            match = SECTION_RE.search(text)

            if match:

                candidate = match.group(0).strip()

                if match.start() == 0:
                    kind = "行首编号标题"
                else:
                    kind = "行内疑似标题"

        # ----------------------------------
        # C. 编号独占一行
        # ----------------------------------

        if kind is None and re.fullmatch(
            r"[1-9]\d?(?:\.\d+){1,3}",
            text
        ):

            if index + 1 < len(lines):

                next_line = lines[index + 1]

                next_text = next_line["text"].strip()

                if (
                    next_line["page"] == line["page"]
                    and 5 <= len(next_text) <= 70
                    and re.match(
                        r"^[A-Z][A-Za-z]",
                        next_text
                    )
                    and not next_text.endswith(
                        (".", "?", "!")
                    )
                ):

                    kind = "编号与标题可能分行"

                    candidate = (
                        text + " " + next_text
                    )

        if kind is None:
            continue

        # ----------------------------------
        # D. 记录排版特征
        # ----------------------------------

        larger = (
            line["font_size"]
            > body_size + 0.4
        )

        candidates.append({
            "page": line["page"],
            "candidate": candidate,
            "type": kind,
            "font_size": line["font_size"],
            "body_font_size": round(
                body_size,
                2
            ),
            "bold": line["bold"],
            "larger_than_body": larger,
            "y_position": line["y_position"],
            "original_line": text,
            "needs_review": True
        })

    return candidates


# ==========================================
# 4. Streamlit 页面
# ==========================================

st.set_page_config(
    page_title="PDF Heading Diagnosis",
    page_icon="📑",
    layout="wide"
)

st.title("📑 PDF 章节标题诊断")

st.caption(
    "结合章节编号、字体大小、加粗情况"
    "和 PDF 行结构，分析章节候选。"
)

st.info(
    "本次仅生成章节候选清单，"
    "并不自动认定这些候选全部正确。"
    "不修改 V3 分块，也不调用 DeepSeek。"
)

uploaded = st.file_uploader(
    "上传 PDF",
    type=["pdf"]
)

if uploaded is None:
    st.stop()

pdf_bytes = uploaded.getvalue()
pdf_hash = hashlib.sha256(
    pdf_bytes
).hexdigest()

if st.button(
    "分析章节标题",
    type="primary"
):

    try:

        with st.spinner(
            "正在分析 PDF 的文字排版..."
        ):

            lines = extract_lines(
                pdf_bytes
            )

            candidates = find_candidates(
                lines
            )

            report = {
                "filename": uploaded.name,
                "pdf_hash": pdf_hash,
                "line_count": len(lines),
                "candidate_count": len(candidates),
                "candidates": candidates
            }

            st.session_state[
                "heading_diagnosis"
            ] = report

    except Exception as exc:

        st.error(
            f"分析失败：{exc}"
        )
        st.stop()


report = st.session_state.get(
    "heading_diagnosis"
)

if (
    report is None
    or report["pdf_hash"] != pdf_hash
):
    st.stop()


# ==========================================
# 5. 展示结果
# ==========================================

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "PDF 文本行数",
        report["line_count"]
    )

with col2:
    st.metric(
        "章节候选数量",
        report["candidate_count"]
    )

st.subheader("章节标题候选")

st.caption(
    "请重点检查 5.2、5.3、5.4、5.5 "
    "等章节，以及数学公式和算法步骤"
    "是否被错误识别。"
)

st.dataframe(
    report["candidates"],
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 6. 查看原始行和排版信息
# ==========================================

st.divider()

st.subheader("候选详情")

for index, item in enumerate(
    report["candidates"],
    start=1
):

    with st.expander(
        f"{index}. 第 {item['page']} 页 | "
        f"{item['candidate'][:65]}"
    ):

        st.write(
            "候选类型：",
            item["type"]
        )

        st.write(
            "字体大小：",
            item["font_size"]
        )

        st.write(
            "正文参考字体大小：",
            item["body_font_size"]
        )

        st.write(
            "是否加粗：",
            item["bold"]
        )

        st.write(
            "原始 PDF 行："
        )

        st.code(
            item["original_line"],
            language=None
        )


# ==========================================
# 7. 导出诊断结果
# ==========================================

st.divider()

st.download_button(
    "📥 下载章节标题诊断 JSON",
    data=json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ).encode("utf-8"),
    file_name="pdf_heading_diagnosis.json",
    mime="application/json"
)
