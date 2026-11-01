
import hashlib
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.pdf_chunker_v3 import (
    extract_pdf_pages,
    split_long_text,
    compact
)

from src.generation.strict_context_builder import (
    build_strict_context
)


# ==========================================
# 1. 配置
# ==========================================

MAPPING_PATH = (
    ROOT / "results"
    / "pdf_v3_section_mapping_report.json"
)

RERANK_PATH = (
    ROOT / "results"
    / "pdf_v3_constrained_rerank_5.json"
)

PREVIOUS_PATH = (
    ROOT / "results"
    / "pdf_v3_retrieval_5_questions.json"
)

OUTPUT_PATH = (
    ROOT / "results"
    / "pdf_v3_neighbor_expansion.json"
)

MAX_CONTEXT_CHARS = 8000
MAX_EXTRA_CHUNKS = 3


# ==========================================
# 2. 读取报告
# ==========================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"找不到文件：{path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


# ==========================================
# 3. 提取每页 PDF 文本行
# ==========================================

def flatten_page(blocks):

    lines = []

    for block in blocks:

        for raw_line in block.splitlines():

            line = " ".join(
                raw_line.split()
            )

            if line:
                lines.append(line)

    return lines


# ==========================================
# 4. 读取章节位置
# ==========================================

def get_markers(report):

    markers = {}

    for event in report["matched_events"]:

        page = int(event["page"])
        index = int(event["line_index"])

        markers.setdefault(
            page, {}
        )[index] = event["section"]

    return markers


# ==========================================
# 5. 重建完整 V3 Chunk
# ==========================================

def build_chunks(
    pages,
    markers,
    paper_id,
    title
):

    chunks = []
    active_section = None

    for page_index, blocks in enumerate(pages):

        page_number = page_index + 1
        lines = flatten_page(blocks)

        events = markers.get(
            page_number, {}
        )

        buffer = []
        chunk_index = 0

        def flush():

            nonlocal buffer, chunk_index

            if not buffer:
                return

            section = (
                active_section
                or f"Page {page_number}"
            )

            text = " ".join(buffer)

            chunks.append({
                "chunk_id": (
                    f"{paper_id}_v3_"
                    f"s{page_index:03d}_"
                    f"p{chunk_index:03d}_c000"
                ),
                "paper_id": paper_id,
                "title": title,
                "section_name": section,
                "section_index": page_index,
                "paragraph_index": chunk_index,
                "piece_index": 0,
                "page_number": page_number,
                "text": text,
                "char_count": len(text),
                "is_reference": (
                    section.casefold()
                    in {"references", "bibliography"}
                )
            })

            chunk_index += 1
            buffer = []

        for line_index, line in enumerate(lines):

            if line_index in events:

                flush()

                active_section = events[
                    line_index
                ]

            pieces = split_long_text(
                line,
                limit=1000
            )

            for piece in pieces:

                candidate = " ".join(
                    buffer + [piece]
                )

                current_length = len(
                    " ".join(buffer)
                )

                if buffer and (
                    len(candidate) > 1000
                    or (
                        current_length >= 450
                        and len(candidate) > 800
                    )
                ):
                    flush()

                buffer.append(piece)

        flush()

    # ======================================
    # 文本完整性
    # ======================================

    problem_pages = []

    for page_number, blocks in enumerate(
        pages,
        start=1
    ):

        reference = compact(
            "".join(blocks)
        )

        reconstructed = compact(
            "".join(
                c["text"]
                for c in chunks
                if c["page_number"]
                == page_number
            )
        )

        if reference != reconstructed:

            problem_pages.append(
                page_number
            )

    if problem_pages:

        raise ValueError(
            "文本完整性失败："
            f"{problem_pages}"
        )

    return chunks


# ==========================================
# 6. 转换成上下文构建器需要的格式
# ==========================================

def prepare_results(chunks):

    return [
        {
            **chunk,
            "rank": index,
            "rrf_score": 0.0
        }
        for index, chunk in enumerate(
            chunks, start=1
        )
    ]


def evaluate_context(chunks):

    results = prepare_results(chunks)

    context, sources = build_strict_context(
        results,
        max_chars=MAX_CONTEXT_CHARS
    )

    return {
        "context": context,
        "source_count": len(sources),
        "context_chars": len(context),
        "all_included": (
            len(sources) == len(chunks)
        )
    }


# ==========================================
# 7. 邻近 Chunk 扩展
# ==========================================

def expand_neighbors(
    selected_results,
    all_chunks
):

    chunk_lookup = {
        c["chunk_id"]: c
        for c in all_chunks
    }

    index_lookup = {
        c["chunk_id"]: index
        for index, c in enumerate(all_chunks)
    }

    anchors = []

    for result in selected_results:

        chunk_id = result["chunk_id"]

        if chunk_id not in chunk_lookup:

            raise ValueError(
                f"找不到原始 Chunk：{chunk_id}"
            )

        chunk = chunk_lookup[chunk_id]

        # 确认重新构建的正文与之前一致
        if compact(chunk["text"]) != compact(
            result["text"]
        ):
            raise ValueError(
                "新旧 Chunk 正文不一致："
                f"{chunk_id}"
            )

        anchors.append(chunk)

    baseline = evaluate_context(anchors)

    if not baseline["all_included"]:

        raise ValueError(
            "原始 Top-5 已超过上下文预算，"
            "不能安全进行邻近扩展。"
        )

    selected = list(anchors)

    selected_ids = {
        c["chunk_id"]
        for c in selected
    }

    additions = []

    # 依照原始排名顺序扩展
    for anchor in anchors:

        if len(additions) >= MAX_EXTRA_CHUNKS:
            break

        anchor_index = index_lookup[
            anchor["chunk_id"]
        ]

        # 优先检查前一个 Chunk
        for offset in (-1, 1):

            neighbor_index = (
                anchor_index + offset
            )

            if not (
                0 <= neighbor_index < len(all_chunks)
            ):
                continue

            neighbor = all_chunks[
                neighbor_index
            ]

            if (
                neighbor["chunk_id"]
                in selected_ids
            ):
                continue

            # 只扩展同一章节的正文
            if (
                neighbor["section_name"]
                != anchor["section_name"]
            ):
                continue

            # 普通内容问答不扩展参考文献
            if neighbor["is_reference"]:
                continue

            trial = selected + [neighbor]

            check = evaluate_context(trial)

            # 必须保证所有原始证据仍可放入
            if not check["all_included"]:
                continue

            selected.append(neighbor)

            selected_ids.add(
                neighbor["chunk_id"]
            )

            additions.append({
                "anchor_chunk_id": (
                    anchor["chunk_id"]
                ),
                "chunk_id": neighbor["chunk_id"],
                "page": neighbor["page_number"],
                "section": neighbor["section_name"],
                "text": neighbor["text"],
                "chars": len(neighbor["text"]),
                "direction": (
                    "previous"
                    if offset == -1
                    else "next"
                )
            })

            # 每个 Anchor 最多扩展一个邻居
            break

    expanded = evaluate_context(selected)

    return {
        "before": {
            "context_chars": baseline[
                "context_chars"
            ],
            "source_count": baseline[
                "source_count"
            ]
        },
        "after": {
            "context_chars": expanded[
                "context_chars"
            ],
            "source_count": expanded[
                "source_count"
            ]
        },
        "anchors_preserved": (
            expanded["all_included"]
        ),
        "anchor_ids": [
            c["chunk_id"]
            for c in anchors
        ],
        "added_neighbors": additions,
        "expanded_chunk_ids": [
            c["chunk_id"]
            for c in selected
        ],
        "expanded_sources": [
            {
                "chunk_id": c["chunk_id"],
                "page": c["page_number"],
                "section": c["section_name"],
                "text": c["text"]
            }
            for c in selected
        ]
    }


# ==========================================
# 8. Streamlit 页面
# ==========================================

st.set_page_config(
    page_title="V3 Neighbor Expansion",
    page_icon="📚",
    layout="wide"
)

st.title("📚 第59步：同章节邻近证据扩展")

st.caption(
    "保留原始 Top-5，在 8000 字符预算内"
    "补充同章节的相邻 Chunk。"
)

st.info(
    "不重建 FAISS，不加载 BGE，"
    "不调用 DeepSeek，也不修改正式 Agent。"
)

uploaded = st.file_uploader(
    "上传 HandbookChapter5.pdf",
    type=["pdf"]
)

if uploaded is None:
    st.stop()

pdf_bytes = uploaded.getvalue()
pdf_hash = hashlib.sha256(
    pdf_bytes
).hexdigest()


# ==========================================
# 9. 运行
# ==========================================

if st.button(
    "开始邻近证据扩展",
    type="primary"
):

    try:

        mapping = load_json(
            MAPPING_PATH
        )

        constrained = load_json(
            RERANK_PATH
        )

        previous = load_json(
            PREVIOUS_PATH
        )

        if previous.get("pdf_hash") != pdf_hash:
            raise ValueError(
                "PDF 与第56步的文件不一致"
            )

        if uploaded.name != mapping["filename"]:
            raise ValueError(
                "PDF 文件名与章节映射报告不一致"
            )

        with st.spinner(
            "正在重建 Chunk 并扩展邻近证据..."
        ):

            pages = extract_pdf_pages(
                pdf_bytes
            )

            markers = get_markers(
                mapping
            )

            chunks = build_chunks(
                pages=pages,
                markers=markers,
                paper_id=(
                    "upload_" + pdf_hash[:12]
                ),
                title=Path(uploaded.name).stem
            )

            if len(chunks) != 247:
                raise ValueError(
                    "V3 Chunk 数量与之前实验不一致："
                    f"{len(chunks)}"
                )

            records = []

            for record in constrained["records"]:

                result = expand_neighbors(
                    selected_results=record[
                        "constrained"
                    ]["results"],
                    all_chunks=chunks
                )

                records.append({
                    "question": record["question"],
                    **result
                })

            report = {
                "experiment": (
                    "V3 Same-Section Neighbor Expansion"
                ),
                "filename": uploaded.name,
                "pdf_hash": pdf_hash,
                "chunk_count": len(chunks),
                "max_context_chars": (
                    MAX_CONTEXT_CHARS
                ),
                "max_extra_chunks": (
                    MAX_EXTRA_CHUNKS
                ),
                "records": records
            }

            OUTPUT_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            OUTPUT_PATH.write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2
                ),
                encoding="utf-8"
            )

            st.session_state[
                "neighbor_report"
            ] = report

    except Exception as exc:

        st.error(
            f"实验失败：{exc}"
        )

        st.stop()


# ==========================================
# 10. 展示结果
# ==========================================

report = st.session_state.get(
    "neighbor_report"
)

if (
    report is None
    or report["pdf_hash"] != pdf_hash
):
    st.stop()

st.success("邻近证据扩展实验完成！")

for index, record in enumerate(
    report["records"],
    start=1
):

    st.divider()

    st.subheader(
        f"问题 {index}"
    )

    st.write(record["question"])

    a, b, c = st.columns(3)

    a.metric(
        "扩展前证据数",
        record["before"]["source_count"]
    )

    b.metric(
        "扩展后证据数",
        record["after"]["source_count"]
    )

    c.metric(
        "扩展后字符数",
        record["after"]["context_chars"]
    )

    if record["anchors_preserved"]:
        st.success("原始证据全部保留。")
    else:
        st.error("存在证据缺失。")

    if not record["added_neighbors"]:
        st.warning(
            "本题没有添加邻近证据。"
        )

    for item in record["added_neighbors"]:

        with st.expander(
            f"新增证据 | "
            f"第 {item['page']} 页 | "
            f"{item['section']}"
        ):

            st.write(
                "扩展方向：",
                item["direction"]
            )

            st.write(
                "Chunk ID：",
                item["chunk_id"]
            )

            st.write(item["text"])


# ==========================================
# 11. 检查 Leapfrog 公式
# ==========================================

leapfrog_record = next(
    (
        record
        for record in report["records"]
        if "leapfrog integrator"
        in record["question"].lower()
    ),
    None
)

if leapfrog_record:

    st.divider()

    st.subheader("🔍 Leapfrog 核心公式检查")

    evidence_text = " ".join(
        source["text"]
        for source in leapfrog_record[
            "expanded_sources"
        ]
    )

    formula_markers = [
        "5.18",
        "5.19",
        "5.20"
    ]

    found = [
        marker
        for marker in formula_markers
        if marker in evidence_text
    ]

    st.write(
        "检测到的公式编号：",
        found
    )

    if len(found) == 3:
        st.success(
            "已在扩展上下文中找到"
            "Leapfrog 三条更新公式的编号。"
        )
    else:
        st.warning(
            "尚未找到全部公式编号，"
            "需要继续检查证据扩展范围。"
        )


# ==========================================
# 12. 下载报告
# ==========================================

st.divider()

st.download_button(
    "📥 下载邻近证据扩展报告",
    data=json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ).encode("utf-8"),
    file_name="pdf_v3_neighbor_expansion.json",
    mime="application/json"
)

st.caption(
    "仅测试已验证的 HMC 文档。"
    "实际公式可读性和答案质量仍需人工检查。"
)
