
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 项目配置
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.pdf_chunker_v3 import (
    extract_pdf_pages,
    split_long_text,
    compact
)

from src.retrieval.uploaded_pdf_retriever import (
    UploadedPDFRetriever
)

from src.retrieval.bm25_retriever import tokenize

from src.generation.strict_context_builder import (
    build_strict_context
)


HEADING_PATH = (
    ROOT / "results" / "pdf_headings_filtered.json"
)

OUTPUT_PATH = (
    ROOT / "results" / "pdf_v3_retrieval_5_questions.json"
)

QUESTIONS = [
    "What is Hamiltonian Monte Carlo and why is it useful?",
    "How does the leapfrog integrator work in HMC?",
    "How are the step size and trajectory length tuned in HMC?",
    "How do mass matrices and correlations affect HMC?",
    "How does HMC handle multimodal distributions and tempering?"
]


# ==========================================
# 2. 加载同一个 BGE 模型
# ==========================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        device="cpu",
        cache_folder=str(
            ROOT / "models" / "embedding_cache"
        )
    )


# ==========================================
# 3. 文本匹配辅助函数
# ==========================================

def normalize(text):
    return " ".join(str(text).split())


def match_key(text):

    return normalize(
        unicodedata.normalize(
            "NFKC", str(text)
        )
    ).casefold()


def flatten_page(blocks):

    lines = []

    for block in blocks:
        for raw_line in block.splitlines():

            line = normalize(raw_line)

            if line:
                lines.append(line)

    return lines


# ==========================================
# 4. 章节位置匹配
# ==========================================

def heading_span(lines, index, heading):

    number = heading.get("section_number")
    title = heading["section_title"]

    if number is None:
        return (
            1 if match_key(lines[index])
            == match_key(title)
            else 0
        )

    complete = f"{number} {title}"

    if match_key(lines[index]) == match_key(complete):
        return 1

    if match_key(
        lines[index].rstrip(".")
    ) != match_key(number):
        return 0

    for count in (1, 2, 3):

        end = index + 1 + count

        if end > len(lines):
            break

        joined = " ".join(
            lines[index + 1:end]
        )

        if match_key(joined) == match_key(title):
            return count + 1

    return 0


def locate_headings(pages, headings):

    by_page = {}

    for heading in headings:

        page = int(heading["page"])

        by_page.setdefault(
            page, []
        ).append(heading)

    markers = {}
    matched = []
    unmatched = []

    for page_number, blocks in enumerate(
        pages, start=1
    ):

        lines = flatten_page(blocks)

        candidates = sorted(
            by_page.get(page_number, []),
            key=lambda x: x["y_position"]
        )

        events = {}
        start = 0

        for heading in candidates:

            found = False

            for index in range(
                start, len(lines)
            ):

                span = heading_span(
                    lines, index, heading
                )

                if span:

                    events[index] = heading["candidate"]

                    matched.append({
                        "page": page_number,
                        "section": heading["candidate"]
                    })

                    start = index + span
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
# 5. 构建 V3 Chunk
# ==========================================

def build_v3_chunks(
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
        events = markers.get(page_number, {})

        buffer = []
        chunk_index = 0

        def flush():

            nonlocal buffer, chunk_index

            if not buffer:
                return

            text = " ".join(buffer)

            section = (
                active_section
                or f"Page {page_number}"
            )

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
                    == "references"
                )
            })

            chunk_index += 1
            buffer = []

        for line_index, line in enumerate(lines):

            if line_index in events:

                flush()
                active_section = events[line_index]

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
    # 逐页检查文本完整性
    # ======================================

    problem_pages = []

    for page_number, blocks in enumerate(
        pages, start=1
    ):

        original = compact(
            "".join(blocks)
        )

        rebuilt = compact(
            "".join(
                item["text"]
                for item in chunks
                if item["page_number"]
                == page_number
            )
        )

        if original != rebuilt:

            problem_pages.append(
                page_number
            )

    if problem_pages:

        raise ValueError(
            "V3 文本完整性检查失败："
            f"{problem_pages}"
        )

    return chunks


# ==========================================
# 6. V3 Hybrid 检索器
# ==========================================

class V3PDFRetriever(UploadedPDFRetriever):

    def __init__(
        self,
        original,
        chunks,
        model
    ):

        self.model = model
        self.title = original.title
        self.paper_id = original.paper_id
        self.page_count = original.page_count
        self.chunks = chunks

        tokenized = [
            tokenize(item["text"])
            for item in self.chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized
        )

        texts = [
            item["text"]
            for item in self.chunks
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
# 7. 执行单题检索
# ==========================================

def evaluate(retriever, question):

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
        "reference_hits": sum(
            item.get(
                "section_name", ""
            ).casefold() == "references"
            for item in results
        ),
        "results": [
            {
                "rank": int(item["rank"]),
                "page": int(item["page_number"]),
                "section": item["section_name"],
                "chunk_id": item["chunk_id"],
                "chars": len(item["text"]),
                "rrf_score": float(
                    item["rrf_score"]
                ),
                "text": item["text"]
            }
            for item in results
        ]
    }


# ==========================================
# 8. Streamlit 页面
# ==========================================

st.set_page_config(
    page_title="V3 Retrieval Comparison",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 V3 结构感知检索实验")

st.caption(
    "原始 Chunk vs 带真实章节标签的 V3 Chunk"
)

st.info(
    "使用相同的 BGE、BM25、FAISS、RRF、"
    "Top-5 和 8000 字符上下文预算。"
    "不调用 DeepSeek。"
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
# 9. 运行实验
# ==========================================

if st.button(
    "开始 V3 检索对比",
    type="primary"
):

    try:

        if not HEADING_PATH.exists():

            raise FileNotFoundError(
                f"找不到章节文件：{HEADING_PATH}"
            )

        heading_data = json.loads(
            HEADING_PATH.read_text(
                encoding="utf-8"
            )
        )

        if uploaded.name != heading_data["filename"]:

            raise ValueError(
                "文件名与章节诊断结果不一致。"
                "请上传原来的 HandbookChapter5.pdf"
            )

        with st.spinner(
            "正在构建原始索引和 V3 索引..."
        ):

            model = load_model()

            pages = extract_pdf_pages(
                pdf_bytes
            )

            markers, matched, unmatched = locate_headings(
                pages,
                heading_data["headings"]
            )

            if unmatched:

                raise ValueError(
                    "存在未匹配章节："
                    f"{unmatched}"
                )

            original = UploadedPDFRetriever(
                pdf_bytes=pdf_bytes,
                filename=uploaded.name,
                model=model
            )

            chunks = build_v3_chunks(
                pages=pages,
                markers=markers,
                paper_id=original.paper_id,
                title=original.title
            )

            v3 = V3PDFRetriever(
                original=original,
                chunks=chunks,
                model=model
            )

            records = []

            for question in QUESTIONS:

                records.append({
                    "question": question,
                    "original": evaluate(
                        original,
                        question
                    ),
                    "v3": evaluate(
                        v3,
                        question
                    )
                })

            report = {
                "experiment": "V1 vs V3 PDF Retrieval",
                "filename": uploaded.name,
                "pdf_hash": pdf_hash,
                "matched_headings": len(matched),
                "original_chunk_count": len(
                    original.chunks
                ),
                "v3_chunk_count": len(
                    v3.chunks
                ),
                "text_integrity_passed": True,
                "top_k": 5,
                "context_budget": 8000,
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
                "v3_retrieval_report"
            ] = report

    except Exception as exc:

        st.error(
            f"实验失败：{exc}"
        )

        st.stop()


# ==========================================
# 10. 显示实验结果
# ==========================================

report = st.session_state.get(
    "v3_retrieval_report"
)

if (
    report is None
    or report["pdf_hash"] != pdf_hash
):
    st.stop()

st.success(
    "V3 检索对比完成，报告已保存！"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "匹配章节",
        report["matched_headings"]
    )

with col2:
    st.metric(
        "原始 Chunk",
        report["original_chunk_count"]
    )

with col3:
    st.metric(
        "V3 Chunk",
        report["v3_chunk_count"]
    )

st.divider()

for index, record in enumerate(
    report["records"],
    start=1
):

    st.subheader(
        f"问题 {index}"
    )

    st.write(
        record["question"]
    )

    left, right = st.columns(2)

    for column, method, name in [
        (left, "original", "原始分块"),
        (right, "v3", "V3 结构感知")
    ]:

        data = record[method]

        with column:

            st.markdown(
                f"**{name}**"
            )

            st.caption(
                f"实际上下文："
                f"{data['context_chars']} 字符"
            )

            st.caption(
                f"参考文献命中："
                f"{data['reference_hits']}"
            )

            for item in data["results"]:

                with st.expander(
                    f"排名 {item['rank']} | "
                    f"第 {item['page']} 页 | "
                    f"{item['chars']} 字符"
                ):

                    st.caption(
                        f"章节：{item['section']}"
                    )

                    st.caption(
                        f"RRF："
                        f"{item['rrf_score']:.5f}"
                    )

                    st.write(
                        item["text"]
                    )

    st.divider()


st.download_button(
    label="📥 下载 V3 检索对比 JSON",
    data=json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ).encode("utf-8"),
    file_name="pdf_v3_retrieval_5_questions.json",
    mime="application/json"
)

st.caption(
    "本次检索尚无人工标准答案，"
    "不能直接计算准确率。"
    "章节候选来自本篇 HMC 文档的专用标注，"
    "结果不能直接推广到任意 PDF。"
)
