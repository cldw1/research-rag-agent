
import hashlib
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import streamlit as st

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


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

from src.retrieval.bm25_retriever import tokenize

from src.retrieval.uploaded_pdf_retriever import (
    UploadedPDFRetriever
)

from src.generation.strict_context_builder import (
    build_strict_context
)


# ==========================================
# 2. 实验文件
# ==========================================

MAPPING_PATH = (
    ROOT
    / "results"
    / "pdf_v3_section_mapping_report.json"
)

PREVIOUS_PATH = (
    ROOT
    / "results"
    / "pdf_v3_retrieval_5_questions.json"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "pdf_v3_section_rerank_5.json"
)


# ==========================================
# 3. 实验参数
# ==========================================

MIN_CHARS = 450
TARGET_CHARS = 800
MAX_CHARS = 1000

SECTION_BONUS = 0.012
REFERENCE_PENALTY = 0.025

QUESTIONS = [
    "What is Hamiltonian Monte Carlo and why is it useful?",
    "How does the leapfrog integrator work in HMC?",
    "How are the step size and trajectory length tuned in HMC?",
    "How do mass matrices and correlations affect HMC?",
    "How does HMC handle multimodal distributions and tempering?"
]


# ==========================================
# 4. 加载 BGE
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
# 5. 文本处理
# ==========================================

def normalize(text):

    return " ".join(
        str(text).split()
    )


def flatten_page(blocks):

    lines = []

    for block in blocks:

        for raw_line in block.splitlines():

            line = normalize(raw_line)

            if line:
                lines.append(line)

    return lines


# ==========================================
# 6. 从已验证的报告读取章节位置
# ==========================================

def load_section_markers():

    if not MAPPING_PATH.exists():

        raise FileNotFoundError(
            f"找不到章节映射报告：{MAPPING_PATH}"
        )

    report = json.loads(
        MAPPING_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not report.get("integrity_passed"):

        raise ValueError(
            "之前的章节映射未通过文本完整性检查"
        )

    if report["unmatched_headings"]:

        raise ValueError(
            "章节映射中仍有未匹配的标题"
        )

    markers = {}

    for event in report["matched_events"]:

        page = int(event["page"])
        line_index = int(event["line_index"])

        markers.setdefault(
            page, {}
        )[line_index] = event["section"]

    return report, markers


# ==========================================
# 7. 根据章节位置重新构建 V3 Chunk
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

        events = markers.get(
            page_number, {}
        )

        buffer = []
        chunk_index = 0

        # ----------------------------------
        # 保存一个 Chunk
        # ----------------------------------

        def flush():

            nonlocal buffer
            nonlocal chunk_index

            if not buffer:
                return

            text = " ".join(buffer)

            section = (
                active_section
                or f"Page {page_number}"
            )

            chunk = {
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
            }

            chunks.append(chunk)

            chunk_index += 1
            buffer = []

        # ----------------------------------
        # 遍历 PDF 行
        # ----------------------------------

        for line_index, line in enumerate(lines):

            if line_index in events:

                flush()

                active_section = events[
                    line_index
                ]

            pieces = split_long_text(
                line,
                limit=MAX_CHARS
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

        flush()

    # ======================================
    # 逐页文本完整性检查
    # ======================================

    problems = []

    for page_number, blocks in enumerate(
        pages,
        start=1
    ):

        original_text = compact(
            "".join(blocks)
        )

        rebuilt_text = compact(
            "".join(
                chunk["text"]
                for chunk in chunks
                if chunk["page_number"]
                == page_number
            )
        )

        if original_text != rebuilt_text:

            problems.append(page_number)

    if problems:

        raise ValueError(
            "V3 文本完整性检查失败："
            f"{problems}"
        )

    return chunks


# ==========================================
# 8. 构建 V3 Hybrid 检索器
# ==========================================

class V3Retriever(UploadedPDFRetriever):

    def __init__(
        self,
        chunks,
        model,
        title,
        paper_id,
        page_count
    ):

        self.model = model
        self.title = title
        self.paper_id = paper_id
        self.page_count = page_count
        self.chunks = chunks

        # BM25
        tokenized = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

        # Dense
        vectors = model.encode(
            [chunk["text"] for chunk in chunks],
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        self.vectors = np.ascontiguousarray(
            vectors,
            dtype=np.float32
        )

        # FAISS
        self.index = faiss.IndexFlatIP(
            self.vectors.shape[1]
        )

        self.index.add(self.vectors)

        self.paper_rows = {
            paper_id: list(
                range(len(chunks))
            )
        }


# ==========================================
# 9. 从问题和章节标题中提取关键词
# ==========================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was",
    "were", "do", "does", "did", "how",
    "what", "why", "when", "where", "which",
    "and", "or", "of", "in", "on", "at",
    "to", "for", "with", "by", "from",
    "this", "that", "these", "those",
    "it", "its", "be", "been", "being",
    "can", "could", "would", "should",
    "work", "useful", "handle", "affect",
    "hmc", "hamiltonian", "monte", "carlo"
}


def keyword_tokens(text):

    text = str(text).casefold()

    text = re.sub(
        r"\bstep[\s\-]+size\b",
        "stepsize",
        text
    )

    words = re.findall(
        r"[a-z][a-z0-9]*",
        text
    )

    normalized = []

    for word in words:

        # 简单处理常见英文词形
        if word.endswith("ies") and len(word) > 5:
            word = word[:-3] + "y"

        elif word.endswith("s") and len(word) > 5:
            word = word[:-1]

        if word == "tuned":
            word = "tuning"

        if word in STOPWORDS:
            continue

        if len(word) < 3:
            continue

        normalized.append(word)

    return set(normalized)


# ==========================================
# 10. 判断是否是参考文献类问题
# ==========================================

def asks_for_references(question):

    text = question.casefold()

    reference_phrases = [
        "references",
        "bibliography",
        "citations",
        "cited papers",
        "cited works",
        "reference list"
    ]

    return any(
        phrase in text
        for phrase in reference_phrases
    )


# ==========================================
# 11. 章节感知重排序
# ==========================================

def section_rerank(
    question,
    candidates,
    chunk_lookup,
    top_k=5
):

    question_words = keyword_tokens(
        question
    )

    allow_references = asks_for_references(
        question
    )

    reranked = []

    for item in candidates:

        chunk_id = item["chunk_id"]

        chunk = chunk_lookup[chunk_id]

        section = chunk["section_name"]

        section_words = keyword_tokens(
            section
        )

        # ----------------------------------
        # A. 章节关键词匹配
        # ----------------------------------

        matched_words = sorted(
            question_words & section_words
        )

        match_count = min(
            len(matched_words),
            2
        )

        section_bonus = (
            SECTION_BONUS * match_count
        )

        # ----------------------------------
        # B. 参考文献降权
        # ----------------------------------

        reference_penalty = 0.0

        if (
            chunk["is_reference"]
            and not allow_references
        ):

            reference_penalty = (
                REFERENCE_PENALTY
            )

        # ----------------------------------
        # C. 最终分数
        # ----------------------------------

        score = (
            float(item["rrf_score"])
            + section_bonus
            - reference_penalty
        )

        record = dict(item)

        record["rerank_score"] = score

        record["matched_keywords"] = (
            matched_words
        )

        record["section_bonus"] = (
            section_bonus
        )

        record["reference_penalty"] = (
            reference_penalty
        )

        record["is_reference"] = (
            chunk["is_reference"]
        )

        reranked.append(record)

    reranked.sort(
        key=lambda item: (
            -item["rerank_score"],
            -item["rrf_score"],
            item["chunk_id"]
        )
    )

    return reranked[:top_k]


# ==========================================
# 12. 整理单种方案的结果
# ==========================================

def summarize_results(
    results,
    chunk_lookup
):

    context, sources = build_strict_context(
        results,
        max_chars=8000
    )

    return {
        "context_chars": len(context),
        "source_count": len(sources),
        "reference_hits": sum(
            chunk_lookup[
                item["chunk_id"]
            ]["is_reference"]
            for item in results
        ),
        "results": [
            {
                "rank": rank,
                "chunk_id": item["chunk_id"],
                "page": int(
                    chunk_lookup[
                        item["chunk_id"]
                    ]["page_number"]
                ),
                "section": item["section_name"],
                "rrf_score": float(
                    item["rrf_score"]
                ),
                "rerank_score": float(
                    item.get(
                        "rerank_score",
                        item["rrf_score"]
                    )
                ),
                "matched_keywords": item.get(
                    "matched_keywords", []
                ),
                "reference_penalty": float(
                    item.get(
                        "reference_penalty", 0
                    )
                ),
                "text": item["text"]
            }
            for rank, item in enumerate(
                results,
                start=1
            )
        ]
    }


# ==========================================
# 13. Streamlit 页面
# ==========================================

st.set_page_config(
    page_title="V3 Section Reranking",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 V3 章节感知重排序实验")

st.caption(
    "原始 V3 vs 扩大候选 vs 章节重排序"
)

st.info(
    "本次只研究检索排序，"
    "不调用 DeepSeek，也不修改正式 app.py。"
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
# 14. 运行实验
# ==========================================

if st.button(
    "开始章节重排序实验",
    type="primary"
):

    try:

        mapping_report, markers = (
            load_section_markers()
        )

        if uploaded.name != mapping_report["filename"]:

            raise ValueError(
                "上传的文件名与章节映射报告不一致"
            )

        # 如果上一轮报告存在，
        # 使用 SHA256 确认是同一份 PDF。
        if PREVIOUS_PATH.exists():

            previous = json.loads(
                PREVIOUS_PATH.read_text(
                    encoding="utf-8"
                )
            )

            if previous.get("pdf_hash") != pdf_hash:

                raise ValueError(
                    "上传 PDF 与第 56 步使用的 "
                    "PDF 内容不一致"
                )

        with st.spinner(
            "正在构建 V3 索引并比较排序..."
        ):

            model = load_model()

            pages = extract_pdf_pages(
                pdf_bytes
            )

            chunks = build_v3_chunks(
                pages=pages,
                markers=markers,
                paper_id=(
                    "upload_" + pdf_hash[:12]
                ),
                title=Path(uploaded.name).stem
            )

            expected_chunks = (
                mapping_report["chunk_count"]
            )

            if len(chunks) != expected_chunks:

                raise ValueError(
                    "V3 Chunk 数量与第 55 步不一致："
                    f"{len(chunks)} != "
                    f"{expected_chunks}"
                )

            retriever = V3Retriever(
                chunks=chunks,
                model=model,
                title=Path(uploaded.name).stem,
                paper_id=(
                    "upload_" + pdf_hash[:12]
                ),
                page_count=len(pages)
            )

            lookup = {
                item["chunk_id"]: item
                for item in chunks
            }

            n = len(chunks)

            records = []

            # ----------------------------------
            # 对同一道问题比较三种方法
            # ----------------------------------

            for question in QUESTIONS:

                # A. 之前的 V3 检索
                baseline = retriever.search(
                    query=question,
                    paper_id=retriever.paper_id,
                    top_k=5,
                    candidate_k=50
                )

                # B. 扩大候选范围
                full_candidates = retriever.search(
                    query=question,
                    paper_id=retriever.paper_id,
                    top_k=n,
                    candidate_k=n
                )

                expanded = full_candidates[:5]

                # C. 章节重排序
                reranked = section_rerank(
                    question=question,
                    candidates=full_candidates,
                    chunk_lookup=lookup,
                    top_k=5
                )

                records.append({
                    "question": question,
                    "baseline": summarize_results(
                        baseline, lookup
                    ),
                    "expanded": summarize_results(
                        expanded, lookup
                    ),
                    "reranked": summarize_results(
                        reranked, lookup
                    )
                })

            report = {
                "experiment": (
                    "V3 Section-Aware Reranking"
                ),
                "filename": uploaded.name,
                "pdf_hash": pdf_hash,
                "chunk_count": n,
                "matched_headings": len(
                    mapping_report["matched_events"]
                ),
                "section_bonus": SECTION_BONUS,
                "reference_penalty": (
                    REFERENCE_PENALTY
                ),
                "top_k": 5,
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
                "section_rerank_report"
            ] = report

    except Exception as exc:

        st.error(
            f"实验失败：{exc}"
        )

        st.stop()


# ==========================================
# 15. 展示实验结果
# ==========================================

report = st.session_state.get(
    "section_rerank_report"
)

if (
    report is None
    or report["pdf_hash"] != pdf_hash
):
    st.stop()


st.success(
    "章节重排序实验完成，报告已保存！"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "V3 Chunk 数量",
        report["chunk_count"]
    )

with col2:

    st.metric(
        "章节匹配数量",
        report["matched_headings"]
    )


# ==========================================
# 16. 逐题对比
# ==========================================

for index, record in enumerate(
    report["records"],
    start=1
):

    st.divider()

    st.subheader(
        f"问题 {index}"
    )

    st.write(
        record["question"]
    )

    columns = st.columns(3)

    methods = [
        (
            columns[0],
            "baseline",
            "A：原始 V3"
        ),
        (
            columns[1],
            "expanded",
            "B：扩大候选"
        ),
        (
            columns[2],
            "reranked",
            "C：章节重排序"
        )
    ]

    for column, key, title in methods:

        data = record[key]

        with column:

            st.markdown(
                f"**{title}**"
            )

            st.caption(
                f"上下文："
                f"{data['context_chars']} 字符"
            )

            st.caption(
                f"参考文献："
                f"{data['reference_hits']} 个"
            )

            for item in data["results"]:

                with st.expander(
                    f"Top {item['rank']} | "
                    f"第 {item['page']} 页"
                ):

                    st.write(
                        "章节：",
                        item["section"]
                    )

                    st.write(
                        "RRF：",
                        round(
                            item["rrf_score"], 5
                        )
                    )

                    if key == "reranked":

                        st.write(
                            "重排序得分：",
                            round(
                                item["rerank_score"], 5
                            )
                        )

                        st.write(
                            "匹配关键词：",
                            item["matched_keywords"]
                        )

                    st.write(
                        item["text"]
                    )


# ==========================================
# 17. 下载实验结果
# ==========================================

st.divider()

st.download_button(
    "📥 下载章节重排序实验 JSON",
    data=json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ).encode("utf-8"),
    file_name="pdf_v3_section_rerank_5.json",
    mime="application/json"
)

st.caption(
    "本实验基于已人工筛选的 HMC 章节映射。"
    "章节奖励分数是启发式设计，"
    "结果尚不能直接视为检索准确率。"
)
