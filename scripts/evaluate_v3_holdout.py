
import hashlib
import json
import re
import sys
from collections import Counter
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


DATASET_PATH = (
    ROOT / "results"
    / "pdf_v3_holdout_questions.json"
)

MAPPING_PATH = (
    ROOT / "results"
    / "pdf_v3_section_mapping_report.json"
)

PREVIOUS_PATH = (
    ROOT / "results"
    / "pdf_v3_retrieval_5_questions.json"
)

OUTPUT_PATH = (
    ROOT / "results"
    / "pdf_v3_holdout_evaluation.json"
)


# ==========================================
# 2. 固定参数，不根据测试集调参
# ==========================================

TOP_K = 5
SECTION_BONUS = 0.010
REFERENCE_PENALTY = 0.025
MAX_PER_SECTION = 2

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
    "hmc", "hamiltonian", "monte", "carlo",
    "distribution", "distributions",
    "method", "algorithm", "theory"
}


# ==========================================
# 3. 读取 JSON
# ==========================================

def read_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"找不到文件：{path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


# ==========================================
# 4. 读取 BGE 模型
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
# 5. PDF 文本行
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


def get_markers(mapping):

    markers = {}

    for event in mapping["matched_events"]:

        page = int(event["page"])
        index = int(event["line_index"])

        markers.setdefault(
            page, {}
        )[index] = event["section"]

    return markers


# ==========================================
# 6. 重建 V3 Chunk
# ==========================================

def build_chunks(pages, markers, paper_id, title):

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

            section = (
                active_section
                or f"Page {page_number}"
            )

            body = " ".join(buffer)

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
                "text": body,
                "char_count": len(body)
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

    # 逐页校验提取文本
    problem_pages = []

    for page_number, blocks in enumerate(
        pages, start=1
    ):

        original = compact("".join(blocks))

        rebuilt = compact(
            "".join(
                c["text"]
                for c in chunks
                if c["page_number"] == page_number
            )
        )

        if original != rebuilt:
            problem_pages.append(page_number)

    if problem_pages:
        raise ValueError(
            f"文本完整性失败：{problem_pages}"
        )

    return chunks


# ==========================================
# 7. V3 检索器
# ==========================================

class V3Retriever(UploadedPDFRetriever):

    def __init__(
        self,
        chunks,
        model,
        paper_id,
        title,
        page_count
    ):

        self.model = model
        self.paper_id = paper_id
        self.title = title
        self.page_count = page_count
        self.chunks = chunks

        tokenized = [
            tokenize(c["text"])
            for c in chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

        vectors = model.encode(
            [c["text"] for c in chunks],
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

        self.index.add(self.vectors)

        self.paper_rows = {
            paper_id: list(range(len(chunks)))
        }


# ==========================================
# 8. 第58步关键词评分规则
# ==========================================

def keyword_tokens(text):

    text = str(text).casefold()

    text = re.sub(
        r"\bstep[\s-]+size\b",
        "stepsize",
        text
    )

    words = re.findall(
        r"[a-z][a-z0-9]*",
        text
    )

    tokens = set()

    for word in words:

        if word in {
            "tune", "tuned", "tunes", "tuning"
        }:
            word = "tuning"

        elif (
            word.endswith("ies")
            and len(word) > 5
        ):
            word = word[:-3] + "y"

        elif (
            word.endswith("s")
            and len(word) > 5
        ):
            word = word[:-1]

        if word in STOPWORDS:
            continue

        if len(word) < 3:
            continue

        tokens.add(word)

    return tokens


def asks_for_references(question):

    text = question.casefold()

    return any(
        phrase in text
        for phrase in [
            "reference",
            "bibliography",
            "citation",
            "cited papers",
            "cited works"
        ]
    )


# ==========================================
# 9. 受约束章节重排序
# ==========================================

def constrained_rerank(
    question,
    baseline,
    candidates
):

    question_words = keyword_tokens(question)

    scored = []

    for item in candidates:

        section = item["section_name"]

        section_words = keyword_tokens(section)

        matched = sorted(
            question_words & section_words
        )

        bonus = (
            SECTION_BONUS
            if matched
            else 0.0
        )

        is_reference = (
            section.casefold()
            in {"references", "bibliography"}
        )

        penalty = (
            REFERENCE_PENALTY
            if (
                is_reference
                and not asks_for_references(question)
            )
            else 0.0
        )

        score = (
            float(item["rrf_score"])
            + bonus
            - penalty
        )

        scored.append({
            **item,
            "constrained_score": score,
            "matched_keywords": matched,
            "section_bonus": bonus,
            "reference_penalty": penalty
        })

    scored.sort(
        key=lambda x: (
            -x["constrained_score"],
            -x["rrf_score"],
            x["chunk_id"]
        )
    )

    # 保留原始 Hybrid Top-1
    anchor_id = baseline[0]["chunk_id"]

    anchor = next(
        item
        for item in scored
        if item["chunk_id"] == anchor_id
    )

    selected = [anchor]
    selected_ids = {anchor_id}

    counts = Counter({
        anchor["section_name"]: 1
    })

    for item in scored:

        if len(selected) >= TOP_K:
            break

        chunk_id = item["chunk_id"]
        section = item["section_name"]

        if chunk_id in selected_ids:
            continue

        if counts[section] >= MAX_PER_SECTION:
            continue

        selected.append(item)
        selected_ids.add(chunk_id)
        counts[section] += 1

    selected.sort(
        key=lambda x: (
            -x["constrained_score"],
            -x["rrf_score"],
            x["chunk_id"]
        )
    )

    return selected


# ==========================================
# 10. 章节级 Hit@5 / MRR@5
# ==========================================

def get_section_number(section_name):

    match = re.match(
        r"^(\d+(?:\.\d+)+)(?![\d.])",
        str(section_name).strip()
    )

    if match:
        return match.group(1)

    return None


def first_gold_rank(results, gold_section):

    for rank, item in enumerate(
        results[:TOP_K],
        start=1
    ):

        found_section = get_section_number(
            item["section_name"]
        )

        if found_section == gold_section:
            return rank

    return None


def calculate_metrics(records, method):

    ranks = [
        item[method]["gold_rank"]
        for item in records
    ]

    hits = sum(
        rank is not None
        for rank in ranks
    )

    reciprocal_ranks = [
        1.0 / rank if rank is not None else 0.0
        for rank in ranks
    ]

    n = len(ranks)

    return {
        "question_count": n,
        "hit_count": hits,
        "hit_at_5": hits / n,
        "mrr_at_5": (
            sum(reciprocal_ranks) / n
        )
    }


def serialize_results(results, gold_section):

    gold_rank = first_gold_rank(
        results, gold_section
    )

    top5 = []

    for rank, item in enumerate(
        results[:TOP_K],
        start=1
    ):

        top5.append({
            "rank": rank,
            "chunk_id": item["chunk_id"],
            "page": item["page_number"],
            "section": item["section_name"],
            "rrf_score": float(
                item["rrf_score"]
            ),
            "constrained_score": float(
                item.get(
                    "constrained_score",
                    item["rrf_score"]
                )
            ),
            "text": item["text"]
        })

    return {
        "gold_rank": gold_rank,
        "hit_at_5": gold_rank is not None,
        "top5": top5
    }


# ==========================================
# 11. Streamlit 页面
# ==========================================

st.set_page_config(
    page_title="V3 Holdout Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 第64步：V3 独立检索评测")

st.caption(
    "7 道新问题；比较原始 V3 Hybrid "
    "和受约束章节重排序。"
)

st.info(
    "冻结第58步的评分参数。"
    "本次不调用 DeepSeek、Cross-Encoder，"
    "也不修改正式 Agent。"
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
# 12. 开始评测
# ==========================================

if st.button(
    "开始 7 道题评测",
    type="primary"
):

    try:

        dataset = read_json(DATASET_PATH)
        mapping = read_json(MAPPING_PATH)
        previous = read_json(PREVIOUS_PATH)

        questions = dataset["questions"]

        if len(questions) != 7:
            raise ValueError(
                "独立测试集不是7道题"
            )

        if previous["pdf_hash"] != pdf_hash:
            raise ValueError(
                "上传的 PDF 与以前的实验文件不一致"
            )

        if uploaded.name != mapping["filename"]:
            raise ValueError(
                "PDF 文件名与章节映射不一致"
            )

        if not mapping["integrity_passed"]:
            raise ValueError(
                "章节映射未通过完整性检查"
            )

        if mapping["unmatched_headings"]:
            raise ValueError(
                "仍有章节标题未匹配"
            )

        known_sections = {
            event["section"]
            for event in mapping["matched_events"]
        }

        for case in questions:

            if (
                case["gold_section_title"]
                not in known_sections
            ):
                raise ValueError(
                    "标准章节未出现在映射中："
                    f"{case['id']}"
                )

        with st.spinner(
            "正在构建 V3 索引并评测..."
        ):

            pages = extract_pdf_pages(
                pdf_bytes
            )

            markers = get_markers(mapping)

            paper_id = (
                "upload_" + pdf_hash[:12]
            )

            chunks = build_chunks(
                pages=pages,
                markers=markers,
                paper_id=paper_id,
                title=Path(uploaded.name).stem
            )

            if (
                len(chunks)
                != mapping["chunk_count"]
            ):
                raise ValueError(
                    "V3 Chunk 数量发生变化："
                    f"{len(chunks)}"
                )

            model = load_model()

            retriever = V3Retriever(
                chunks=chunks,
                model=model,
                paper_id=paper_id,
                title=Path(uploaded.name).stem,
                page_count=len(pages)
            )

            records = []
            n = len(chunks)

            for case in questions:

                question = case["question"]
                gold = case["gold_section"]

                # A. 原始 V3 Hybrid
                baseline = retriever.search(
                    query=question,
                    paper_id=paper_id,
                    top_k=TOP_K,
                    candidate_k=50
                )

                # B. 完整候选池
                full_candidates = retriever.search(
                    query=question,
                    paper_id=paper_id,
                    top_k=n,
                    candidate_k=n
                )

                if not baseline:
                    raise ValueError(
                        "原始检索没有返回结果："
                        f"{case['id']}"
                    )

                if not full_candidates:
                    raise ValueError(
                        "完整候选检索没有返回结果："
                        f"{case['id']}"
                    )

                reranked = constrained_rerank(
                    question,
                    baseline,
                    full_candidates
                )

                records.append({
                    "id": case["id"],
                    "question": question,
                    "gold_section": gold,
                    "gold_section_title": (
                        case["gold_section_title"]
                    ),
                    "full_candidate_count": len(
                        full_candidates
                    ),
                    "baseline": serialize_results(
                        baseline, gold
                    ),
                    "constrained": serialize_results(
                        reranked, gold
                    )
                })

            baseline_metrics = calculate_metrics(
                records, "baseline"
            )

            constrained_metrics = calculate_metrics(
                records, "constrained"
            )

            report = {
                "experiment": (
                    "V3 Holdout Section Retrieval Evaluation"
                ),
                "filename": uploaded.name,
                "pdf_hash": pdf_hash,
                "chunk_count": len(chunks),
                "question_count": len(records),
                "annotation_status": dataset[
                    "annotation_status"
                ],
                "evaluation_level": "section",
                "top_k": TOP_K,
                "section_bonus": SECTION_BONUS,
                "reference_penalty": (
                    REFERENCE_PENALTY
                ),
                "max_per_section": MAX_PER_SECTION,
                "candidate_pool_note": (
                    "All indexed V3 chunks used "
                    "as reranking candidates."
                ),
                "baseline_metrics": (
                    baseline_metrics
                ),
                "constrained_metrics": (
                    constrained_metrics
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
                "holdout_report"
            ] = report

    except Exception as exc:

        st.error(f"评测失败：{exc}")
        st.stop()


# ==========================================
# 13. 展示结果
# ==========================================

report = st.session_state.get(
    "holdout_report"
)

if (
    report is None
    or report["pdf_hash"] != pdf_hash
):
    st.stop()

st.success(
    "7 道题评测已完成，报告保存成功！"
)

a, b, c, d = st.columns(4)

a.metric(
    "原始 V3 Hit@5",
    f"{report['baseline_metrics']['hit_at_5']:.3f}"
)

b.metric(
    "重排序 Hit@5",
    f"{report['constrained_metrics']['hit_at_5']:.3f}"
)

c.metric(
    "原始 V3 MRR@5",
    f"{report['baseline_metrics']['mrr_at_5']:.3f}"
)

d.metric(
    "重排序 MRR@5",
    f"{report['constrained_metrics']['mrr_at_5']:.3f}"
)

rows = []

for item in report["records"]:

    rows.append({
        "题号": item["id"],
        "标准章节": item["gold_section"],
        "原始命中排名": (
            item["baseline"]["gold_rank"]
            or "未命中"
        ),
        "重排序命中排名": (
            item["constrained"]["gold_rank"]
            or "未命中"
        )
    })

st.subheader("逐题评测结果")
st.table(rows)

st.divider()

for item in report["records"]:

    with st.expander(
        f"{item['id']} | {item['question']}"
    ):

        left, right = st.columns(2)

        for column, key, label in [
            (left, "baseline", "原始 V3"),
            (right, "constrained", "受约束重排序")
        ]:

            with column:

                st.markdown(f"**{label}**")

                rank = item[key]["gold_rank"]

                st.write(
                    "标准章节排名：",
                    rank if rank else "Top-5 未命中"
                )

                for result in item[key]["top5"]:

                    st.markdown(
                        f"**Top {result['rank']} | "
                        f"第 {result['page']} 页**"
                    )

                    st.caption(
                        result["section"]
                    )

                    st.write(
                        result["text"]
                    )

                    st.divider()


st.download_button(
    "📥 下载独立评测 JSON",
    data=json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ).encode("utf-8"),
    file_name="pdf_v3_holdout_evaluation.json",
    mime="application/json"
)

st.caption(
    "本轮只计算章节级命中指标。"
    "标准标签仍需人工核验；"
    "命中章节不等于答案正确。"
)
