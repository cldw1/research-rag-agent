
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

from src.retrieval.retrieval_strategy import (
    retrieve_with_strategy
)


MAPPING_PATH = (
    ROOT / "results"
    / "pdf_v3_section_mapping_report.json"
)

PREVIOUS_PATH = (
    ROOT / "results"
    / "pdf_v3_holdout_evaluation.json"
)

OUTPUT_PATH = (
    ROOT / "results"
    / "pdf_v3_strategy_real_pdf_regression.json"
)


# ==========================================
# 2. 加载 JSON
# ==========================================

def read_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"缺少文件：{path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


# ==========================================
# 3. 加载 BGE
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
# 4. PDF 文本处理
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
# 5. 重建 V3 Chunk
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
    # 逐页检查是否丢失文本
    # ======================================

    for page_number, blocks in enumerate(
        pages,
        start=1
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

            raise ValueError(
                f"第{page_number}页文本完整性失败"
            )

    return chunks


# ==========================================
# 6. V3 检索器
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
            tokenize(item["text"])
            for item in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized
        )

        vectors = model.encode(
            [item["text"] for item in chunks],
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
            paper_id: list(
                range(len(chunks))
            )
        }


# ==========================================
# 7. 计算章节排名
# ==========================================

def get_section_number(section):

    match = re.match(
        r"^(\d+(?:\.\d+)+)(?![\d.])",
        str(section).strip()
    )

    return (
        match.group(1)
        if match
        else None
    )


def gold_rank(results, gold_section):

    for rank, item in enumerate(
        results[:5],
        start=1
    ):

        found = get_section_number(
            item["section_name"]
        )

        if found == gold_section:
            return rank

    return None


# ==========================================
# 8. 计算 Hit@5 和 MRR@5
# ==========================================

def metrics(ranks):

    n = len(ranks)

    hits = sum(
        rank is not None
        for rank in ranks
    )

    mrr = sum(
        1 / rank
        for rank in ranks
        if rank is not None
    ) / n

    return {
        "hit_count": hits,
        "hit_at_5": hits / n,
        "mrr_at_5": mrr
    }


# ==========================================
# 9. 精简输出
# ==========================================

def serialize(results):

    return [
        {
            "rank": rank,
            "chunk_id": item["chunk_id"],
            "page": item["page_number"],
            "section": item["section_name"],
            "rrf_score": float(
                item["rrf_score"]
            ),
            "text": item["text"]
        }
        for rank, item in enumerate(
            results,
            start=1
        )
    ]


def get_ids(results):

    return [
        item["chunk_id"]
        for item in results
    ]


# ==========================================
# 10. Streamlit 页面
# ==========================================

st.set_page_config(
    page_title="V3 Strategy Regression",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 第68步：真实 PDF 检索回归测试")

st.caption(
    "验证第67步新接口是否能复现"
    "第64步的真实PDF检索结果。"
)

st.info(
    "原始 Hybrid 仍为默认模式。"
    "本实验不调用 DeepSeek，"
    "也不修改正式 app.py。"
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
# 11. 执行真实检索
# ==========================================

if st.button(
    "开始真实 PDF 回归测试",
    type="primary"
):

    try:

        mapping = read_json(
            MAPPING_PATH
        )

        previous = read_json(
            PREVIOUS_PATH
        )

        if pdf_hash != previous["pdf_hash"]:

            raise ValueError(
                "PDF 与第64步文件不一致"
            )

        if uploaded.name != mapping["filename"]:

            raise ValueError(
                "PDF 文件名与映射不一致"
            )

        if not mapping["integrity_passed"]:

            raise ValueError(
                "映射完整性检查未通过"
            )

        if mapping["unmatched_headings"]:

            raise ValueError(
                "映射仍有未匹配章节"
            )

        with st.spinner(
            "构建真实 V3 索引并检查两种策略..."
        ):

            pages = extract_pdf_pages(
                pdf_bytes
            )

            paper_id = (
                "upload_" + pdf_hash[:12]
            )

            chunks = build_chunks(
                pages=pages,
                markers=get_markers(mapping),
                paper_id=paper_id,
                title=Path(uploaded.name).stem
            )

            if (
                len(chunks)
                != mapping["chunk_count"]
            ):

                raise ValueError(
                    "Chunk 数量不一致："
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

            for case in previous["records"]:

                question = case["question"]
                gold = case["gold_section"]

                # A. 新接口的原始检索
                hybrid = retrieve_with_strategy(
                    retriever=retriever,
                    query=question,
                    paper_id=paper_id,
                    strategy="hybrid",
                    top_k=5
                )

                # B. 新接口的章节重排序
                section_aware = retrieve_with_strategy(
                    retriever=retriever,
                    query=question,
                    paper_id=paper_id,
                    strategy="section_aware",
                    top_k=5
                )

                # 第64步保存的真实结果
                old_hybrid_ids = [
                    item["chunk_id"]
                    for item in case[
                        "baseline"
                    ]["top5"]
                ]

                old_section_ids = [
                    item["chunk_id"]
                    for item in case[
                        "constrained"
                    ]["top5"]
                ]

                # 检查新旧 Top-5 是否一致
                hybrid_same = (
                    get_ids(hybrid)
                    == old_hybrid_ids
                )

                section_same = (
                    get_ids(section_aware)
                    == old_section_ids
                )

                anchor_retained = (
                    bool(hybrid)
                    and any(
                        item["chunk_id"]
                        == hybrid[0]["chunk_id"]
                        for item in section_aware
                    )
                )

                section_counts = Counter(
                    item["section_name"]
                    for item in section_aware
                )

                section_limit_passed = (
                    max(
                        section_counts.values(),
                        default=0
                    ) <= 2
                )

                new_hybrid_rank = gold_rank(
                    hybrid,
                    gold
                )

                new_section_rank = gold_rank(
                    section_aware,
                    gold
                )

                records.append({
                    "id": case["id"],
                    "question": question,
                    "gold_section": gold,
                    "hybrid_same_as_step64": (
                        hybrid_same
                    ),
                    "section_same_as_step64": (
                        section_same
                    ),
                    "anchor_retained": (
                        anchor_retained
                    ),
                    "section_limit_passed": (
                        section_limit_passed
                    ),
                    "hybrid_gold_rank": (
                        new_hybrid_rank
                    ),
                    "section_gold_rank": (
                        new_section_rank
                    ),
                    "hybrid": serialize(hybrid),
                    "section_aware": serialize(
                        section_aware
                    )
                })

            hybrid_metrics = metrics([
                item["hybrid_gold_rank"]
                for item in records
            ])

            section_metrics = metrics([
                item["section_gold_rank"]
                for item in records
            ])

            # 检查指标与第64步一致
            expected_hybrid = previous[
                "baseline_metrics"
            ]

            expected_section = previous[
                "constrained_metrics"
            ]

            metrics_match = all([
                abs(
                    hybrid_metrics["hit_at_5"]
                    - expected_hybrid["hit_at_5"]
                ) < 1e-9,

                abs(
                    hybrid_metrics["mrr_at_5"]
                    - expected_hybrid["mrr_at_5"]
                ) < 1e-9,

                abs(
                    section_metrics["hit_at_5"]
                    - expected_section["hit_at_5"]
                ) < 1e-9,

                abs(
                    section_metrics["mrr_at_5"]
                    - expected_section["mrr_at_5"]
                ) < 1e-9
            ])

            all_rankings_match = all(
                item["hybrid_same_as_step64"]
                and item["section_same_as_step64"]
                for item in records
            )

            all_constraints_pass = all(
                item["anchor_retained"]
                and item["section_limit_passed"]
                for item in records
            )

            report = {
                "experiment": (
                    "Real V3 Retrieval Strategy Regression"
                ),
                "filename": uploaded.name,
                "pdf_hash": pdf_hash,
                "chunk_count": len(chunks),
                "question_count": len(records),
                "hybrid_metrics": hybrid_metrics,
                "section_metrics": section_metrics,
                "metrics_match_step64": (
                    metrics_match
                ),
                "all_rankings_match_step64": (
                    all_rankings_match
                ),
                "all_constraints_pass": (
                    all_constraints_pass
                ),
                "regression_passed": (
                    metrics_match
                    and all_rankings_match
                    and all_constraints_pass
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
                "real_regression_report"
            ] = report

    except Exception as exc:

        st.error(f"测试失败：{exc}")
        st.stop()


# ==========================================
# 12. 展示结果
# ==========================================

report = st.session_state.get(
    "real_regression_report"
)

if (
    report is None
    or report["pdf_hash"] != pdf_hash
):
    st.stop()


if report["regression_passed"]:

    st.success(
        "真实PDF回归测试通过！"
    )

else:

    st.error(
        "部分结果与第64步不一致，"
        "请查看下方明细。"
    )


a, b, c, d = st.columns(4)

a.metric(
    "Hybrid Hit@5",
    f"{report['hybrid_metrics']['hit_at_5']:.3f}"
)

b.metric(
    "Section-Aware Hit@5",
    f"{report['section_metrics']['hit_at_5']:.3f}"
)

c.metric(
    "Hybrid MRR@5",
    f"{report['hybrid_metrics']['mrr_at_5']:.3f}"
)

d.metric(
    "Section-Aware MRR@5",
    f"{report['section_metrics']['mrr_at_5']:.3f}"
)


st.subheader("回归检查")

st.write(
    "Top-5 排名完全一致：",
    report["all_rankings_match_step64"]
)

st.write(
    "四项指标一致：",
    report["metrics_match_step64"]
)

st.write(
    "章节约束全部通过：",
    report["all_constraints_pass"]
)


rows = []

for case in report["records"]:

    rows.append({
        "题号": case["id"],
        "Hybrid Top-5 一致": (
            case["hybrid_same_as_step64"]
        ),
        "Section-Aware Top-5 一致": (
            case["section_same_as_step64"]
        ),
        "原始命中排名": (
            case["hybrid_gold_rank"]
            or "未命中"
        ),
        "章节命中排名": (
            case["section_gold_rank"]
            or "未命中"
        )
    })

st.table(rows)


st.download_button(
    "下载真实 PDF 回归报告",
    data=json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ).encode("utf-8"),
    file_name="pdf_v3_strategy_real_pdf_regression.json",
    mime="application/json"
)
