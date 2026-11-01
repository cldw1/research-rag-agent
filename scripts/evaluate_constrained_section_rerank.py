
import json
import re
from pathlib import Path
from collections import Counter


# ==========================================
# 1. 项目路径
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_NAMES = [
    "pdf_v3_section_rerank_5.json",
    "pdf_v3_section_rerank_5(1).json"
]

OUTPUT_PATH = (
    ROOT
    / "results"
    / "pdf_v3_constrained_rerank_5.json"
)


# ==========================================
# 2. 实验参数
# ==========================================

TOP_K = 5

SECTION_BONUS = 0.010

REFERENCE_PENALTY = 0.025

MAX_CHUNKS_PER_SECTION = 2


# ==========================================
# 3. 停用词
# ==========================================

# 这是当前 HMC 文档的试验词表，
# 尚不能直接视为任意领域论文的通用词表。

STOPWORDS = {
    "a", "an", "the", "is", "are",
    "was", "were", "do", "does",
    "did", "how", "what", "why",
    "when", "where", "which",
    "and", "or", "of", "in",
    "on", "at", "to", "for",
    "with", "by", "from",
    "this", "that", "these",
    "those", "it", "its",
    "be", "been", "being",
    "can", "could", "would",
    "should", "work", "useful",
    "handle", "affect",
    "hmc", "hamiltonian",
    "monte", "carlo",
    "distribution", "distributions",
    "method", "algorithm", "theory"
}


# ==========================================
# 4. 读取上一步实验结果
# ==========================================

def load_previous_report():

    for filename in INPUT_NAMES:

        path = ROOT / "results" / filename

        if path.exists():

            print("读取实验文件：", path)

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:
                data = json.load(f)

            if data.get("experiment") != (
                "V3 Section-Aware Reranking"
            ):
                raise ValueError(
                    "文件不是第57步的实验报告"
                )

            if data.get("chunk_count") != 247:
                raise ValueError(
                    "Chunk 数量与本次实验预期不一致"
                )

            return data

    raise FileNotFoundError(
        "未找到 pdf_v3_section_rerank_5.json，"
        "请确认文件位于 results 文件夹"
    )


# ==========================================
# 5. 提取关键词
# ==========================================

def keyword_tokens(text):

    text = str(text).casefold()

    # step size -> stepsize
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

        # 统一 tune / tuned / tuning
        if word in {
            "tune", "tuned",
            "tunes", "tuning"
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


# ==========================================
# 6. 判断是否询问参考文献
# ==========================================

def asks_for_references(question):

    text = question.casefold()

    phrases = [
        "reference",
        "bibliography",
        "citation",
        "cited papers",
        "cited works"
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


# ==========================================
# 7. 合并已有的候选 Chunk
# ==========================================

def build_candidate_pool(record):
    """
    从上一步报告中合并：

    A. 原始 V3 Top-5
    B. 扩大候选 Top-5
    C. 旧版重排序 Top-5

    按 Chunk ID 去重。

    注意：
    这里只包含之前导出的候选，
    不是整个 FAISS 索引的247个Chunk。
    """

    candidates = {}

    for method in [
        "baseline",
        "expanded",
        "reranked"
    ]:

        for item in record[method]["results"]:

            chunk_id = item["chunk_id"]

            if chunk_id not in candidates:
                candidates[chunk_id] = dict(item)

    return list(candidates.values())


# ==========================================
# 8. 为候选计算新分数
# ==========================================

def score_candidate(
    question,
    candidate
):

    question_words = keyword_tokens(
        question
    )

    section = candidate["section"]

    section_words = keyword_tokens(
        section
    )

    matched = sorted(
        question_words & section_words
    )

    # ======================================
    # 只要匹配到关键词就加一次分
    # 不再按匹配数量累加
    # ======================================

    bonus = (
        SECTION_BONUS
        if matched
        else 0.0
    )

    # ======================================
    # 参考文献降权
    # ======================================

    is_reference = (
        section.casefold()
        in {"references", "bibliography"}
    )

    penalty = 0.0

    if (
        is_reference
        and not asks_for_references(question)
    ):

        penalty = REFERENCE_PENALTY

    base_score = float(
        candidate["rrf_score"]
    )

    final_score = (
        base_score
        + bonus
        - penalty
    )

    return {
        **candidate,
        "matched_keywords": matched,
        "section_bonus": bonus,
        "reference_penalty": penalty,
        "constrained_score": final_score
    }


# ==========================================
# 9. 受约束的章节重排序
# ==========================================

def constrained_rerank(record):

    question = record["question"]

    baseline = record[
        "baseline"
    ]["results"]

    candidates = build_candidate_pool(
        record
    )

    scored = [
        score_candidate(
            question,
            candidate
        )
        for candidate in candidates
    ]

    # --------------------------------------
    # A. 按新分数排序
    # --------------------------------------

    scored.sort(
        key=lambda item: (
            -item["constrained_score"],
            -item["rrf_score"],
            item["chunk_id"]
        )
    )

    # --------------------------------------
    # B. 保留原始 Top-1
    # --------------------------------------

    anchor_id = baseline[0]["chunk_id"]

    anchor = next(
        item
        for item in scored
        if item["chunk_id"] == anchor_id
    )

    selected = [anchor]

    section_counts = Counter({
        anchor["section"]: 1
    })

    selected_ids = {
        anchor_id
    }

    # --------------------------------------
    # C. 从候选中依次选择
    # --------------------------------------

    for item in scored:

        if len(selected) >= TOP_K:
            break

        chunk_id = item["chunk_id"]

        section = item["section"]

        # 已选过，不重复添加
        if chunk_id in selected_ids:
            continue

        # 每个章节最多两个 Chunk
        if (
            section_counts[section]
            >= MAX_CHUNKS_PER_SECTION
        ):
            continue

        selected.append(item)

        selected_ids.add(
            chunk_id
        )

        section_counts[section] += 1

    # --------------------------------------
    # D. 最后重新按得分排列
    # --------------------------------------

    selected.sort(
        key=lambda item: (
            -item["constrained_score"],
            -item["rrf_score"],
            item["chunk_id"]
        )
    )

    return selected


# ==========================================
# 10. 整理实验输出
# ==========================================

def summarize(items):

    return {
        "chunk_count": len(items),

        "reference_hits": sum(
            item["section"].casefold()
            in {"references", "bibliography"}
            for item in items
        ),

        "max_same_section": max(
            Counter(
                item["section"]
                for item in items
            ).values(),
            default=0
        ),

        # 仅统计返回文本字符数，
        # 不等于最终 Prompt 的字符数
        "retrieved_text_chars": sum(
            len(item["text"])
            for item in items
        ),

        "results": items
    }


# ==========================================
# 11. 运行五道固定问题
# ==========================================

def main():

    print("=" * 65)
    print("第58步：受约束的章节重排序实验")
    print("=" * 65)

    previous = load_previous_report()

    records = []

    for index, record in enumerate(
        previous["records"],
        start=1
    ):

        baseline = record[
            "baseline"
        ]["results"]

        old_reranked = record[
            "reranked"
        ]["results"]

        constrained = constrained_rerank(
            record
        )

        baseline_anchor = baseline[0][
            "chunk_id"
        ]

        new_ids = {
            item["chunk_id"]
            for item in constrained
        }

        anchor_retained = (
            baseline_anchor in new_ids
        )

        candidate_count = len(
            build_candidate_pool(record)
        )

        result = {
            "question": record["question"],
            "candidate_pool_size": candidate_count,
            "anchor_retained": anchor_retained,
            "baseline": summarize(baseline),
            "previous_reranked": summarize(
                old_reranked
            ),
            "constrained": summarize(
                constrained
            )
        }

        records.append(result)

        print(
            f"\n问题 {index}: "
            f"{record['question']}"
        )

        print(
            f"候选池：{candidate_count} 个"
        )

        print(
            "原始 Top-1 保留：",
            anchor_retained
        )

        print(
            "单章节最多 Chunk：",
            result["constrained"][
                "max_same_section"
            ]
        )

        print(
            "参考文献命中：",
            result["constrained"][
                "reference_hits"
            ]
        )

        print("\n新的 Top-5 章节：")

        for rank, item in enumerate(
            constrained,
            start=1
        ):

            print(
                f"  {rank}. "
                f"{item['section']} "
                f"(第{item['page']}页)"
            )

        if len(constrained) < TOP_K:

            print(
                "提醒：受限于候选池或章节数量，"
                "本题不足5个结果。"
            )

    # ======================================
    # 12. 总结统计
    # ======================================

    all_anchors_retained = all(
        record["anchor_retained"]
        for record in records
    )

    max_section_count = max(
        record["constrained"][
            "max_same_section"
        ]
        for record in records
    )

    report = {
        "experiment": (
            "Constrained Section-Aware Reranking"
        ),
        "source_experiment": (
            "V3 Section-Aware Reranking"
        ),
        "candidate_pool_note": (
            "Union of previously exported "
            "baseline, expanded and reranked "
            "Top-5 only; not all indexed chunks."
        ),
        "section_bonus": SECTION_BONUS,
        "reference_penalty": REFERENCE_PENALTY,
        "max_chunks_per_section": (
            MAX_CHUNKS_PER_SECTION
        ),
        "top_k": TOP_K,
        "all_anchors_retained": (
            all_anchors_retained
        ),
        "max_section_count": (
            max_section_count
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

    print("\n" + "=" * 65)
    print("最终检查")
    print("=" * 65)

    print(
        "5道问题是否保留原始Top-1：",
        all_anchors_retained
    )

    print(
        "单章节最多Chunk：",
        max_section_count
    )

    print(
        "报告已保存：",
        OUTPUT_PATH
    )

    print(
        "\n第58步实验完成！"
    )


if __name__ == "__main__":
    main()
