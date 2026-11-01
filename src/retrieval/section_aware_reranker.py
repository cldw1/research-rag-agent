
import re
from collections import Counter


# ==========================================
# 默认实验参数
# ==========================================

DEFAULT_SECTION_BONUS = 0.010
DEFAULT_REFERENCE_PENALTY = 0.025
DEFAULT_MAX_PER_SECTION = 2


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


def keyword_tokens(text):
    """
    将英文文本转换为关键词集合。
    保留之前实验使用的基本归一化规则。
    """

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

        if len(word) >= 3:
            tokens.add(word)

    return tokens


def asks_for_references(query):

    text = str(query).casefold()

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


def constrained_section_rerank(
    query,
    baseline_results,
    candidates,
    top_k=5,
    section_bonus=DEFAULT_SECTION_BONUS,
    reference_penalty=DEFAULT_REFERENCE_PENALTY,
    max_per_section=DEFAULT_MAX_PER_SECTION
):
    """
    对同一篇论文中的候选 Chunk 进行重排序。

    参数：
        query: 用户问题
        baseline_results: 原始 Hybrid 排名结果
        candidates: 更大的候选列表
        top_k: 最终返回数量

    每个候选需要包含：
        chunk_id
        section_name
        rrf_score

    返回：
        带 constrained_score 和新 rank 的结果列表

    核心约束：
        1. 原始 Top-1 必须保留
        2. 每个章节最多 max_per_section 个 Chunk
        3. 参考文献在普通问题中降权
    """

    if top_k < 1:
        raise ValueError(
            "top_k 必须大于等于1"
        )

    if max_per_section < 1:
        raise ValueError(
            "max_per_section 必须大于等于1"
        )

    if not baseline_results:
        return []

    # ======================================
    # 1. 构建候选池并去重
    # ======================================

    pool = {}

    for item in candidates:

        chunk_id = item["chunk_id"]

        if chunk_id not in pool:
            pool[chunk_id] = dict(item)

    # 保证原始第一名必定存在
    anchor = baseline_results[0]

    anchor_id = anchor["chunk_id"]

    if anchor_id not in pool:
        pool[anchor_id] = dict(anchor)

    # ======================================
    # 2. 计算问题关键词
    # ======================================

    query_words = keyword_tokens(
        query
    )

    allow_references = asks_for_references(
        query
    )

    scored = []

    # ======================================
    # 3. 对每个 Chunk 加权
    # ======================================

    for item in pool.values():

        section = item.get(
            "section_name", ""
        )

        section_words = keyword_tokens(
            section
        )

        matched = sorted(
            query_words & section_words
        )

        bonus = (
            section_bonus
            if matched
            else 0.0
        )

        is_reference = (
            bool(item.get("is_reference", False))
            or section.casefold()
            in {"references", "bibliography"}
        )

        penalty = (
            reference_penalty
            if is_reference
            and not allow_references
            else 0.0
        )

        base_score = float(
            item["rrf_score"]
        )

        final_score = (
            base_score
            + bonus
            - penalty
        )

        scored.append({
            **item,
            "constrained_score": final_score,
            "matched_keywords": matched,
            "section_bonus": bonus,
            "reference_penalty": penalty
        })

    # ======================================
    # 4. 按新分数排序
    # ======================================

    def sort_key(item):

        return (
            -item["constrained_score"],
            -float(item["rrf_score"]),
            item["chunk_id"]
        )

    scored.sort(key=sort_key)

    # ======================================
    # 5. 强制保留原始 Top-1
    # ======================================

    anchor_scored = next(
        item
        for item in scored
        if item["chunk_id"] == anchor_id
    )

    selected = [anchor_scored]

    selected_ids = {anchor_id}

    section_counts = Counter({
        anchor_scored.get(
            "section_name", ""
        ): 1
    })

    # ======================================
    # 6. 保证章节多样性
    # ======================================

    for item in scored:

        if len(selected) >= top_k:
            break

        chunk_id = item["chunk_id"]

        section = item.get(
            "section_name", ""
        )

        if chunk_id in selected_ids:
            continue

        if (
            section_counts[section]
            >= max_per_section
        ):
            continue

        selected.append(item)

        selected_ids.add(chunk_id)

        section_counts[section] += 1

    # ======================================
    # 7. 重新排序并更新 rank
    # ======================================

    selected.sort(key=sort_key)

    results = []

    for rank, item in enumerate(
        selected,
        start=1
    ):

        results.append({
            **item,
            "rank": rank
        })

    return results
