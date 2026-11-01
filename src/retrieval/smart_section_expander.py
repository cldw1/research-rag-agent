
"""
Smart Section Neighbor Expander

在已验证的 V3 PDF 索引中：
1. 保留原始 Anchor
2. 跨页寻找同一真实章节中的邻居
3. 根据问题、正文关键词及公式特征进行排序

本模块只提供邻居候选排序。
最终上下文预算由 RAG Pipeline 控制。
"""

import re


STOPWORDS = {
    "a", "an", "the", "is", "are",
    "was", "were", "be", "been",
    "what", "why", "how", "does",
    "do", "did", "can", "could",
    "and", "or", "of", "in",
    "on", "at", "to", "for",
    "with", "by", "from",
    "this", "that", "it", "its",
    "as", "which",
    "hmc", "hamiltonian", "monte",
    "carlo", "method", "methods"
}


# 识别 5.18、5.19 等公式编号。
# 不依赖某一个指定的公式编号。
EQUATION_NUMBER_PATTERN = re.compile(
    r"(?<![\d.])\(?([1-9]\d{0,2}\.\d{1,3})\)?(?![\d.])"
)


# ==========================================
# 1. 提取关键词
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

    result = set()

    for word in words:

        if word in {
            "tune", "tuned",
            "tunes", "tuning"
        }:
            word = "tuning"

        elif word == "matrices":
            word = "matrix"

        elif (
            word.endswith("ies")
            and len(word) > 5
        ):
            word = word[:-3] + "y"

        elif (
            word.endswith("s")
            and len(word) > 4
        ):
            word = word[:-1]

        if (
            word not in STOPWORDS
            and len(word) >= 3
        ):
            result.add(word)

    return result


# ==========================================
# 2. 识别公式编号
# ==========================================

def equation_numbers(text):

    return sorted(
        set(
            EQUATION_NUMBER_PATTERN.findall(
                str(text)
            )
        )
    )


# ==========================================
# 3. 判断问题是否偏向算法或公式
# ==========================================

def asks_for_procedure(question):

    words = keyword_tokens(question)

    procedure_words = {
        "integrator",
        "leapfrog",
        "formula",
        "equation",
        "update",
        "calculate",
        "derive",
        "compute",
        "algorithm",
        "gradient"
    }

    return bool(
        words & procedure_words
    )


# ==========================================
# 4. 为同章节邻居排序
# ==========================================

def rank_section_neighbors(
    question,
    anchors,
    all_chunks,
    max_distance=10
):
    """
    输入：
        question:
            用户问题

        anchors:
            原始检索 Top-5

        all_chunks:
            当前论文的全部 V3 Chunk

        max_distance:
            最多检查 Anchor 前后各多少个 Chunk

    注意：
        Chunk 距离按文档顺序计算，
        可以跨 PDF 页，
        但不能跨真实章节。

    返回：
        按证据评分从高到低排列的邻居候选。
    """

    if max_distance < 1:
        raise ValueError(
            "max_distance 必须大于等于1"
        )

    index_by_id = {
        item["chunk_id"]: index
        for index, item in enumerate(
            all_chunks
        )
    }

    anchor_ids = {
        item["chunk_id"]
        for item in anchors
    }

    question_terms = keyword_tokens(
        question
    )

    procedure = asks_for_procedure(
        question
    )

    candidates = {}

    # ======================================
    # 5. 遍历每个 Anchor
    # ======================================

    for anchor_rank, anchor in enumerate(
        anchors,
        start=1
    ):

        anchor_id = anchor["chunk_id"]

        if anchor_id not in index_by_id:
            raise ValueError(
                f"找不到 Anchor：{anchor_id}"
            )

        anchor_index = index_by_id[
            anchor_id
        ]

        section = all_chunks[
            anchor_index
        ]["section_name"]

        if section.strip().casefold() in {
            "references",
            "bibliography"
        }:
            continue

        start = max(
            0,
            anchor_index - max_distance
        )

        end = min(
            len(all_chunks),
            anchor_index + max_distance + 1
        )

        # ==================================
        # 6. 检查邻居
        # ==================================

        for index in range(start, end):

            chunk = all_chunks[index]

            chunk_id = chunk["chunk_id"]

            # 原始 Anchor 不重复添加
            if chunk_id in anchor_ids:
                continue

            # 必须属于同一个真实章节
            if (
                chunk["section_name"]
                != section
            ):
                continue

            if chunk.get(
                "is_reference", False
            ):
                continue

            text = chunk["text"]

            matched = sorted(
                question_terms
                & keyword_tokens(text)
            )

            equation_ids = equation_numbers(
                text
            )

            math_symbols = (
                text.count("=")
                + text.count("∂")
                + text.count("∇")
            )

            # ==================================
            # 7. 公式证据奖励
            # ==================================

            formula_bonus = 0.0

            if procedure:

                if len(equation_ids) >= 2:

                    formula_bonus = (
                        4.5
                        if math_symbols > 0
                        else 2.5
                    )

                elif (
                    equation_ids
                    and math_symbols > 0
                ):

                    formula_bonus = 1.5

                elif math_symbols >= 2:

                    formula_bonus = 0.5

            # ==================================
            # 8. 距离奖励
            # ==================================

            distance = abs(
                index - anchor_index
            )

            distance_bonus = (
                0.6 / distance
            )

            anchor_bonus = (
                0.15 / anchor_rank
            )

            # ==================================
            # 9. 综合评分
            # ==================================

            score = (
                1.5 * min(
                    len(matched), 4
                )
                + formula_bonus
                + distance_bonus
                + anchor_bonus
            )

            # 避免仅因距离近就新增无关证据
            if score < 1.1:
                continue

            record = {
                **chunk,
                "source_role": "neighbor",
                "neighbor_selection": (
                    "smart_section"
                ),
                "neighbor_of": anchor_id,
                "neighbor_score": round(
                    score, 6
                ),
                "matched_query_words": matched,
                "equation_numbers": (
                    equation_ids
                )
            }

            # 同一 Chunk 可能是多个 Anchor 的邻居
            # 只保留分数最高的一次
            previous = candidates.get(
                chunk_id
            )

            if (
                previous is None
                or score
                > previous["neighbor_score"]
            ):
                candidates[
                    chunk_id
                ] = record

    # ======================================
    # 10. 全局排序
    # ======================================

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -item["neighbor_score"],
            item["chunk_id"]
        )
    )

    return ranked
