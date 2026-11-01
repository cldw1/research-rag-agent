
import json
import re
from pathlib import Path


# ==========================================
# 1. 项目路径
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "results"
    / "pdf_v3_neighbor_expansion.json"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "pdf_v3_smart_neighbor_selection.json"
)


# ==========================================
# 2. 实验参数
# ==========================================

MAX_EXTRA_CHUNKS = 1

MIN_RELEVANCE_SCORE = 1.5

SECTION_MATCH_WEIGHT = 4.0

TEXT_MATCH_WEIGHT = 1.5

FORMULA_BONUS = 2.5


# ==========================================
# 3. 常见停用词
# ==========================================

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
    "method", "methods",
    "algorithm", "algorithms",
    "distribution", "distributions"
}


# ==========================================
# 4. 关键词提取
# ==========================================

def keyword_tokens(text):

    text = str(text).casefold()

    text = re.sub(
        r"\bstep[\s-]+size\b",
        "stepsize",
        text
    )

    words = re.findall(
        r"[a-z]+",
        text
    )

    tokens = set()

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
# 5. 检测公式编号
# ==========================================

def find_equation_numbers(text):

    # 匹配 (5.18)、(5.19)、(5.20)
    # 也支持其他章节的类似编号
    return sorted(
        set(
            re.findall(
                r"\(\d+(?:\.\d+)+\)",
                text
            )
        )
    )


# ==========================================
# 6. 判断是否需要过程类证据
# ==========================================

def needs_procedure_evidence(question):

    tokens = keyword_tokens(question)

    procedure_words = {
        "integrator",
        "equation",
        "formula",
        "update",
        "derive",
        "calculate",
        "compute"
    }

    if tokens & procedure_words:
        return True

    text = question.casefold()

    if "how" in text and "work" in text:
        return True

    return False


# ==========================================
# 7. 计算邻近证据得分
# ==========================================

def score_neighbor(question, neighbor):

    question_words = keyword_tokens(
        question
    )

    section_words = keyword_tokens(
        neighbor["section"]
    )

    content_words = keyword_tokens(
        neighbor["text"]
    )

    section_matches = sorted(
        question_words & section_words
    )

    text_matches = sorted(
        question_words & content_words
    )

    # 标题匹配最多加一次分
    section_score = (
        SECTION_MATCH_WEIGHT
        if section_matches
        else 0.0
    )

    # 正文最多按三个匹配词计分
    text_score = (
        TEXT_MATCH_WEIGHT
        * min(len(text_matches), 3)
    )

    equation_numbers = find_equation_numbers(
        neighbor["text"]
    )

    formula_score = 0.0

    if (
        needs_procedure_evidence(question)
        and len(equation_numbers) >= 2
    ):
        formula_score = FORMULA_BONUS

    final_score = (
        section_score
        + text_score
        + formula_score
    )

    return {
        **neighbor,
        "relevance_score": final_score,
        "section_matches": section_matches,
        "text_matches": text_matches,
        "equation_numbers": equation_numbers,
        "score_breakdown": {
            "section": section_score,
            "text": text_score,
            "formula": formula_score
        }
    }


# ==========================================
# 8. 智能选择邻近证据
# ==========================================

def smart_select(record):

    question = record["question"]

    candidates = [
        score_neighbor(question, item)
        for item in record["added_neighbors"]
    ]

    # 先按相关性排序
    candidates.sort(
        key=lambda item: (
            -item["relevance_score"],
            -len(item["section_matches"]),
            -len(item["text_matches"]),
            item["chunk_id"]
        )
    )

    selected = []

    for item in candidates:

        if len(selected) >= MAX_EXTRA_CHUNKS:
            break

        if (
            item["relevance_score"]
            < MIN_RELEVANCE_SCORE
        ):
            continue

        selected.append(item)

    selected_ids = {
        item["chunk_id"]
        for item in selected
    }

    for item in candidates:
        item["selected"] = (
            item["chunk_id"] in selected_ids
        )

    return selected, candidates


# ==========================================
# 9. 构造新的证据集合
# ==========================================

def build_new_evidence(record, selected):

    all_sources = {
        item["chunk_id"]: item
        for item in record["expanded_sources"]
    }

    anchor_ids = record["anchor_ids"]

    new_ids = list(anchor_ids)

    for item in selected:

        chunk_id = item["chunk_id"]

        if chunk_id not in new_ids:
            new_ids.append(chunk_id)

    missing = [
        chunk_id
        for chunk_id in new_ids
        if chunk_id not in all_sources
    ]

    if missing:
        raise ValueError(
            f"找不到证据正文：{missing}"
        )

    new_sources = [
        all_sources[chunk_id]
        for chunk_id in new_ids
    ]

    anchors_preserved = all(
        chunk_id in new_ids
        for chunk_id in anchor_ids
    )

    # 当前报告只证明新集合是第59步
    # 已验证集合的子集。
    old_ids = set(
        record["expanded_chunk_ids"]
    )

    subset_verified = set(
        new_ids
    ).issubset(old_ids)

    if not subset_verified:
        raise ValueError(
            "新证据不属于此前验证的集合"
        )

    return {
        "new_chunk_count": len(new_sources),
        "anchors_preserved": anchors_preserved,
        "subset_of_previous_expansion": (
            subset_verified
        ),
        "retrieved_text_chars": sum(
            len(item["text"])
            for item in new_sources
        ),
        "chunk_ids": new_ids,
        "sources": new_sources
    }


# ==========================================
# 10. 主程序
# ==========================================

def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"找不到文件：{INPUT_PATH}"
        )

    data = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    if data.get("experiment") != (
        "V3 Same-Section Neighbor Expansion"
    ):
        raise ValueError(
            "输入文件不是第59步的报告"
        )

    if (
        data.get("max_context_chars")
        != 8000
    ):
        raise ValueError(
            "上下文预算不是预期的8000字符"
        )

    results = []

    print("=" * 65)
    print("第60步：智能邻近证据选择")
    print("=" * 65)

    for index, record in enumerate(
        data["records"],
        start=1
    ):

        selected, candidates = smart_select(
            record
        )

        evidence = build_new_evidence(
            record,
            selected
        )

        previous_after = record["after"]

        if (
            previous_after["context_chars"]
            > data["max_context_chars"]
        ):
            raise ValueError(
                "上一轮上下文超出预算"
            )

        selected_equations = sorted(
            set(
                number
                for item in selected
                for number in item["equation_numbers"]
            )
        )

        result = {
            "question": record["question"],
            "previous_extra_count": len(
                record["added_neighbors"]
            ),
            "selected_extra_count": len(
                selected
            ),
            "previous_context_chars": (
                previous_after["context_chars"]
            ),
            "previous_context_chars_is_upper_bound": (
                True
            ),
            "selected_equation_numbers": (
                selected_equations
            ),
            "selected_neighbors": selected,
            "all_scored_candidates": candidates,
            "evidence": evidence
        }

        results.append(result)

        print(f"\n问题 {index}")
        print(record["question"])

        print(
            "之前新增：",
            len(record["added_neighbors"])
        )

        print(
            "现在新增：",
            len(selected)
        )

        print(
            "原始证据保留：",
            evidence["anchors_preserved"]
        )

        print("\n邻居相关性排名：")

        for rank, item in enumerate(
            candidates,
            start=1
        ):

            status = (
                "选中"
                if item["selected"]
                else "跳过"
            )

            print(
                f"{rank}. "
                f"[{status}] "
                f"得分={item['relevance_score']:.2f} "
                f"第{item['page']}页 "
                f"{item['section']}"
            )

        if not selected:
            print(
                "没有足够相关的邻居，"
                "本题保持原始5个证据。"
            )

        if "leapfrog integrator" in (
            record["question"].casefold()
        ):
            print(
                "新增证据中的公式编号：",
                selected_equations
            )

    # ======================================
    # 11. 汇总
    # ======================================

    total_before = sum(
        item["previous_extra_count"]
        for item in results
    )

    total_after = sum(
        item["selected_extra_count"]
        for item in results
    )

    all_anchors_preserved = all(
        item["evidence"]["anchors_preserved"]
        for item in results
    )

    report = {
        "experiment": (
            "V3 Smart Neighbor Selection - "
            "Exported Candidates Only"
        ),
        "filename": data["filename"],
        "pdf_hash": data["pdf_hash"],
        "max_context_chars": (
            data["max_context_chars"]
        ),
        "max_extra_chunks_per_question": (
            MAX_EXTRA_CHUNKS
        ),
        "min_relevance_score": (
            MIN_RELEVANCE_SCORE
        ),
        "candidate_limitation": (
            "Only the three neighbors exported "
            "for each question in Step 59 "
            "were considered."
        ),
        "previous_total_extra": total_before,
        "new_total_extra": total_after,
        "all_anchors_preserved": (
            all_anchors_preserved
        ),
        "records": results
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
    print("实验总结")
    print("=" * 65)

    print("原来新增总数：", total_before)
    print("现在新增总数：", total_after)
    print("全部保留Anchor：", all_anchors_preserved)

    print(
        "实验报告：",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
