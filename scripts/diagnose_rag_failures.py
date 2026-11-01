
import json
import re
import sys
from pathlib import Path

import numpy as np


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.paper_scoped_retriever import PaperScopedRetriever
from src.retrieval.bm25_retriever import tokenize


GOLD_PATH = (
    PROJECT_ROOT
    / "results"
    / "rag_gold_comparison_5.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "rag_failure_diagnosis_2.json"
)


# ==========================================
# 2. 两道需要诊断的问题
# ==========================================

TARGET_CASES = {
    "1708.05873": {
        "name": "主题识别问题",
        "keywords": [
            "exclusivity",
            "semantic coherence",
            "16 topic"
        ]
    },
    "1905.10810": {
        "name": "PlEWi 标注问题",
        "keywords": [
            "error correction pairs",
            "error type annotations"
        ]
    }
}


# ==========================================
# 3. 文本标准化
# ==========================================

def normalize_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return " ".join(text.split())


def normalize_whitespace(text):

    return " ".join(
        str(text).lower().split()
    )


# ==========================================
# 4. 标准证据匹配
# ==========================================

def matches_gold_evidence(text, gold_evidence):

    """
    与之前证据评估使用相同的基本包含匹配思路。

    注意：
    如果标准证据被分块截断，
    这种匹配方法可能漏检。
    """

    text = normalize_whitespace(text)

    if not text:
        return False

    for evidence in gold_evidence:

        evidence = normalize_whitespace(evidence)

        if not evidence:
            continue

        if evidence in text:
            return True

        if len(text) >= 80 and text in evidence:
            return True

    return False


# ==========================================
# 5. 读取标准答案
# ==========================================

def load_target_cases():

    with open(
        GOLD_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    records = data.get("results", [])

    selected = []

    for item in records:

        paper_id = str(item.get("paper_id", ""))

        if paper_id not in TARGET_CASES:
            continue

        evidence = []

        for answer in item.get("gold_answers", []):

            for text in answer.get("gold_evidence", []):

                if isinstance(text, str) and text.strip():
                    evidence.append(text.strip())

        evidence = list(dict.fromkeys(evidence))

        selected.append({
            "paper_id": paper_id,
            "question_id": item.get("question_id"),
            "question": item["question"],
            "model_answer": item.get("model_answer", ""),
            "gold_answers": item.get("gold_answers", []),
            "gold_evidence": evidence,
            "name": TARGET_CASES[paper_id]["name"],
            "keywords": TARGET_CASES[paper_id]["keywords"]
        })

    if len(selected) != len(TARGET_CASES):
        raise ValueError(
            "没有找到全部两道诊断问题，"
            "请检查 rag_gold_comparison_5.json"
        )

    return selected


# ==========================================
# 6. 获取三种方法的完整排名
# ==========================================

def get_rankings(retriever, question, paper_id):

    paper_data = retriever.get_paper_data(paper_id)

    chunks = paper_data["chunks"]
    n = len(chunks)

    # -------- BM25 --------

    bm25_scores = paper_data["bm25"].get_scores(
        tokenize(question)
    )

    bm25_order = sorted(
        range(n),
        key=lambda i: (
            -float(bm25_scores[i]),
            i
        )
    )

    # -------- Dense --------

    instruction = (
        "Represent this sentence for searching "
        "relevant passages: "
    )

    query_vector = retriever.dense.model.encode(
        [instruction + question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )[0].astype(np.float32)

    dense_scores = (
        paper_data["vectors"] @ query_vector
    )

    dense_order = sorted(
        range(n),
        key=lambda i: (
            -float(dense_scores[i]),
            i
        )
    )

    # -------- Hybrid --------

    hybrid_results = retriever.search(
        query=question,
        paper_id=paper_id,
        top_k=n,
        candidate_k=n
    )

    chunk_id_to_index = {
        chunk["chunk_id"]: i
        for i, chunk in enumerate(chunks)
    }

    hybrid_order = [
        chunk_id_to_index[item["chunk_id"]]
        for item in hybrid_results
    ]

    return chunks, {
        "BM25": bm25_order,
        "Dense": dense_order,
        "Hybrid": hybrid_order
    }


# ==========================================
# 7. 分析一种检索方法
# ==========================================

def analyze_method(
    chunks,
    order,
    gold_evidence,
    keywords
):

    normalized_keywords = [
        normalize_text(keyword)
        for keyword in keywords
    ]

    gold_first_rank = None

    keyword_first_ranks = {
        keyword: None
        for keyword in keywords
    }

    top_results = []

    for rank, index in enumerate(order, start=1):

        chunk = chunks[index]
        text = chunk["text"]

        # 是否匹配人工标注证据
        gold_match = matches_gold_evidence(
            text,
            gold_evidence
        )

        if gold_match and gold_first_rank is None:
            gold_first_rank = rank

        # 检查关键答案词语
        normalized_chunk = normalize_text(text)

        found_keywords = []

        for keyword, normalized_keyword in zip(
            keywords,
            normalized_keywords
        ):

            if normalized_keyword in normalized_chunk:

                found_keywords.append(keyword)

                if keyword_first_ranks[keyword] is None:
                    keyword_first_ranks[keyword] = rank

        # 保存 Top-10 供后续分析
        if rank <= 10:

            top_results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "section_name": chunk.get(
                    "section_name", ""
                ),
                "gold_match": gold_match,
                "keywords_found": found_keywords,
                "text_preview": text[:500]
            })

    return {
        "gold_first_rank": gold_first_rank,
        "gold_hit_at": {
            str(k): (
                gold_first_rank is not None
                and gold_first_rank <= k
            )
            for k in (5, 10, 20, 50)
        },
        "keyword_first_ranks": keyword_first_ranks,
        "top_10": top_results
    }


# ==========================================
# 8. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("科研 RAG Agent - 回答失败原因诊断")
    print("=" * 65)

    print("\n[1] 读取两道失败问题...")

    cases = load_target_cases()

    print("待诊断问题数量:", len(cases))

    print("\n[2] 初始化指定论文检索器...")

    retriever = PaperScopedRetriever()

    all_results = []

    # ======================================
    # 逐题诊断
    # ======================================

    for case_number, case in enumerate(
        cases,
        start=1
    ):

        print("\n" + "=" * 65)
        print(f"问题 {case_number}: {case['name']}")
        print("=" * 65)

        paper_id = case["paper_id"]
        question = case["question"]

        print("\n论文 ID:", paper_id)
        print("问题:", question)

        print("\n[3] QASPER 人工标准答案")

        for gold in case["gold_answers"]:

            print(
                "答案类型:",
                gold.get("answer_type")
            )

            print(
                "标准答案:",
                gold.get("gold_answer")
            )

        print(
            "\n人工证据数量:",
            len(case["gold_evidence"])
        )

        # ----------------------------------
        # 获取三种检索方法排名
        # ----------------------------------

        print("\n[4] 执行 BM25 / Dense / Hybrid...")

        chunks, rankings = get_rankings(
            retriever,
            question,
            paper_id
        )

        print("当前论文 Chunk 数量:", len(chunks))

        # 判断整篇论文中是否存在匹配证据
        matched_chunks = [
            chunk["chunk_id"]
            for chunk in chunks
            if matches_gold_evidence(
                chunk["text"],
                case["gold_evidence"]
            )
        ]

        print(
            "知识库中匹配人工证据的 Chunk 数量:",
            len(matched_chunks)
        )

        # ----------------------------------
        # 逐方法分析
        # ----------------------------------

        method_results = {}

        for method, order in rankings.items():

            analysis = analyze_method(
                chunks=chunks,
                order=order,
                gold_evidence=case["gold_evidence"],
                keywords=case["keywords"]
            )

            method_results[method] = analysis

            print("\n" + "-" * 55)
            print("检索方法:", method)

            print(
                "标准证据首次命中排名:",
                analysis["gold_first_rank"]
            )

            for k, hit in analysis["gold_hit_at"].items():

                print(f"Evidence Hit@{k}: {hit}")

            print("\n关键答案词语首次出现排名:")

            for keyword, rank in (
                analysis["keyword_first_ranks"].items()
            ):

                print(f"  {keyword}: {rank}")

            print("\nTop-5 Chunk:")

            for item in analysis["top_10"][:5]:

                print(
                    f"排名 {item['rank']} | "
                    f"证据匹配={item['gold_match']} | "
                    f"{item['chunk_id']}"
                )

        # ----------------------------------
        # 保存本题结果
        # ----------------------------------

        all_results.append({
            "paper_id": paper_id,
            "question_id": case["question_id"],
            "question": question,
            "model_answer": case["model_answer"],
            "gold_answers": case["gold_answers"],
            "gold_evidence_count": len(
                case["gold_evidence"]
            ),
            "paper_chunk_count": len(chunks),
            "matching_gold_chunk_ids": matched_chunks,
            "methods": method_results
        })

    # ======================================
    # 保存诊断结果
    # ======================================

    print("\n[5] 保存诊断结果...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "description": (
                    "QASPER RAG failure diagnosis"
                ),
                "scope": "paper_scoped",
                "gold_match_rule": (
                    "normalized text containment; "
                    "may miss split evidence"
                ),
                "results": all_results
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n保存路径:", OUTPUT_PATH)

    print("\n说明：")
    print("标准证据匹配采用文本包含规则。")
    print("若证据被截断或拆分，可能出现假阴性。")
    print("关键答案词语只是辅助诊断，不是正式指标。")

    print("\n" + "=" * 65)
    print("RAG 失败原因诊断完成！")
    print("=" * 65)


if __name__ == "__main__":
    main()
