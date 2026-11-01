
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 路径与参数
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "results"
    / "pdf_v3_smart_neighbor_selection.json"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "pdf_v3_semantic_neighbor_comparison.json"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

SEMANTIC_WEIGHT = 0.90
KEYWORD_WEIGHT = 0.10

MAX_KEYWORD_SCORE = 8.0


# ==========================================
# 2. 读取数据
# ==========================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"找不到文件：{INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if len(data.get("records", [])) != 5:
        raise ValueError(
            "本次应有5道测试问题"
        )

    return data


# ==========================================
# 3. 加载 BGE Embedding 模型
# ==========================================

def load_model():

    print("正在加载 BGE 模型...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(
            ROOT / "models" / "embedding_cache"
        )
    )

    print("BGE 模型加载成功！")

    return model


# ==========================================
# 4. 计算一组候选的语义相似度
# ==========================================

def score_candidates(model, record):

    question = record["question"]

    candidates = record[
        "all_scored_candidates"
    ]

    if not candidates:
        return []

    # 问题向量
    question_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )[0]

    # 邻近 Chunk 向量
    texts = [
        item["text"]
        for item in candidates
    ]

    candidate_vectors = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    scored = []

    for item, vector in zip(
        candidates,
        candidate_vectors
    ):

        # 归一化向量的点积即余弦相似度
        similarity = float(
            np.dot(
                question_vector,
                vector
            )
        )

        keyword_score = float(
            item["relevance_score"]
        )

        normalized_keyword = (
            min(
                max(keyword_score, 0.0),
                MAX_KEYWORD_SCORE
            ) / MAX_KEYWORD_SCORE
        )

        hybrid_score = (
            SEMANTIC_WEIGHT * similarity
            + KEYWORD_WEIGHT * normalized_keyword
        )

        scored.append({
            "chunk_id": item["chunk_id"],
            "page": item["page"],
            "section": item["section"],
            "text": item["text"],
            "keyword_score": keyword_score,
            "semantic_similarity": similarity,
            "hybrid_score": float(
                hybrid_score
            ),
            "original_selected": bool(
                item["selected"]
            ),
            "equation_numbers": item.get(
                "equation_numbers", []
            )
        })

    return scored


# ==========================================
# 5. 比较三种方法
# ==========================================

def compare_methods(scored):

    if not scored:
        return {}

    by_keyword = sorted(
        scored,
        key=lambda x: (
            -x["keyword_score"],
            x["chunk_id"]
        )
    )

    by_semantic = sorted(
        scored,
        key=lambda x: (
            -x["semantic_similarity"],
            x["chunk_id"]
        )
    )

    by_hybrid = sorted(
        scored,
        key=lambda x: (
            -x["hybrid_score"],
            x["chunk_id"]
        )
    )

    return {
        "keyword_first": by_keyword[0],
        "semantic_first": by_semantic[0],
        "hybrid_first": by_hybrid[0],
        "semantic_ranking": by_semantic,
        "hybrid_ranking": by_hybrid,
        "semantic_margin": (
            by_semantic[0]["semantic_similarity"]
            - by_semantic[1]["semantic_similarity"]
            if len(by_semantic) > 1
            else None
        )
    }


# ==========================================
# 6. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("第61步：语义邻近证据排序实验")
    print("=" * 65)

    data = load_data()
    model = load_model()

    records = []

    for index, record in enumerate(
        data["records"],
        start=1
    ):

        scored = score_candidates(
            model,
            record
        )

        comparison = compare_methods(
            scored
        )

        old_selected = [
            item["chunk_id"]
            for item in scored
            if item["original_selected"]
        ]

        if not comparison:
            continue

        result = {
            "question": record["question"],
            "candidate_count": len(scored),
            "old_selected_ids": old_selected,
            **comparison
        }

        records.append(result)

        print("\n" + "-" * 65)
        print(f"问题 {index}")
        print(record["question"])

        print("\n关键词策略：")
        print(
            comparison["keyword_first"][
                "section"
            ]
        )

        print("\n纯语义策略：")
        print(
            comparison["semantic_first"][
                "section"
            ]
        )

        print("\n混合策略：")
        print(
            comparison["hybrid_first"][
                "section"
            ]
        )

        print("\n全部候选的语义排序：")

        for rank, item in enumerate(
            comparison["semantic_ranking"],
            start=1
        ):

            print(
                f"{rank}. "
                f"Cosine={item['semantic_similarity']:.4f} "
                f"Hybrid={item['hybrid_score']:.4f} "
                f"第{item['page']}页 "
                f"{item['section']}"
            )

    # ======================================
    # 7. 保存报告
    # ======================================

    report = {
        "experiment": (
            "Semantic vs Keyword vs Hybrid "
            "Neighbor Ranking"
        ),
        "filename": data["filename"],
        "pdf_hash": data["pdf_hash"],
        "model": MODEL_NAME,
        "semantic_weight": SEMANTIC_WEIGHT,
        "keyword_weight": KEYWORD_WEIGHT,
        "candidate_limitation": (
            "Only the 3 exported neighbors "
            "per question from Step 59."
        ),
        "selection_note": (
            "All three methods are compared "
            "without applying a relevance "
            "acceptance threshold."
        ),
        "records": records
    }

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
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 65)
    print("实验完成！")
    print("=" * 65)

    print("比较问题数量：", len(records))
    print("报告保存位置：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
