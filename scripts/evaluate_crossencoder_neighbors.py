
import json
from pathlib import Path

from sentence_transformers import CrossEncoder


# ==========================================
# 1. 路径配置
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "results"
    / "pdf_v3_semantic_neighbor_comparison.json"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "pdf_v3_crossencoder_comparison.json"
)

MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ==========================================
# 2. 读取上一轮实验
# ==========================================

def load_data():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"找不到文件：{INPUT_PATH}"
        )

    data = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not data.get("records"):
        raise ValueError("没有找到实验问题")

    return data


# ==========================================
# 3. 加载 Cross-Encoder
# ==========================================

def load_model():

    print("正在加载 Cross-Encoder...")

    model = CrossEncoder(
        MODEL_NAME,
        device="cpu",
        max_length=512
    )

    print("模型加载成功")

    return model


# ==========================================
# 4. 获取候选段落
# ==========================================

def get_candidates(record):

    # semantic_ranking 已经包含全部
    # 三个候选，每个 Chunk ID 不重复。
    raw_items = record["semantic_ranking"]

    seen = set()
    candidates = []

    for item in raw_items:

        chunk_id = item["chunk_id"]

        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        candidates.append(dict(item))

    return candidates


# ==========================================
# 5. Cross-Encoder 评分
# ==========================================

def evaluate_question(model, record):

    question = record["question"]

    candidates = get_candidates(record)

    # --------------------------------------
    # A. 正文评分
    # --------------------------------------

    text_pairs = [
        [question, item["text"]]
        for item in candidates
    ]

    text_scores = model.predict(
        text_pairs,
        batch_size=8,
        show_progress_bar=False
    )

    # --------------------------------------
    # B. 章节标题 + 正文评分
    # --------------------------------------

    section_pairs = [
        [
            question,
            (
                f"Section: {item['section']}\n"
                f"{item['text']}"
            )
        ]
        for item in candidates
    ]

    section_scores = model.predict(
        section_pairs,
        batch_size=8,
        show_progress_bar=False
    )

    # --------------------------------------
    # C. 保存详细结果
    # --------------------------------------

    scored = []

    for item, text_score, section_score in zip(
        candidates,
        text_scores,
        section_scores
    ):

        scored.append({
            "chunk_id": item["chunk_id"],
            "page": item["page"],
            "section": item["section"],
            "text": item["text"],
            "keyword_score": float(
                item["keyword_score"]
            ),
            "bge_similarity": float(
                item["semantic_similarity"]
            ),
            "previous_hybrid_score": float(
                item["hybrid_score"]
            ),
            "crossencoder_text_score": float(
                text_score
            ),
            "crossencoder_section_score": float(
                section_score
            ),
            "equation_numbers": item.get(
                "equation_numbers", []
            )
        })

    # --------------------------------------
    # D. 不同策略的排名
    # --------------------------------------

    by_bge = sorted(
        scored,
        key=lambda x: (
            -x["bge_similarity"],
            x["chunk_id"]
        )
    )

    by_crossencoder = sorted(
        scored,
        key=lambda x: (
            -x["crossencoder_text_score"],
            x["chunk_id"]
        )
    )

    by_section_crossencoder = sorted(
        scored,
        key=lambda x: (
            -x["crossencoder_section_score"],
            x["chunk_id"]
        )
    )

    return {
        "question": question,
        "candidate_count": len(candidates),
        "bge_first": by_bge[0]["chunk_id"],
        "crossencoder_first": (
            by_crossencoder[0]["chunk_id"]
        ),
        "section_crossencoder_first": (
            by_section_crossencoder[0]["chunk_id"]
        ),
        "bge_ranking": [
            item["chunk_id"]
            for item in by_bge
        ],
        "crossencoder_ranking": [
            item["chunk_id"]
            for item in by_crossencoder
        ],
        "section_crossencoder_ranking": [
            item["chunk_id"]
            for item in by_section_crossencoder
        ],
        "candidates": scored
    }


# ==========================================
# 6. 主程序
# ==========================================

def main():

    data = load_data()
    model = load_model()

    results = []

    print("=" * 65)
    print("第62步：Cross-Encoder 证据排序")
    print("=" * 65)

    for index, record in enumerate(
        data["records"],
        start=1
    ):

        result = evaluate_question(
            model,
            record
        )

        results.append(result)

        lookup = {
            item["chunk_id"]: item
            for item in result["candidates"]
        }

        print("\n" + "-" * 65)
        print(f"问题 {index}")
        print(result["question"])

        print("\nBGE 第一名：")
        print(
            lookup[result["bge_first"]]["section"]
        )

        print("\nCross-Encoder 第一名：")
        print(
            lookup[
                result["crossencoder_first"]
            ]["section"]
        )

        print("\n章节增强 Cross-Encoder 第一名：")
        print(
            lookup[
                result["section_crossencoder_first"]
            ]["section"]
        )

        print("\n候选排名明细：")

        ranked = sorted(
            result["candidates"],
            key=lambda x: (
                -x["crossencoder_section_score"],
                x["chunk_id"]
            )
        )

        for rank, item in enumerate(
            ranked,
            start=1
        ):

            print(
                f"{rank}. "
                f"第{item['page']}页 | "
                f"BGE={item['bge_similarity']:.4f} | "
                f"CE={item['crossencoder_text_score']:.4f} | "
                f"CE+Section="
                f"{item['crossencoder_section_score']:.4f}"
            )

            print(
                "   章节：",
                item["section"]
            )

    # ======================================
    # 7. 保存 JSON
    # ======================================

    report = {
        "experiment": (
            "Cross-Encoder Neighbor Reranking"
        ),
        "filename": data["filename"],
        "pdf_hash": data["pdf_hash"],
        "model": MODEL_NAME,
        "candidate_limitation": (
            "Only 3 previously exported "
            "neighbors per question"
        ),
        "score_note": (
            "Cross-Encoder scores are raw "
            "ranking scores, not calibrated "
            "probabilities. Compare ordering, "
            "not absolute score magnitude."
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
    print("Cross-Encoder 实验完成")
    print("=" * 65)

    print(
        "测试问题数：",
        len(results)
    )

    print(
        "报告：",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
