import json
import sys
import textwrap
from pathlib import Path

from datasets import load_dataset


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PREDICTION_PATH = (
    PROJECT_ROOT / "results" / "rag_answers_debug_5.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "rag_gold_comparison_5.json"
)


# ==========================================
# 2. 读取模型回答
# ==========================================

def load_predictions():

    with open(
        PREDICTION_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    return data["results"]


# ==========================================
# 3. 读取 QASPER 训练集
# ==========================================

def load_qasper_train():

    base_url = (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/refs%2Fconvert%2Fparquet/qasper"
    )

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{base_url}/train/0000.parquet"
        },
        split="train",
        cache_dir=str(
            PROJECT_ROOT / "data" / "raw" / "hf_cache"
        ),
    )

    return dataset


# ==========================================
# 4. 递归提取答案标注
# ==========================================

def find_answer_objects(obj):

    if isinstance(obj, list):

        for item in obj:
            yield from find_answer_objects(item)

    elif isinstance(obj, dict):

        answer_fields = {
            "unanswerable",
            "extractive_spans",
            "free_form_answer",
            "yes_no",
        }

        if answer_fields.intersection(obj.keys()):

            yield obj

        else:

            for key in ("answer", "answers"):

                if key in obj:

                    yield from find_answer_objects(obj[key])


# ==========================================
# 5. 格式化标准答案
# ==========================================

def format_gold_answer(annotation):

    evidence = annotation.get("evidence") or []

    if annotation.get("unanswerable"):

        answer_type = "unanswerable"
        answer_text = "Unanswerable"

    elif annotation.get("extractive_spans"):

        answer_type = "extractive"

        answer_text = "; ".join(
            str(span)
            for span in annotation["extractive_spans"]
        )

    elif annotation.get("free_form_answer"):

        answer_type = "free_form"

        answer_text = str(
            annotation["free_form_answer"]
        )

    elif annotation.get("yes_no") is True:

        answer_type = "yes_no"
        answer_text = "Yes"

    elif annotation.get("yes_no") is False:

        answer_type = "yes_no"
        answer_text = "No"

    else:

        answer_type = "unknown"
        answer_text = ""

    return {
        "answer_type": answer_type,
        "gold_answer": answer_text,
        "gold_evidence": evidence,
    }


# ==========================================
# 6. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("RAG 回答与 QASPER 标准答案对照")
    print("=" * 65)

    print("\n[1] 读取模型生成结果...")

    predictions = load_predictions()

    print(f"模型回答数量：{len(predictions)}")

    print("\n[2] 加载 QASPER 训练集...")

    dataset = load_qasper_train()

    print(f"训练集论文数量：{len(dataset)}")

    # 建立需要查询的问题索引
    prediction_map = {
        (
            str(item["paper_id"]),
            str(item["question_id"]),
        ): item
        for item in predictions
    }

    comparison = []

    print("\n[3] 匹配人工标注答案...")

    for paper in dataset:

        paper_id = str(paper["id"])

        qas = paper["qas"]

        question_ids = qas.get("question_id", [])
        answers = qas.get("answers", [])

        for index, question_id in enumerate(question_ids):

            key = (
                paper_id,
                str(question_id),
            )

            if key not in prediction_map:
                continue

            prediction = prediction_map[key]

            raw_answers = (
                answers[index]
                if index < len(answers)
                else []
            )

            gold_answers = [
                format_gold_answer(annotation)
                for annotation in find_answer_objects(
                    raw_answers
                )
            ]

            record = {
                "paper_id": paper_id,
                "question_id": str(question_id),
                "question": prediction["question"],
                "model_answer": prediction.get("answer", ""),
                "cited_numbers": prediction.get(
                    "cited_numbers", []
                ),
                "gold_answers": gold_answers,
            }

            comparison.append(record)

    # ======================================
    # 7. 展示对照结果
    # ======================================

    for index, item in enumerate(comparison, start=1):

        print("\n" + "=" * 65)
        print(f"第 {index} 道题")
        print("=" * 65)

        print(f"\n问题：{item['question']}")

        print("\n模型回答：")

        print(
            textwrap.fill(
                item["model_answer"],
                width=95,
                replace_whitespace=False,
            )
        )

        print("\nQASPER 人工标注答案：")

        for number, gold in enumerate(
            item["gold_answers"],
            start=1,
        ):

            print(f"\n标注 {number}：")

            print(
                f"答案类型：{gold['answer_type']}"
            )

            print(
                f"标准答案：{gold['gold_answer']}"
            )

            print(
                f"证据段落数量：{len(gold['gold_evidence'])}"
            )

    # ======================================
    # 8. 保存对照结果
    # ======================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "description": "QASPER train answer comparison",
                "results": comparison,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("\n" + "=" * 65)
    print("标准答案提取完成")
    print("=" * 65)

    print(f"成功匹配问题数量：{len(comparison)}")
    print(f"结果保存位置：{OUTPUT_PATH}")

    print("\n注意：尚未自动判定回答正确率。")


if __name__ == "__main__":
    main()