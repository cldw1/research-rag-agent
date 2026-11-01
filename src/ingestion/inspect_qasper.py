
from pathlib import Path

from datasets import load_dataset


# ==========================================
# P2-02：分析 QASPER 数据结构
# ==========================================


def main():
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "data" / "raw" / "hf_cache"

    base_url = (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/refs%2Fconvert%2Fparquet/qasper"
    )

    # 只读取训练集，避免影响后续独立评测
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{base_url}/train/0000.parquet"
        },
        split="train",
        cache_dir=str(cache_dir)
    )

    paper = dataset[0]

    print("=" * 60)
    print("QASPER 数据结构分析")
    print("=" * 60)

    print("\n[1] 论文基本信息")
    print("论文 ID:", paper["id"])
    print("论文标题:", paper["title"])

    # ======================================
    # 1. 分析论文正文
    # ======================================

    print("\n[2] full_text 数据结构")

    full_text = paper["full_text"]

    print("字段:", list(full_text.keys()))

    section_names = full_text["section_name"]
    paragraphs = full_text["paragraphs"]

    print("章节数量:", len(section_names))

    if section_names:
        print("第一个章节标题:", section_names[0])

    if paragraphs:
        print("第一个章节的段落数量:", len(paragraphs[0]))

        if paragraphs[0]:
            print("\n第一段正文前 500 个字符:")
            print(paragraphs[0][0][:500])

    # ======================================
    # 2. 分析问答数据
    # ======================================

    print("\n[3] qas 数据结构")

    qas = paper["qas"]

    print("字段:", list(qas.keys()))

    questions = qas["question"]
    answers = qas["answers"]

    print("问题数量:", len(questions))

    if questions:
        print("\n第一个问题:")
        print(questions[0])

    # ======================================
    # 3. 分析标准答案与证据
    # ======================================

    print("\n[4] 标准答案与证据")

    if answers:
        answer_group = answers[0]

        print("答案标注字段:", list(answer_group.keys()))

        annotations = answer_group.get("answer", [])

        print("第一个问题的答案标注数量:", len(annotations))

        if annotations:
            first_answer = annotations[0]

            print("\n第一条答案标注的字段:")
            print(list(first_answer.keys()))

            print("\n是否不可回答:")
            print(first_answer.get("unanswerable"))

            print("\n自由文本答案:")
            print(first_answer.get("free_form_answer"))

            evidence = first_answer.get("evidence") or []

            print("\n证据数量:", len(evidence))

            if evidence:
                print("\n第一条证据前 500 个字符:")
                print(evidence[0][:500])

    print("\n" + "=" * 60)
    print("QASPER 数据结构分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
