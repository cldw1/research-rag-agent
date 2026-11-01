from pathlib import Path

from datasets import load_dataset


def main():

    project_root = Path(__file__).resolve().parents[1]

    cache_dir = project_root / "data" / "raw" / "hf_cache"

    base_url = (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/refs%2Fconvert%2Fparquet/qasper"
    )

    print("=" * 60)
    print("科研 RAG Agent - QASPER 标准答案检查")
    print("=" * 60)

    # 1. 读取训练集
    print("\n[1] 正在读取 QASPER 训练集...")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"{base_url}/train/0000.parquet"
        },
        split="train",
        cache_dir=str(cache_dir)
    )

    print("训练集读取成功！")
    print("论文数量:", len(dataset))

    # 2. 查找目标论文
    print("\n[2] 查找目标论文...")

    target_paper_id = "1909.00694"

    paper = None

    for item in dataset:
        if item["id"] == target_paper_id:
            paper = item
            break

    if paper is None:
        print("未找到目标论文！")
        return

    print("论文 ID:", paper["id"])
    print("论文标题:", paper["title"])

    # 3. 查找目标问题
    print("\n[3] 查找目标问题...")

    target_question = "What is the seed lexicon?"

    qas = paper["qas"]

    questions = qas["question"]

    question_index = None

    for i, question in enumerate(questions):
        if question.strip().lower() == target_question.lower():
            question_index = i
            break

    if question_index is None:
        print("未找到完全匹配的问题。")
        print("该论文中的问题如下：")

        for i, question in enumerate(questions, start=1):
            print(f"{i}. {question}")

        return

    print("问题:", questions[question_index])
    print("问题 ID:", qas["question_id"][question_index])

    # 4. 读取人工标注答案
    print("\n[4] 人工标注答案...")

    answer_group = qas["answers"][question_index]
    answers = answer_group["answer"]

    print("人工标注数量:", len(answers))

    for index, answer in enumerate(answers, start=1):

        print("\n" + "-" * 60)
        print(f"第 {index} 份人工标注")
        print("-" * 60)

        print("是否不可回答:", answer["unanswerable"])

        free_form_answer = answer["free_form_answer"]

        print("\n自由文本答案:")
        print(free_form_answer if free_form_answer else "无")

        extractive_spans = answer["extractive_spans"]

        print("\n抽取式答案:")
        if extractive_spans:
            for span in extractive_spans:
                print("-", span)
        else:
            print("无")

        evidence_list = answer["evidence"]

        print("\n标准证据数量:", len(evidence_list))

        for evidence_index, evidence in enumerate(
            evidence_list,
            start=1
        ):
            print(f"\n证据 {evidence_index}:")
            print(evidence[:1000])

    print("\n" + "=" * 60)
    print("QASPER 标准答案检查完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()