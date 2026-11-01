import json
from pathlib import Path


def main():

    project_root = Path(__file__).resolve().parents[1]

    input_path = (
        project_root
        / "results"
        / "bm25_train_debug_30.json"
    )

    output_path = (
        project_root
        / "results"
        / "bm25_error_analysis.json"
    )

    print("=" * 60)
    print("科研 RAG Agent - BM25 检索失败案例分析")
    print("=" * 60)

    # 1. 读取之前的实验结果
    with open(input_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    cases = report["detailed_results"]

    print("\n[1] 实验基本信息")
    print("评测问题数量:", len(cases))

    # 2. 按失败原因分类
    categories = {
        "success": [],
        "wrong_paper": [],
        "wrong_chunk": [],
        "missing_evidence": []
    }

    top1_correct_paper = 0
    top10_contains_correct_paper = 0

    for case in cases:

        target_paper_id = case["paper_id"]
        retrieved = case["retrieved"]

        retrieved_paper_ids = [
            item["paper_id"]
            for item in retrieved
        ]

        # Top-1 是否来自正确论文
        if (
            retrieved
            and retrieved[0]["paper_id"] == target_paper_id
        ):
            top1_correct_paper += 1

        # Top-10 是否包含正确论文
        if target_paper_id in retrieved_paper_ids:
            top10_contains_correct_paper += 1

        # 分类
        if not case["gold_available_in_index"]:

            category = "missing_evidence"

        elif case["first_hit_rank"] is not None:

            category = "success"

        elif target_paper_id not in retrieved_paper_ids:

            category = "wrong_paper"

        else:

            category = "wrong_chunk"

        categories[category].append({
            "question_id": case["question_id"],
            "paper_id": target_paper_id,
            "question": case["question"],
            "retrieved_paper_ids": retrieved_paper_ids
        })

    # 3. 打印统计结果
    print("\n[2] 检索失败原因统计")

    print(
        "检索成功:",
        len(categories["success"])
    )

    print(
        "找错论文:",
        len(categories["wrong_paper"])
    )

    print(
        "找到论文但未命中证据:",
        len(categories["wrong_chunk"])
    )

    print(
        "证据无法匹配:",
        len(categories["missing_evidence"])
    )

    print("\n[3] 论文检索情况")

    print(
        "Top-1 来自正确论文:",
        f"{top1_correct_paper}/{len(cases)}"
    )

    print(
        "Top-10 包含正确论文:",
        f"{top10_contains_correct_paper}/{len(cases)}"
    )

    # 4. 展示部分失败案例
    print("\n[4] 失败案例分析")

    for category_name, title in [
        ("wrong_paper", "找错论文"),
        ("wrong_chunk", "找对论文但找错段落")
    ]:

        print("\n" + "-" * 60)
        print(title)
        print("-" * 60)

        examples = categories[category_name][:3]

        if not examples:
            print("没有此类失败案例")
            continue

        for index, example in enumerate(
            examples,
            start=1
        ):

            print(f"\n案例 {index}")

            print("问题:", example["question"])

            print(
                "目标论文:",
                example["paper_id"]
            )

            print(
                "Top-10 论文 ID:",
                example["retrieved_paper_ids"]
            )

    # 5. 保存分析结果
    analysis = {
        "total_questions": len(cases),
        "success_count": len(categories["success"]),
        "wrong_paper_count": len(categories["wrong_paper"]),
        "wrong_chunk_count": len(categories["wrong_chunk"]),
        "missing_evidence_count": len(
            categories["missing_evidence"]
        ),
        "top1_correct_paper": top1_correct_paper,
        "top10_contains_correct_paper":
            top10_contains_correct_paper,
        "categories": categories
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            analysis,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n[5] 保存结果")
    print("输出文件:", output_path)

    print("\n" + "=" * 60)
    print("BM25 失败案例分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()