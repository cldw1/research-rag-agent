
import json
import re
from pathlib import Path


# ==========================================
# 1. 项目目录
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

HEADING_PATH = (
    ROOT / "results"
    / "pdf_headings_filtered.json"
)

OUTPUT_PATH = (
    ROOT / "results"
    / "pdf_v3_holdout_questions.json"
)


# ==========================================
# 2. 新测试问题
# ==========================================

TEST_CASES = [
    {
        "id": "H01",
        "question": (
            "How can a target probability density "
            "be represented using an energy "
            "function and a canonical distribution?"
        ),
        "gold_section": "5.3.1"
    },
    {
        "id": "H02",
        "question": (
            "What conditions must a continuous "
            "target density satisfy for the "
            "standard HMC sampler to be applied?"
        ),
        "gold_section": "5.3.2"
    },
    {
        "id": "H03",
        "question": (
            "What does the high-dimensional "
            "sampling example reveal about "
            "the performance of HMC?"
        ),
        "gold_section": "5.3.3.4"
    },
    {
        "id": "H04",
        "question": (
            "Why does HMC scale differently "
            "from random-walk Metropolis "
            "as the dimension increases?"
        ),
        "gold_section": "5.4.4.2"
    },
    {
        "id": "H05",
        "question": (
            "How can retaining part of the "
            "previous momentum help reduce "
            "random-walk behavior?"
        ),
        "gold_section": "5.5.3"
    },
    {
        "id": "H06",
        "question": (
            "How can HMC avoid wasting "
            "computation on a trajectory "
            "with a badly chosen stepsize?"
        ),
        "gold_section": "5.5.6"
    },
    {
        "id": "H07",
        "question": (
            "What difficulties arise when "
            "using HMC for hierarchical models "
            "and their hyperparameters?"
        ),
        "gold_section": "5.4.5"
    }
]


# ==========================================
# 3. 读取已验证的章节
# ==========================================

def load_sections():

    if not HEADING_PATH.exists():
        raise FileNotFoundError(
            f"找不到文件：{HEADING_PATH}"
        )

    data = json.loads(
        HEADING_PATH.read_text(
            encoding="utf-8"
        )
    )

    section_map = {}

    for heading in data.get(
        "headings", []
    ):

        candidate = str(
            heading.get("candidate", "")
        ).strip()

        match = re.match(
            r"^(\d+(?:\.\d+)+)\b",
            candidate
        )

        if not match:
            continue

        section_id = match.group(1)

        section_map[
            section_id
        ] = candidate

    if not section_map:
        raise ValueError(
            "没有识别到有效的章节标题，"
            "请检查 headings 文件结构"
        )

    return section_map


# ==========================================
# 4. 检查所有标准章节
# ==========================================

def validate_cases(section_map):

    missing = []

    for case in TEST_CASES:

        section_id = case[
            "gold_section"
        ]

        if section_id not in section_map:

            missing.append({
                "id": case["id"],
                "section": section_id
            })

        else:
            case["gold_section_title"] = (
                section_map[section_id]
            )

    return missing


# ==========================================
# 5. 创建评测集
# ==========================================

def main():

    print("=" * 60)
    print("第63步：构建独立检索测试集")
    print("=" * 60)

    section_map = load_sections()

    missing = validate_cases(
        section_map
    )

    if missing:

        print("\n以下标准章节尚未识别：")

        for item in missing:
            print(
                item["id"],
                item["section"]
            )

        raise ValueError(
            "请先核对缺失章节，"
            "本次不生成未经验证的标签"
        )

    dataset = {
        "dataset_name": (
            "Research RAG V3 Holdout"
        ),
        "source_pdf": (
            "HandbookChapter5.pdf"
        ),
        "split": "test",
        "annotation_level": "section",
        "question_count": len(
            TEST_CASES
        ),
        "annotation_status": (
            "Provisional - requires "
            "manual text verification"
        ),
        "notes": (
            "These questions were not used "
            "in Steps 56-62 for tuning. "
            "Gold labels identify primary "
            "sections rather than full answers."
        ),
        "questions": TEST_CASES
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            dataset,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "\n测试题数量：",
        len(TEST_CASES)
    )

    print(
        "已识别的章节数量：",
        len(section_map)
    )

    for case in TEST_CASES:

        print("\n", case["id"])

        print(
            "问题：",
            case["question"]
        )

        print(
            "标准章节：",
            case["gold_section_title"]
        )

    print("\n" + "=" * 60)
    print("测试集生成成功")
    print("=" * 60)

    print(
        "文件位置：",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
