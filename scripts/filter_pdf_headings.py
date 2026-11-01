
import json
import re
from pathlib import Path
from collections import defaultdict


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pdf_heading_diagnosis.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pdf_headings_filtered.json"
)


# ==========================================
# 2. 章节识别规则
# ==========================================

# 匹配：
# 5.1 Introduction
# 5.2.1 Hamilton's Equations
# 5.4.2 Tuning HMC
# 5.2.1.1 Equations of Motion

NUMBERED_HEADING_RE = re.compile(
    r"^([1-9]\d?(?:\.\d+){1,4})\s+(.+)$"
)

SPECIAL_HEADINGS = {
    "abstract",
    "introduction",
    "references",
    "bibliography",
    "conclusion",
    "conclusions",
    "related work",
    "acknowledgments",
    "acknowledgements"
}


# ==========================================
# 3. 读取诊断文件
# ==========================================

def load_report():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"找不到诊断文件：{INPUT_PATH}"
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# ==========================================
# 4. 筛选高置信度标题
# ==========================================

def filter_headings(candidates):

    numbered = []
    standalone = []
    discarded = []

    # --------------------------------------
    # A. 优先保留独立编号且加粗的标题
    # --------------------------------------

    for item in candidates:

        title = item["candidate"].strip()
        kind = item["type"]
        bold = item["bold"]

        match = NUMBERED_HEADING_RE.fullmatch(
            title
        )

        if (
            kind == "编号与标题可能分行"
            and bold
            and match is not None
        ):

            numbered.append({
                **item,
                "section_number": match.group(1),
                "section_title": match.group(2),
                "confidence": "high"
            })

        elif (
            kind == "独立通用标题"
            and bold
            and title.casefold()
            in SPECIAL_HEADINGS
        ):

            standalone.append(item)

        else:

            discarded.append({
                **item,
                "reason": (
                    "缺少可靠的标题排版特征"
                )
            })

    # --------------------------------------
    # B. 去掉重复的通用标题
    # --------------------------------------

    accepted = list(numbered)

    for item in standalone:

        title = item["candidate"].strip()

        duplicate = any(
            heading["page"] == item["page"]
            and heading["section_title"].casefold()
            == title.casefold()
            for heading in numbered
        )

        if duplicate:

            discarded.append({
                **item,
                "reason": "与带编号标题重复"
            })

        else:

            accepted.append({
                **item,
                "section_number": None,
                "section_title": title,
                "confidence": "high"
            })

    # 按原始 PDF 顺序排序
    accepted.sort(
        key=lambda item: (
            item["page"],
            item["y_position"]
        )
    )

    return accepted, discarded


# ==========================================
# 5. 检查章节编号是否有缺失
# ==========================================

def find_number_gaps(headings):

    siblings = defaultdict(set)

    for heading in headings:

        number = heading["section_number"]

        if number is None:
            continue

        parts = tuple(
            int(x)
            for x in number.split(".")
        )

        parent = parts[:-1]
        child = parts[-1]

        siblings[parent].add(child)

    missing = []

    for parent, children in siblings.items():

        if len(children) < 2:
            continue

        for child in range(
            min(children),
            max(children) + 1
        ):

            if child in children:
                continue

            number = ".".join(
                str(x)
                for x in (*parent, child)
            )

            missing.append(number)

    return sorted(missing)


# ==========================================
# 6. 主程序
# ==========================================

def main():

    print("=" * 60)
    print("PDF 高置信度章节标题筛选")
    print("=" * 60)

    data = load_report()

    candidates = data["candidates"]

    headings, discarded = filter_headings(
        candidates
    )

    gaps = find_number_gaps(headings)

    numbered_count = sum(
        item["section_number"] is not None
        for item in headings
    )

    print("\n原始候选数：", len(candidates))
    print("筛选后候选数：", len(headings))
    print("带编号标题：", numbered_count)
    print(
        "独立通用标题：",
        len(headings) - numbered_count
    )
    print("排除候选数：", len(discarded))

    print("\n" + "=" * 60)
    print("筛选后的章节")
    print("=" * 60)

    for item in headings:

        print(
            f"第 {item['page']:02d} 页 | "
            f"{item['candidate']}"
        )

    print("\n" + "=" * 60)
    print("可能遗漏的章节编号")
    print("=" * 60)

    if gaps:

        for number in gaps:
            print(number)

    else:
        print("未发现编号间断")

    # --------------------------------------
    # 保存结果
    # --------------------------------------

    report = {
        "filename": data["filename"],
        "original_candidate_count": len(
            candidates
        ),
        "accepted_count": len(headings),
        "numbered_count": numbered_count,
        "discarded_count": len(discarded),
        "possible_missing_sections": gaps,
        "headings": headings,
        "discarded": discarded
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

    print("\n筛选报告已保存：")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
