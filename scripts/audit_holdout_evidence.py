
import hashlib
import json
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "results"
    / "pdf_v3_holdout_evaluation.json"
)

OUTPUT_PATH = (
    ROOT / "results"
    / "pdf_v3_evidence_audit.json"
)

OPTIONS = (
    "未评分",
    "0：证据不足",
    "1：部分支持",
    "2：足以回答"
)

SCORE_MAP = {
    OPTIONS[1]: 0,
    OPTIONS[2]: 1,
    OPTIONS[3]: 2
}


def load_report():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"找不到文件：{INPUT_PATH}"
        )

    data = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    if len(data.get("records", [])) != 7:
        raise ValueError("预期应有7道问题")

    return data


def method_order(question_id):
    """
    固定打乱两种方法的展示顺序，
    不因Streamlit刷新而改变。
    """

    value = hashlib.sha256(
        f"audit-{question_id}".encode()
    ).digest()[0]

    methods = ["baseline", "constrained"]

    if value % 2:
        methods.reverse()

    return methods


def display_sources(data):

    for item in data["top5"]:

        st.markdown(
            f"**Top {item['rank']}｜"
            f"第 {item['page']} 页**"
        )

        st.caption(item["section"])

        st.write(item["text"])

        st.divider()


st.set_page_config(
    page_title="RAG Evidence Audit",
    page_icon="📋",
    layout="wide"
)

st.title("📋 第65步：证据充分性评估")

st.info(
    "阅读每组 Top-5 证据，判断它们能否"
    "直接回答问题。评分过程中隐藏方法名称，"
    "提交后再计算两种方法的结果。"
)

st.markdown(
    "**0分：无法回答｜"
    "1分：只能部分回答｜"
    "2分：证据足以回答**"
)

report = load_report()

records = report["records"]

with st.form("evidence_audit_form"):

    for index, case in enumerate(
        records,
        start=1
    ):

        with st.expander(
            f"{case['id']}｜{case['question']}",
            expanded=(index == 1)
        ):

            st.caption(
                "暂定主要参考章节："
                + case["gold_section_title"]
            )

            columns = st.columns(2)

            order = method_order(case["id"])

            for column, method, label in zip(
                columns,
                order,
                ["方案甲", "方案乙"]
            ):

                with column:

                    st.subheader(label)

                    st.radio(
                        "证据充分性",
                        options=OPTIONS,
                        index=0,
                        key=(
                            f"score_"
                            f"{case['id']}_{method}"
                        ),
                        horizontal=True
                    )

                    st.text_area(
                        "评分依据（可选）",
                        placeholder=(
                            "例如：缺少关键公式，"
                            "只有背景介绍"
                        ),
                        key=(
                            f"note_"
                            f"{case['id']}_{method}"
                        ),
                        height=75
                    )

                    display_sources(
                        case[method]
                    )

    submitted = st.form_submit_button(
        "保存全部评估结果",
        type="primary"
    )


if submitted:

    annotations = []
    missing = []

    for case in records:

        entry = {
            "id": case["id"],
            "question": case["question"],
            "gold_section": (
                case["gold_section"]
            ),
            "ratings": {}
        }

        order = method_order(
            case["id"]
        )

        for index, method in enumerate(order):

            choice = st.session_state.get(
                f"score_{case['id']}_{method}",
                "未评分"
            )

            if choice not in SCORE_MAP:

                missing.append(
                    f"{case['id']} "
                    f"方案{'甲' if index == 0 else '乙'}"
                )

                continue

            entry["ratings"][method] = {
                "score": SCORE_MAP[choice],
                "note": st.session_state.get(
                    f"note_{case['id']}_{method}",
                    ""
                )
            }

        annotations.append(entry)

    if missing:

        st.warning(
            "还有未评分项："
            + "、".join(missing)
        )

    else:

        def get_scores(method):

            return [
                item["ratings"][method]["score"]
                for item in annotations
            ]

        baseline_scores = get_scores(
            "baseline"
        )

        constrained_scores = get_scores(
            "constrained"
        )

        baseline_sufficient = sum(
            value == 2
            for value in baseline_scores
        )

        constrained_sufficient = sum(
            value == 2
            for value in constrained_scores
        )

        better = sum(
            c > b
            for b, c in zip(
                baseline_scores,
                constrained_scores
            )
        )

        worse = sum(
            c < b
            for b, c in zip(
                baseline_scores,
                constrained_scores
            )
        )

        equal = 7 - better - worse

        output = {
            "experiment": (
                "V3 Holdout Evidence Sufficiency Audit"
            ),
            "source_experiment": (
                report["experiment"]
            ),
            "filename": report["filename"],
            "pdf_hash": report["pdf_hash"],
            "question_count": 7,
            "rating_scale": {
                "0": "Insufficient",
                "1": "Partially sufficient",
                "2": "Sufficient to answer"
            },
            "annotation_note": (
                "Single-reviewer manual ratings; "
                "method display order was masked. "
                "Gold section labels remain "
                "provisional."
            ),
            "baseline_summary": {
                "average_score": (
                    sum(baseline_scores) / 7
                ),
                "sufficient_count": (
                    baseline_sufficient
                ),
                "sufficient_rate": (
                    baseline_sufficient / 7
                )
            },
            "constrained_summary": {
                "average_score": (
                    sum(constrained_scores) / 7
                ),
                "sufficient_count": (
                    constrained_sufficient
                ),
                "sufficient_rate": (
                    constrained_sufficient / 7
                )
            },
            "comparison": {
                "constrained_better": better,
                "constrained_worse": worse,
                "equal": equal
            },
            "annotations": annotations
        }

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        OUTPUT_PATH.write_text(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        st.session_state["audit_output"] = output


audit_output = st.session_state.get(
    "audit_output"
)

if audit_output:

    st.success("人工评估结果已保存！")

    a, b, c = st.columns(3)

    a.metric(
        "原始 V3：充分证据率",
        f"{audit_output['baseline_summary']['sufficient_rate']:.1%}"
    )

    b.metric(
        "受约束重排序：充分证据率",
        f"{audit_output['constrained_summary']['sufficient_rate']:.1%}"
    )

    c.metric(
        "重排序更优/更差/持平",
        (
            f"{audit_output['comparison']['constrained_better']}"
            f" / {audit_output['comparison']['constrained_worse']}"
            f" / {audit_output['comparison']['equal']}"
        )
    )

    st.download_button(
        "下载证据充分性报告",
        data=json.dumps(
            audit_output,
            ensure_ascii=False,
            indent=2
        ).encode("utf-8"),
        file_name="pdf_v3_evidence_audit.json",
        mime="application/json"
    )

    st.caption(
        "本结果为单人主观评估，"
        "并非模型回答准确率。"
    )
