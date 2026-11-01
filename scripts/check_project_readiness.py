
import ast
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ==========================================
# 1. 项目路径
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    ROOT / "results"
    / "project_readiness_report.json"
)


# ==========================================
# 2. 核心项目文件
# ==========================================

CORE_FILES = [
    "app.py",
    "src/agent/paper_search_tool.py",
    "src/generation/rag_pipeline.py",
    "src/generation/strict_context_builder.py",
    "src/retrieval/uploaded_pdf_retriever.py",
    "src/retrieval/context_expander.py",
    "src/retrieval/section_aware_reranker.py",
    "src/retrieval/retrieval_strategy.py",
    "src/retrieval/strategy_proxy.py",
    "src/retrieval/verified_v3_pdf_retriever.py",
    "src/retrieval/smart_section_expander.py",
    "src/ingestion/pdf_chunker_v3.py",
]


# ==========================================
# 3. 已有的基础测试脚本
# ==========================================

TEST_SCRIPTS = [
    "scripts/test_section_aware_reranker.py",
    "scripts/test_retrieval_strategy.py",
    "scripts/test_step69_strategy_proxy.py",
    "scripts/test_step70_smart_expander.py",
]


# ==========================================
# 4. 历史实验报告
# ==========================================

MAPPING_PATH = (
    ROOT / "results"
    / "pdf_v3_section_mapping_report.json"
)

EVALUATION_PATH = (
    ROOT / "results"
    / "pdf_v3_holdout_evaluation.json"
)

AUDIT_PATH = (
    ROOT / "results"
    / "pdf_v3_evidence_audit.json"
)

REGRESSION_PATH = (
    ROOT / "results"
    / "pdf_v3_strategy_real_pdf_regression.json"
)


def read_json(path):

    if not path.is_file():
        raise FileNotFoundError(str(path))

    return json.loads(
        path.read_text(encoding="utf-8")
    )


# ==========================================
# 5. 检查文件和 Python 语法
# ==========================================

def check_core_files():

    records = []

    for relative_path in CORE_FILES:

        path = ROOT / relative_path

        record = {
            "file": relative_path,
            "exists": path.is_file(),
            "syntax_ok": False,
            "error": None
        }

        if path.is_file():

            try:

                source = path.read_text(
                    encoding="utf-8-sig"
                )

                ast.parse(
                    source,
                    filename=str(path)
                )

                record["syntax_ok"] = True

            except Exception as exc:

                record["error"] = str(exc)

        records.append(record)

    return records


# ==========================================
# 6. 运行已有轻量测试
# ==========================================

def run_unit_tests():

    results = []

    for relative_path in TEST_SCRIPTS:

        path = ROOT / relative_path

        if not path.is_file():

            results.append({
                "file": relative_path,
                "status": "missing",
                "output": "未找到测试文件"
            })

            continue

        try:

            process = subprocess.run(
                [
                    sys.executable,
                    str(path)
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90
            )

            output = (
                process.stdout
                + "\n"
                + process.stderr
            )

            results.append({
                "file": relative_path,
                "status": (
                    "passed"
                    if process.returncode == 0
                    else "failed"
                ),
                "exit_code": process.returncode,
                "output": output[-4000:]
            })

        except subprocess.TimeoutExpired:

            results.append({
                "file": relative_path,
                "status": "timeout",
                "output": "运行超过90秒"
            })

        except Exception as exc:

            results.append({
                "file": relative_path,
                "status": "error",
                "output": str(exc)
            })

    return results


# ==========================================
# 7. 核对 V3 实验报告
# ==========================================

def check_experiments():

    checks = {}

    # --------------------------------------
    # 章节分块报告
    # --------------------------------------

    try:

        mapping = read_json(MAPPING_PATH)

        checks["v3_mapping"] = {
            "passed": (
                mapping.get("integrity_passed") is True
                and mapping.get("chunk_count") == 247
                and len(
                    mapping.get("matched_events", [])
                ) == 50
                and not mapping.get(
                    "unmatched_headings"
                )
            ),
            "chunk_count": mapping.get(
                "chunk_count"
            ),
            "matched_headings": len(
                mapping.get("matched_events", [])
            )
        }

    except Exception as exc:

        checks["v3_mapping"] = {
            "passed": False,
            "error": str(exc)
        }

    # --------------------------------------
    # 七道独立测试题指标
    # --------------------------------------

    try:

        evaluation = read_json(
            EVALUATION_PATH
        )

        baseline = evaluation[
            "baseline_metrics"
        ]

        constrained = evaluation[
            "constrained_metrics"
        ]

        checks["holdout_evaluation"] = {
            "passed": (
                evaluation.get("question_count") == 7
                and math.isclose(
                    baseline["hit_at_5"],
                    6 / 7,
                    abs_tol=1e-9
                )
                and math.isclose(
                    constrained["hit_at_5"],
                    1.0,
                    abs_tol=1e-9
                )
                and math.isclose(
                    baseline["mrr_at_5"],
                    0.7857142857142857,
                    abs_tol=1e-9
                )
                and math.isclose(
                    constrained["mrr_at_5"],
                    0.9285714285714286,
                    abs_tol=1e-9
                )
            ),
            "baseline": baseline,
            "constrained": constrained
        }

    except Exception as exc:

        checks["holdout_evaluation"] = {
            "passed": False,
            "error": str(exc)
        }

    # --------------------------------------
    # 人工证据充分性评估
    # --------------------------------------

    try:

        audit = read_json(AUDIT_PATH)

        original = audit[
            "baseline_summary"
        ]

        constrained = audit[
            "constrained_summary"
        ]

        checks["evidence_audit"] = {
            "passed": (
                audit.get("question_count") == 7
                and original["sufficient_count"] == 6
                and constrained["sufficient_count"] == 6
            ),
            "baseline": original,
            "constrained": constrained,
            "comparison": audit.get(
                "comparison"
            )
        }

    except Exception as exc:

        checks["evidence_audit"] = {
            "passed": False,
            "error": str(exc)
        }

    # --------------------------------------
    # 第68步真实 PDF 回归测试
    # --------------------------------------

    try:

        regression = read_json(
            REGRESSION_PATH
        )

        checks["real_pdf_regression"] = {
            "passed": (
                regression.get(
                    "regression_passed"
                ) is True
                and regression.get(
                    "all_rankings_match_step64"
                ) is True
                and regression.get(
                    "metrics_match_step64"
                ) is True
                and regression.get(
                    "all_constraints_pass"
                ) is True
            )
        }

    except Exception as exc:

        checks["real_pdf_regression"] = {
            "passed": False,
            "error": str(exc)
        }

    return checks


# ==========================================
# 8. 主程序
# ==========================================

def main():

    print("=" * 65)
    print("第71步：Research RAG Agent 自动化验收")
    print("=" * 65)

    core_files = check_core_files()

    tests = run_unit_tests()

    experiments = check_experiments()

    print("\n[1] 核心文件")

    for item in core_files:

        status = (
            "PASS"
            if item["exists"] and item["syntax_ok"]
            else "FAIL"
        )

        print(
            f"[{status}] {item['file']}"
        )

        if item["error"]:
            print("  错误：", item["error"])

    print("\n[2] 基础测试")

    for item in tests:

        print(
            f"[{item['status'].upper()}] "
            f"{item['file']}"
        )

        if item["status"] != "passed":
            print(item.get("output", ""))

    print("\n[3] 实验报告核对")

    for name, item in experiments.items():

        status = (
            "PASS"
            if item["passed"]
            else "FAIL"
        )

        print(
            f"[{status}] {name}"
        )

        if item.get("error"):
            print(
                "  错误：",
                item["error"]
            )

    # ======================================
    # 9. 汇总
    # ======================================

    file_passed = all(
        item["exists"] and item["syntax_ok"]
        for item in core_files
    )

    tests_passed = all(
        item["status"] == "passed"
        for item in tests
    )

    experiments_passed = all(
        item["passed"]
        for item in experiments.values()
    )

    all_passed = (
        file_passed
        and tests_passed
        and experiments_passed
    )

    report = {
        "project": "Research RAG Agent",
        "check_time": datetime.now().astimezone().isoformat(),
        "validation_scope": (
            "Static files, lightweight unit tests, "
            "and historical experiment consistency. "
            "Not a full live Agent test."
        ),
        "core_files": core_files,
        "unit_tests": tests,
        "experiments": experiments,
        "summary": {
            "files_passed": file_passed,
            "tests_passed": tests_passed,
            "experiments_passed": (
                experiments_passed
            ),
            "all_passed": all_passed
        }
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("验收结果")
    print("=" * 65)

    if all_passed:
        print("基础自动化验收通过！")
    else:
        print("存在未通过的检查，请查看上方明细。")

    print(
        "报告保存位置：",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
