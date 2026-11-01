"""Check published metrics, Markdown paths and Git safety behavior offline."""

import ast
import json
import math
import re
from pathlib import Path

import pytest

from scripts.export_experiment_summary import summarize


ROOT = Path(__file__).resolve().parents[1]


def test_public_summary_metrics_recompute_from_ranked_sections_and_manual_scores():
    summary = json.loads((ROOT / "results/hmc_v3_public_summary.json").read_text(encoding="utf-8"))
    assert summary["question_count"] == len(summary["records"]) == 7
    for method in ("hybrid", "section_aware"):
        ranks = []
        for row in summary["records"]:
            ranked = row[method]["top5"]
            rank = next((c["rank"] for c in ranked if re.match(r"^(\d+(?:\.\d+)+)(?![\d.])", c["section"].strip()).group(1) == row["gold_section"]), None)
            assert rank == row[method]["gold_rank"]
            ranks.append(rank)
        metrics = summary["metrics"][method]
        assert math.isclose(metrics["hit_at_5"], sum(r is not None for r in ranks) / 7)
        assert math.isclose(metrics["mrr_at_5"], sum(1 / r if r else 0 for r in ranks) / 7)
        assert metrics["sufficient_count"] == sum(r[method]["evidence_score"] == 2 for r in summary["records"]) == 6

    def check_keys(obj):
        if isinstance(obj, dict):
            assert not ({"text", "content", "messages", "api_key", "preview_chunks"} & obj.keys())
            for value in obj.values():
                check_keys(value)
        elif isinstance(obj, list):
            for value in obj:
                check_keys(value)
    check_keys(summary)


def test_export_refuses_inconsistent_historical_data():
    with pytest.raises(ValueError):
        summarize({"records": [], "question_count": 0}, {"annotations": [], "question_count": 0})


def test_markdown_local_links_exist():
    files = [ROOT / "README.md", *ROOT.glob("docs/**/*.md")]
    for path in files:
        content = path.read_text(encoding="utf-8")
        content = re.sub(r"```.*?```", "", content, flags=re.S)
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if re.match(r"[a-zA-Z][\w+.-]*:|#", target):
                continue
            target = target.split("#", 1)[0]
            assert (path.parent / target).exists(), f"{path.name}: missing {target}"


def test_all_project_python_files_parse():
    files = [ROOT / "app.py", *ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")]
    for path in files:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


@pytest.mark.parametrize("filename", ["config.py", ".env.example"])
def test_secret_scanner_fails_without_printing_token(monkeypatch, tmp_path, capsys, filename):
    from scripts import check_github_safety as scanner
    path = tmp_path / filename
    token = "sk-" + "A" * 32
    path.write_text('credential = "' + token + '"\n', encoding="utf-8")
    monkeypatch.setattr(scanner, "ROOT", tmp_path)
    monkeypatch.setattr(scanner, "get_candidate_files", lambda: [path])
    assert scanner.main() == 1
    output = capsys.readouterr().out
    assert token not in output
    assert filename in output
