"""Verify every PDF identity and evidence coordinate without a model or API key."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.pdf_benchmark import load_dataset, sha256, verify_sources, verify_candidate_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/pdf_qa/pilot.json")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/pdf_benchmark/sources")
    parser.add_argument("--review-output", type=Path, help="Write a new Markdown annotation review sheet; never overwrite")
    parser.add_argument("--release-root", type=Path, help="Also verify selection, full corpus and exact development/test views")
    parser.add_argument("--output", type=Path, help="Write a new JSON audit report; never overwrite an existing one")
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    _, result = verify_sources(dataset, args.sources)
    if args.release_root:
        result["candidate_release"] = verify_candidate_release(args.release_root)
    if args.review_output:
        papers = {p["paper_id"]: p for p in dataset["papers"]}
        lines = ["# PDF 问答题目与标注审核", "",
                 "本表由已通过来源校验的标注生成。AI 已核对来源，独立人工复核待完成；预留测试题不参与开发评分。", "",
                 "定位页码为 PDF 阅读器从 1 开始的页码。参考答案是评分要点，不是模型生成回答。", ""]
        for q in dataset["questions"]:
            paper = papers[q["paper_id"]]
            lines += [f"## {q['question_id']}", "", f"论文：{paper['title']}（{paper['arxiv_version']}）",
                      "", f"划分：`{paper['split']}`；类型：`{q['category']}`；可回答：{q['answerable']}；人工复核：`{q['annotation']['human_review']}`",
                      "", q["question"], "", "参考要点：", ""]
            lines += ["- " + p for p in q["reference_points"]]
            lines += ["", "原文定位：", ""]
            lines += [f"- {a['evidence_id']}：PDF 第 {a['page_number']} 页，章节 {a['section']}，页内块 {a['block_index']}，字符 [{a['char_start']}, {a['char_end']})。"
                      for a in q["evidence"]] or ["- 未报告所问具体信息；需独立复核全文。"]
            if q.get("review_note"):
                lines += ["", "核查备注：" + q["review_note"]]
            lines += ["", "人工复核记录：复核者 ______ 日期 ______ 问题/答案/证据修改 ______", ""]
        with args.review_output.open("x", encoding="utf-8") as stream:
            stream.write("\n".join(lines))
    result.update(dataset_sha256=sha256(args.dataset.read_bytes()),
                  answerable=sum(q["answerable"] for q in dataset["questions"]),
                  unanswerable=sum(not q["answerable"] for q in dataset["questions"]),
                  categories=dict(Counter(q["category"] for q in dataset["questions"])),
                  human_review_pending=sum(q["annotation"]["human_review"] == "pending" for q in dataset["questions"]),
                  generation_api_calls=0)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
