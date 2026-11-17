"""Export the replayed retry comparison; no generation or judge API calls."""

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.pdf_answers import read_json, write_new
from src.evaluation.pdf_retry import retry_report


def export_retry(run, baseline, retrieval, sources, root, reviews, output):
    report, pack = retry_report(run, baseline, retrieval, sources, root, reviews_file=reviews)
    if report["incremental_generation"]["status_counts"].get("pending", 0):
        raise ValueError("Cannot publish a partial execution as the finished comparison")
    if not report["policy_ai_review"] or not report["policy_ai_review"]["complete"]:
        raise ValueError("Complete effective-answer review required")
    write_new(output, report)
    sheet = Path(run) / "human_answer_review.csv"
    if not sheet.exists():
        by_id = {r["question_id"]: r for r in read_json(reviews)["records"]}
        with sheet.open("x", encoding="utf-8-sig", newline="") as stream:
            fields = ["question_id", "question", "answerable", "reference_points", "answer", "answer_sha256",
                "ai_correctness_0_2", "ai_faithfulness_0_2", "ai_reason", "human_correctness_0_2",
                "human_faithfulness_0_2", "human_citation_support_0_2", "human_citation_coverage_0_2", "human_reviewer", "human_notes"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in pack:
                ai = by_id[row["question_id"]]
                writer.writerow({key: row[key] for key in ("question_id", "question", "answerable", "answer", "answer_sha256")} |
                    {"reference_points": " | ".join(row["reference_points"]), "ai_correctness_0_2": ai["correctness"],
                     "ai_faithfulness_0_2": ai["faithfulness"], "ai_reason": ai["reason"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/pdf_benchmark/answer_runs/20261115T171727Z-a10a57a1")
    parser.add_argument("--retrieval-run", type=Path, default=ROOT / "data/pdf_benchmark/runs/20261115T165227Z-1bd0349b")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/pdf_benchmark/sources")
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export_retry(args.run, args.baseline, args.retrieval_run, args.sources, ROOT, args.reviews, args.output)
    print({"output": str(args.output), "correctness_changes": report["correctness_changes"],
           "incremental_calls": report["incremental_generation"]["generation_api_attempts"]})


if __name__ == "__main__":
    main()
