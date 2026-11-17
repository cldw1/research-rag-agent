"""Verify actual generation attempts and export numeric diagnostics and source audits."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.answer_review import review_pack, summarize_reviews
from src.evaluation.pdf_answers import collect_results, read_json, summarize_answers, write_new
from src.evaluation.pdf_benchmark import sha256


def export_answers(run, retrieval_run, sources_root, output, *, reviews_file=None, rubric_file=None):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    snapshot_verified = (run / "source_snapshot").is_dir()
    if snapshot_verified:
        for name, expected in manifest["source_code_sha256"].items():
            path = (run / "source_snapshot" / name).resolve()
            if (run / "source_snapshot").resolve() not in path.parents or sha256(path.read_bytes()) != expected:
                raise ValueError("Generation source snapshot changed")
    requests = read_json(run / "requests.json")
    pack = review_pack(run, retrieval_run, sources_root)
    results = collect_results(run, requests)
    summary = summarize_answers(results, manifest)
    dataset = read_json(run / "dataset.json")
    if len(results) != 45 or set(manifest["question_ids"]) != {r["question_id"] for r in results}:
        raise ValueError("Expected unchanged 45-question development population")
    for record in results:
        usage = record.get("usage", {})
        if any(type(value) is not int or value < 0 for value in usage.values()):
            raise ValueError("Invalid provider token usage")
        if all(k in usage for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            if usage["prompt_tokens"]+usage["completion_tokens"] != usage["total_tokens"]:
                raise ValueError("Inconsistent provider token accounting")
    review_summary, reviews = None, None
    if reviews_file:
        if rubric_file is None:
            raise ValueError("Review rubric required")
        reviews, rubric = read_json(reviews_file), read_json(rubric_file)
        review_summary = summarize_reviews(pack, reviews, rubric)
        if not review_summary["complete"]:
            raise ValueError("Partial reviews must not be presented as complete")
    report = {**summary, "run_id": run.name, "variant": manifest["variant"],
        "generation_source_snapshot_verified": snapshot_verified,
        "model_config": manifest["model_config"], "thinking": manifest["thinking"],
        "retrieval_run_id": manifest["retrieval_run_id"], "dataset_sha256": manifest["dataset_sha256"],
        "observed_response_models": sorted({r.get("response_metadata", {}).get("model") for r in results if r.get("response_metadata", {}).get("model")}),
        "price_reference": manifest["price_reference"], "transport": manifest["transport"],
        "answerable_count": sum(q["answerable"] for q in dataset["questions"]),
        "unanswerable_count": sum(not q["answerable"] for q in dataset["questions"]),
        "source_review": review_summary,
        "review_status": "ai_source_audited_human_pending" if reviews and reviews["reviewer_type"] == "ai_source_audit" else "pending",
        "artifacts_sha256": {name: sha256((run / name).read_bytes()) for name in ("manifest.json", "requests.json", "dataset.json")},
        "attempt_result_sha256": {r["question_id"]: sha256((run / "attempts" / r["question_id"] / "result.json").read_bytes())
                                  for r in results if (run / "attempts" / r["question_id"] / "result.json").exists()}}
    if reviews:
        report["reviews_sha256"] = sha256(Path(reviews_file).read_bytes())
        report["rubric_sha256"] = sha256(Path(rubric_file).read_bytes())
        report["review_records"] = reviews["records"]
        report["reviewer_type"] = reviews["reviewer_type"]
    # No answers, raw passages, credentials or provider request headers in public report.
    write_new(output, report)
    sheet = run / "human_answer_review.csv"
    if not sheet.exists():
        by_id = {r["question_id"]: r for r in pack}
        ai_by_id = {r["question_id"]: r for r in reviews["records"]} if reviews else {}
        with sheet.open("x", encoding="utf-8-sig", newline="") as stream:
            fields = ["question_id", "question", "answerable", "reference_points", "answer", "answer_sha256",
                "ai_correctness_0_2", "ai_faithfulness_0_2", "ai_reason", "human_correctness_0_2",
                "human_faithfulness_0_2", "human_citation_support_0_2", "human_citation_coverage_0_2", "human_reviewer", "human_notes"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for qid in manifest["question_ids"]:
                r, ai = by_id[qid], ai_by_id.get(qid, {})
                writer.writerow({key: r.get(key, "") for key in ("question_id", "question", "answerable", "answer", "answer_sha256")} |
                    {"reference_points": " | ".join(r.get("reference_points", [])),
                     "ai_correctness_0_2": ai.get("correctness", ""), "ai_faithfulness_0_2": ai.get("faithfulness", ""), "ai_reason": ai.get("reason", "")})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--retrieval-run", type=Path, default=ROOT / "data/pdf_benchmark/runs/20261115T165227Z-1bd0349b")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/pdf_benchmark/sources")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--rubric", type=Path, default=ROOT / "benchmarks/pdf_qa/answer_review_rubric.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_answers(args.run, args.retrieval_run, args.sources, args.output,
                            reviews_file=args.reviews, rubric_file=args.rubric)
    print(json.dumps({"output": str(args.output), "generation_api_attempts": result["generation_api_attempts"],
                      "status_counts": result["status_counts"], "review_status": result["review_status"]}))


if __name__ == "__main__":
    main()
