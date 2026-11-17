"""Recompute a development PDF run before exporting a small public result."""

import argparse
import json
import math
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.pdf_benchmark import evidence_coverage, load_dataset, retrieval_metrics, sha256


def export_result(run, output):
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary.get("source_snapshot_saved"):
        for name, expected in summary["source_code_sha256"].items():
            path = (run / "source_snapshot" / name).resolve()
            if (run / "source_snapshot").resolve() not in path.parents or sha256(path.read_bytes()) != expected:
                raise ValueError("Source snapshot changed")
    for name in ("dataset.json", "protocol.json", "alignments.json", "records.json", "chunks.json", "generation_requests.jsonl"):
        if sha256((run / name).read_bytes()) != summary["artifacts_sha256"][name]:
            raise ValueError("Run checksum mismatch: " + name)
    dataset = load_dataset(run / "dataset.json")
    if (dataset["status"] != "development_pilot" or any(p["split"] != "dev" for p in dataset["papers"])
            or summary["generation_api_calls"] != 0 or summary["answer_metrics"] is not None):
        raise ValueError("Expected retrieval-only development pilot")
    records = json.loads((run / "records.json").read_text(encoding="utf-8"))
    chunks = json.loads((run / "chunks.json").read_text(encoding="utf-8"))
    alignments = json.loads((run / "alignments.json").read_text(encoding="utf-8"))
    by_question = {q["question_id"]: q for q in dataset["questions"]}
    if (len(records) != len(by_question) or {r["question_id"] for r in records} != set(by_question)
            or len({r["question_id"] for r in records}) != len(records)):
        raise ValueError("Question population changed")
    for record in records:
        question = by_question[record["question_id"]]
        if any(record[key] != question[key] for key in ("paper_id", "category", "answerable")):
            raise ValueError("Question metadata changed")
        aligned = [a for a in alignments if a["question_id"] == record["question_id"]]
        if len(aligned) != len(question["evidence"]):
            raise ValueError("Gold evidence denominator changed")
        all_chunks = chunks[record["paper_id"]]
        by_id = {c["chunk_id"]: c for c in all_chunks}
        for method, result in record["methods"].items():
            if len(set(result["ranked_chunk_ids"])) != len(result["ranked_chunk_ids"]):
                raise ValueError("Duplicate ranked chunk")
            ranked = [by_id[key] for key in result["ranked_chunk_ids"]]
            expected = {str(k): retrieval_metrics(all_chunks, ranked, aligned, k) for k in (1, 5, 10)} if aligned else None
            if expected != result["metrics"]:
                raise ValueError("Per-question retrieval metrics do not reproduce")
        for context in record["contexts"].values():
            selected = [by_id[s["chunk_id"]] for s in context["sources"]]
            expected = evidence_coverage(selected, aligned) if aligned else None
            if expected != context["metrics"]:
                raise ValueError("Context coverage does not reproduce")
    answerable = [r for r in records if r["answerable"]]
    if (summary["question_count"] != len(records) or summary["answerable_count"] != len(answerable)
            or summary["unanswerable_count"] != len(records)-len(answerable)
            or summary["paper_count"] != len(dataset["papers"])):
        raise ValueError("Summary denominator changed")
    if (summary["alignment"] != dict(Counter(a["status"] for a in alignments))
            or summary["human_review_pending"] != sum(q["annotation"]["human_review"] == "pending" for q in dataset["questions"])
            or any(b["chunks"] != len(chunks[b["paper_id"]]) for b in summary["builds"])):
        raise ValueError("Source or chunk counts changed")
    for method in ("bm25", "dense", "hybrid"):
        for k in ("1", "5", "10"):
            for key, value in summary["methods"][method]["metrics"][k].items():
                computed = sum(r["methods"][method]["metrics"][k][key] for r in answerable)/len(answerable)
                if not math.isclose(computed, value, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("Summary metrics do not reproduce")
        for field, population_key in (("per_paper_at_5", "paper_id"), ("per_category_at_5", "category")):
            for label, metrics in summary["methods"][method][field].items():
                group = [r for r in answerable if r[population_key] == label]
                for key, value in metrics.items():
                    computed = sum(r["methods"][method]["metrics"]["5"][key] for r in group)/len(group)
                    if not math.isclose(computed, value, rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError("Grouped metrics do not reproduce")
    for variant, metrics in summary["contexts"].items():
        for key, value in metrics.items():
            computed = sum(r["contexts"][variant]["metrics"][key] for r in answerable)/len(answerable)
            if not math.isclose(computed, value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Context summary does not reproduce")
    if (summary["dataset_sha256"] != sha256((run / "dataset.json").read_bytes())
            or summary["protocol_sha256"] != sha256((run / "protocol.json").read_bytes())):
        raise ValueError("Protocol or dataset file bytes differ from run snapshot")
    result = {**summary, "run_id": run.name,
              "papers": [{key: p[key] for key in ("paper_id", "title", "arxiv_version", "pdf_sha256", "page_count", "split")}
                         for p in dataset["papers"]],
              "bad_cases": [{"question_id": r["question_id"], "paper_id": r["paper_id"],
                  "hybrid_at_5": r["methods"]["hybrid"]["metrics"]["5"]}
                  for r in answerable if not r["methods"]["hybrid"]["metrics"]["5"]["all_evidence_covered"]]}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New output file; existing results are never overwritten")
    args = parser.parse_args()
    report = export_result(args.run, args.output)
    print(json.dumps({"output": str(args.output), "questions": report["question_count"], "verified": True}))


if __name__ == "__main__":
    main()
