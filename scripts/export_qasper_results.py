"""Validate a complete benchmark run and export text-free public results."""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(content):
    return hashlib.sha256(content).hexdigest()


def export_run(directory, destination, prefix="qasper_test"):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", prefix):
        raise ValueError("Invalid output prefix")
    raw = (directory / "per_question.jsonl").read_bytes()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if sha(raw) != summary["artifacts"]["per_question_jsonl_sha256"]:
        raise ValueError("Per-question checksum mismatch")
    if sha((directory / "protocol.json").read_bytes()) != summary["artifacts"]["protocol_sha256"]:
        raise ValueError("Protocol checksum mismatch")
    if json.loads((directory / "protocol.json").read_text(encoding="utf-8")) != summary["protocol"]:
        raise ValueError("Summary protocol differs from the frozen configuration")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if (len(records) != summary["question_count"] or len({r["question_id"] for r in records}) != len(records)
            or len({r["paper_id"] for r in records}) != summary["paper_count"]):
        raise ValueError("Incomplete or duplicate evaluation records")
    metrics = ("hit", "precision", "mrr", "ndcg", "recall", "full_paragraph_recall", "all_gold_covered", "evidence_f1")
    for method in summary["protocol"]["methods"]:
        for k in summary["protocol"]["k_values"]:
            for metric in metrics:
                values = [r["methods"][method][str(k)][metric] for r in records]
                if any(not math.isfinite(v) or not 0 <= v <= 1 + 1e-12 for v in values):
                    raise ValueError("Invalid metric value")
                mean = sum(values) / len(values)
                if not math.isclose(mean, summary["methods"][method]["metrics"][str(k)][metric], rel_tol=0, abs_tol=1e-12):
                    raise ValueError("Summary differs from individual questions")
    for variant, values in summary["contexts"].items():
        for key, expected in values.items():
            mean = sum(r["contexts"][variant][key] for r in records) / len(records)
            if not math.isclose(mean, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Context summary differs from individual questions")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, ["paper_id", "question_id", "method", "k", *metrics], lineterminator="\n")
    writer.writeheader()
    for record in records:
        for method in summary["protocol"]["methods"]:
            writer.writerow({"paper_id": record["paper_id"], "question_id": record["question_id"],
                             "method": method, "k": 5,
                             **{key: record["methods"][method]["5"][key] for key in metrics}})
    csv_content = output.getvalue().encode("utf-8")
    patterns = {
        "all_methods_miss_at_5": lambda h: not any(h.values()),
        "bm25_miss_dense_hit": lambda h: not h["bm25"] and h["dense"],
        "dense_miss_bm25_hit": lambda h: not h["dense"] and h["bm25"],
        "hybrid_miss_dense_hit": lambda h: not h["hybrid"] and h["dense"],
        "hybrid_hit_dense_miss": lambda h: h["hybrid"] and not h["dense"],
    }
    bad_cases = {}
    for name, condition in patterns.items():
        chosen = [r for r in records if condition({m: r["methods"][m]["5"]["hit"] for m in ("bm25", "dense", "hybrid")})]
        bad_cases[name] = {"count": len(chosen), "examples": [
            {"paper_id": r["paper_id"], "question_id": r["question_id"], "ranked_chunk_ordinals": r["ranks"]}
            for r in chosen[:10]]}
    bad_content = (json.dumps(bad_cases, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    public = {key: deepcopy(summary[key]) for key in (
        "protocol", "corpus", "question_count", "paper_count", "methods", "paired_hybrid_minus_dense_at_5",
        "contexts", "timings", "generation_api_calls", "scope", "artifacts")}
    public["public_artifacts"] = {"per_question_at_5": prefix + "_per_question.csv",
        "per_question_at_5_sha256": sha(csv_content), "bad_cases": prefix + "_bad_cases.json",
        "bad_cases_sha256": sha(bad_content), "rows": len(records) * len(summary["protocol"]["methods"])}
    files = {prefix + "_per_question.csv": csv_content, prefix + "_bad_cases.json": bad_content,
             prefix + "_summary.json": (json.dumps(public, ensure_ascii=False, indent=2) + "\n").encode("utf-8")}
    if any((destination / name).exists() for name in files):
        raise FileExistsError("Public outputs already exist; existing results are never overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        with (destination / name).open("xb") as stream:
            stream.write(content)
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--prefix", default="qasper_test")
    args = parser.parse_args()
    report = export_run(args.run_directory, args.output_dir, args.prefix)
    print(json.dumps({"questions": report["question_count"], "public_artifacts": report["public_artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
