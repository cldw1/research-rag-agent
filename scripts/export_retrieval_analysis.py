"""Verify failure cohorts and Validation selection before publishing numeric artifacts."""

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_qasper_failures import failure_groups
from scripts.compare_rank_fusion import summarize_records


def export_analysis(validation, audit_file, destination):
    report = json.loads((validation / "summary.json").read_text(encoding="utf-8"))
    raw = (validation / "per_question.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != report["records_sha256"]:
        raise ValueError("Validation record checksum mismatch")
    if json.loads((validation / "protocol.json").read_text(encoding="utf-8")) != report["protocol"]:
        raise ValueError("Validation configuration changed")
    if report["protocol"]["split"] != "validation":
        raise ValueError("Policy selection must use Validation")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(records) != report["question_count"] or len({r["question_id"] for r in records}) != len(records):
        raise ValueError("Incomplete Validation records")
    results, selected = summarize_records(records, report["protocol"]["policies"])
    if results != report["results"] or selected != report["selected_policy"]:
        raise ValueError("Validation metrics or selection do not reproduce")
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "results/qasper_test_summary.json").read_text(encoding="utf-8"))
    if audit["baseline_records_sha256"] != baseline["artifacts"]["per_question_jsonl_sha256"]:
        raise ValueError("Audit belongs to another baseline")
    grouped = defaultdict(dict)
    with (ROOT / "results/qasper_test_per_question.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["question_id"]][row["method"]] = {"5": {"hit": float(row["hit"])}}
    expected = failure_groups([{"question_id": qid, "methods": methods} for qid, methods in grouped.items()])
    actual = {row["question_id"]: row["category"] for row in audit["records"]}
    if actual != {qid: value[0] for qid, value in expected.items()}:
        raise ValueError("Audit does not include the full baseline failure population")
    if not audit["original_rankings_reproduced"] or not audit["source_chunks_completely_cover_all_gold"]:
        raise ValueError("Audit did not reproduce rankings and complete sources")
    metric_names = list(next(iter(records[0]["methods"].values())))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, ["paper_id", "question_id", "policy", *metric_names], lineterminator="\n")
    writer.writeheader()
    for record in records:
        for policy, values in record["methods"].items():
            writer.writerow({"paper_id": record["paper_id"], "question_id": record["question_id"],
                             "policy": policy, **{key: values[key] for key in metric_names}})
    csv_bytes = output.getvalue().encode("utf-8")
    public = {key: report[key] for key in (
        "protocol", "corpus", "question_count", "results", "selected_policy", "elapsed_seconds",
        "cache_hits", "records_sha256", "scope")}
    public["public_artifacts"] = {"per_question": "qasper_fusion_validation_per_question.csv",
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "rows": sum(len(r["methods"]) for r in records)}
    files = {"qasper_failure_audit.json": (json.dumps(audit, ensure_ascii=False, indent=2) + "\n").encode(),
        "qasper_fusion_validation.json": (json.dumps(public, ensure_ascii=False, indent=2) + "\n").encode(),
        "qasper_fusion_validation_per_question.csv": csv_bytes}
    if any((destination / name).exists() for name in files):
        raise FileExistsError("Existing public results are never overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        with (destination / name).open("xb") as stream:
            stream.write(content)
    return {"diagnosed_questions": len(actual), "validation_questions": len(records), "selected_policy": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_run", type=Path)
    parser.add_argument("audit_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    print(json.dumps(export_analysis(args.validation_run, args.audit_file, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
