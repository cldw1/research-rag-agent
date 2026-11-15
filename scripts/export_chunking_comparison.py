"""Verify a completed chunking run and export numeric-only, immutable results."""

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.chunking_comparison import STRATEGIES, summarize_comparison


def export_comparison(run, destination):
    report = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    filenames = {"per_question_sha256": "per_question.jsonl", "protocol_sha256": "protocol.json",
                 "per_paper_sha256": "per_paper.json", "excluded_sha256": "excluded.json"}
    for key, name in filenames.items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != report["artifacts"][key]:
            raise ValueError("Run checksum mismatch: " + name)
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    if protocol != report["protocol"] or protocol["split"] != "validation":
        raise ValueError("Expected unchanged Validation protocol")
    if any(not report[key] for key in ("adaptive_baseline_reproduced", "all_source_characters_preserved", "production_default_unchanged")):
        raise ValueError("Comparison did not preserve baseline or sources")
    papers = json.loads((run / "per_paper.json").read_text(encoding="utf-8"))
    if (len(papers) != report["corpus"]["total_papers"]
            or len({p["paper_id"] for p in papers}) != len(papers)):
        raise ValueError("Incomplete indexed paper population")
    for strategy in STRATEGIES:
        config = protocol["configs"][strategy]
        builds = [p["strategies"][strategy] for p in papers]
        for p in builds:
            count = p["chunk_count"]
            if (len(p["chunk_tokens"]) != count or len(p["chunk_characters"]) != count
                    or any(not 0 < t <= config["max_tokens"] for t in p["chunk_tokens"])
                    or any(not 0 < t <= config["max_chars"] for t in p["chunk_characters"])
                    or p["vector_bytes"] != count * protocol["embedding_profile"]["dimension"] * 4):
                raise ValueError("Invalid chunk budget or vector size")
        for key in ("chunk_count", "vector_bytes", "merged_paragraph_chunks", "cache_hit"):
            if report["engineering"][strategy][key] != sum(p[key] for p in builds):
                raise ValueError("Engineering counts do not reproduce")
    records = [json.loads(line) for line in (run / "per_question.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if (len(records) != report["question_count"] or len(records) != report["corpus"]["eligible_questions"]
            or len({r["question_id"] for r in records}) != len(records)
            or len({r["paper_id"] for r in records}) != report["corpus"]["eligible_papers"]):
        raise ValueError("Incomplete question population")
    computed = summarize_comparison(records)
    if any(report[key] != value for key, value in computed.items()):
        raise ValueError("Comparison statistics or recommendation do not reproduce")
    metrics = list(records[0]["strategies"]["adaptive"]["methods"]["hybrid"]["5"])
    context_metrics = list(records[0]["strategies"]["adaptive"]["contexts"]["top5"])
    retrieval, contexts = io.StringIO(newline=""), io.StringIO(newline="")
    retrieval_writer = csv.DictWriter(retrieval, ["paper_id", "question_id", "strategy", "method", *metrics], lineterminator="\n")
    context_writer = csv.DictWriter(contexts, ["paper_id", "question_id", "strategy", "variant", *context_metrics], lineterminator="\n")
    retrieval_writer.writeheader()
    context_writer.writeheader()
    for record in records:
        for strategy in STRATEGIES:
            values = record["strategies"][strategy]
            identity = {"paper_id": record["paper_id"], "question_id": record["question_id"], "strategy": strategy}
            for method in ("bm25", "dense", "hybrid"):
                row = values["methods"][method]["5"]
                if any(not math.isfinite(v) or not 0 <= v <= 1 for v in row.values()):
                    raise ValueError("Invalid retrieval metric")
                retrieval_writer.writerow({**identity, "method": method, **row})
            for variant in ("top5", "anchor_first"):
                row = values["contexts"][variant]
                if (row["characters"] > protocol["evidence_char_budget"]
                        or any(not math.isfinite(v) or v < 0 for v in row.values())):
                    raise ValueError("Invalid context metric or budget")
                context_writer.writerow({**identity, "variant": variant, **row})
    files = {"qasper_chunking_validation_per_question.csv": retrieval.getvalue().encode("utf-8"),
             "qasper_chunking_validation_contexts.csv": contexts.getvalue().encode("utf-8")}
    public = {**report, "public_artifacts": {}}
    for name, raw in files.items():
        public["public_artifacts"][name] = {"sha256": hashlib.sha256(raw).hexdigest(),
            "rows": len(records) * (6 if "per_question" in name else 4)}
    files["qasper_chunking_validation.json"] = (json.dumps(public, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if any((destination / name).exists() for name in files):
        raise FileExistsError("Existing public results are never overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    for name, raw in files.items():
        with (destination / name).open("xb") as stream:
            stream.write(raw)
    return {"questions": len(records), "recommended_for_followup": report["recommended_for_followup"],
            "files": list(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    print(json.dumps(export_comparison(args.run, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
