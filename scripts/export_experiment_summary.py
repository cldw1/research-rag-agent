"""Export historical HMC results without passage text or conversation records.

No model loading or API calls. The original local reports remain unchanged.
"""

import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "results" / "hmc_v3_public_summary.json"
SECTION_NUMBER_PATTERN = re.compile(r"^(\d+(?:\.\d+)+)(?![\d.])")


def read_report(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def summarize(retrieval, audit):
    """Recompute metrics and refuse inconsistent historical reports."""
    records = retrieval["records"]
    annotations = {r["id"]: r for r in audit["annotations"]}
    if (not records or len(records) != retrieval["question_count"]
            or len(records) != audit["question_count"]
            or len({r["id"] for r in records}) != len(records)
            or {r["id"] for r in records} != set(annotations)
            or len(annotations) != len(audit["annotations"])
            or retrieval["pdf_hash"] != audit["pdf_hash"]
            or retrieval["top_k"] != 5):
        raise ValueError("Historical reports disagree on questions, PDF or Top-k")

    public_records = []
    for record in records:
        annotation = annotations[record["id"]]
        if annotation["gold_section"] != record["gold_section"]:
            raise ValueError("Gold section mismatch")
        public = {k: record[k] for k in ("id", "question", "gold_section", "gold_section_title")}
        for source, target in (("baseline", "hybrid"), ("constrained", "section_aware")):
            top5 = record[source]["top5"]
            if len(top5) > 5 or [c["rank"] for c in top5] != list(range(1, len(top5) + 1)):
                raise ValueError("Invalid ranking")
            # Match the historical evaluator's exact numbered-section rule.
            rank = None
            for candidate in top5:
                match = SECTION_NUMBER_PATTERN.match(candidate["section"].strip())
                if match and match.group(1) == record["gold_section"]:
                    rank = candidate["rank"]
                    break
            if rank != record[source]["gold_rank"] or bool(rank) != record[source]["hit_at_5"]:
                raise ValueError("Stored gold rank disagrees with ranked sections")
            score = annotation["ratings"][source]["score"]
            if score not in (0, 1, 2):
                raise ValueError("Invalid manual score")
            public[target] = {
                "gold_rank": rank,
                "evidence_score": score,
                "top5": [{k: c[k] for k in ("rank", "chunk_id", "page", "section", "rrf_score", "constrained_score")} for c in top5],
            }
        public_records.append(public)

    metrics = {}
    for source, target in (("baseline", "hybrid"), ("constrained", "section_aware")):
        ranks = [r[target]["gold_rank"] for r in public_records]
        scores = [r[target]["evidence_score"] for r in public_records]
        n = len(ranks)
        computed = {
            "hit_count": sum(r is not None for r in ranks),
            "hit_at_5": sum(r is not None for r in ranks) / n,
            "mrr_at_5": sum(1 / r if r else 0 for r in ranks) / n,
            "sufficient_count": scores.count(2),
            "sufficient_rate": scores.count(2) / n,
            "average_evidence_score": sum(scores) / n,
        }
        for key in ("hit_count", "hit_at_5", "mrr_at_5"):
            if not math.isclose(computed[key], retrieval[f"{source}_metrics"][key]):
                raise ValueError("Retrieval summary mismatch: " + key)
        for key in ("sufficient_count", "sufficient_rate", "average_score"):
            actual = computed["average_evidence_score" if key == "average_score" else key]
            if not math.isclose(actual, audit[f"{source}_summary"][key]):
                raise ValueError("Manual audit summary mismatch: " + key)
        metrics[target] = computed

    return {
        "schema_version": 1,
        "experiment": retrieval["experiment"],
        "filename": retrieval["filename"],
        "pdf_sha256": retrieval["pdf_hash"],
        "chunk_count": retrieval["chunk_count"],
        "question_count": len(records),
        "evaluation_level": "section",
        "top_k": 5,
        "annotation_status": retrieval["annotation_status"],
        "annotation_note": audit["annotation_note"],
        "candidate_pool_note": retrieval["candidate_pool_note"],
        "parameters": {k: retrieval[k] for k in ("section_bonus", "reference_penalty", "max_per_section")},
        "metrics": metrics,
        "records": public_records,
        "scope": "Historical single-paper results; export recomputes metrics, not retrieval or answer generation.",
    }


def main():
    names = ("pdf_v3_holdout_evaluation.json", "pdf_v3_evidence_audit.json")
    try:
        result = summarize(*(read_report(name) for name in names))
    except FileNotFoundError as exc:
        raise SystemExit("需要本地原始报告；仓库已附公开摘要，此导出命令不负责重新运行检索。") from exc
    result["source_reports"] = [
        {"filename": name, "sha256": hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest()}
        for name in names
    ]
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Historical metrics verified; public summary written:", OUTPUT_PATH.name)


if __name__ == "__main__":
    main()
