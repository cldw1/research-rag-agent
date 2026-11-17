"""Source-audit material and validation; no model judge or invented human scores."""

import json
from pathlib import Path

from src.evaluation.pdf_answers import collect_results, read_json
from src.evaluation.pdf_benchmark import load_dataset, sha256, verify_sources


def review_pack(answer_run, retrieval_run, sources_root):
    answer_run, retrieval_run = Path(answer_run), Path(retrieval_run)
    manifest = read_json(answer_run / "manifest.json")
    requests = read_json(answer_run / "requests.json")
    if sha256(json.dumps(requests, sort_keys=True, ensure_ascii=False).encode()) != manifest["requests_sha256"]:
        raise ValueError("Generation inputs changed")
    dataset = load_dataset(answer_run / "dataset.json")
    if any(p["split"] != "dev" for p in dataset["papers"]):
        raise ValueError("Only development answers can be reviewed here")
    if sha256((answer_run / "dataset.json").read_bytes()) != manifest["dataset_sha256"]:
        raise ValueError("Answer reference dataset changed")
    if sha256((retrieval_run / "dataset.json").read_bytes()) != manifest["dataset_sha256"]:
        raise ValueError("Retrieval and generation datasets differ")
    native, _ = verify_sources(dataset, sources_root)
    chunks = read_json(retrieval_run / "chunks.json")
    record_map = {r["question_id"]: r for r in read_json(retrieval_run / "records.json")}
    results = {r["question_id"]: r for r in collect_results(answer_run, requests)}
    request_map = {r["question_id"]: r for r in requests}
    output = []
    for question in dataset["questions"]:
        qid, paper = question["question_id"], question["paper_id"]
        result, request = results[qid], request_map[qid]
        if result["status"] not in {"succeeded", "truncated", "empty"}:
            output.append({"question_id": qid, "paper_id": paper, "status": result["status"]})
            continue
        lookup = {c["chunk_id"]: c for c in chunks[paper]}
        provided = [{"citation": s["citation"], "page_number": s.get("page_number"),
                     "chunk_id": s["chunk_id"], "text": lookup[s["chunk_id"]]["text"]} for s in request["sources"]]
        gold = [{"evidence_id": a["evidence_id"], "page_number": a["page_number"],
                 "text": native[paper][(a["page_number"], a["block_index"])]["text"][a["char_start"]:a["char_end"]]}
                for a in question["evidence"]]
        output.append({"question_id": qid, "paper_id": paper, "question": question["question"],
            "answerable": question["answerable"], "category": question["category"],
            "reference_points": question["reference_points"], "gold_evidence": gold,
            "answer": result["answer"], "answer_sha256": sha256(result["answer"].encode()),
            "status": result["status"], "cited_numbers": result["cited_numbers"],
            "invalid_citations": result["invalid_citations"], "provided_sources": provided,
            "context_coverage": record_map[qid]["contexts"][request["variant"]]["metrics"]})
    return output


def validate_reviews(pack, reviews, rubric):
    if reviews["rubric_version"] != rubric["rubric_version"]:
        raise ValueError("Review rubric changed")
    if reviews["reviewer_type"] not in {"ai_source_audit", "human"}:
        raise ValueError("Reviewer provenance required")
    if not reviews.get("reviewer") or not reviews.get("reviewed_at"):
        raise ValueError("Reviewer identity and timestamp required")
    if reviews["reviewer_type"] == "ai_source_audit" and reviews.get("independent_human_review"):
        raise ValueError("AI source audit cannot be called independent human review")
    by_id = {r["question_id"]: r for r in pack if "answer_sha256" in r}
    seen = set()
    for review in reviews["records"]:
        qid = review["question_id"]
        if qid in seen or qid not in by_id:
            raise ValueError("Duplicate/unknown reviewed answer")
        seen.add(qid)
        if review["answer_sha256"] != by_id[qid]["answer_sha256"]:
            raise ValueError("Reviewed answer was changed")
        for metric in ("correctness", "faithfulness", "citation_support", "citation_coverage"):
            allowed = {0, 1, 2} if metric in {"correctness", "faithfulness"} else {0, 1, 2, None}
            if review[metric] not in allowed or (review[metric] is not None and type(review[metric]) is not int):
                raise ValueError("Invalid review score")
        if set(review.get("checked_citations", [])) != set(by_id[qid]["cited_numbers"]):
            raise ValueError("Citation audit population changed")
        if not review["reason"].strip() or not set(review["failure_categories"]) <= set(rubric["failure_categories"]):
            raise ValueError("Review rationale/category required")
    return {"reviewed_count": len(seen), "complete": seen == set(by_id),
            "reviewer_type": reviews["reviewer_type"],
            "independent_human_review": reviews["reviewer_type"] == "human" and reviews.get("independent_human_review") is True}


def summarize_reviews(pack, reviews, rubric):
    check = validate_reviews(pack, reviews, rubric)
    rows = reviews["records"]
    metrics = {}
    for key in ("correctness", "faithfulness", "citation_support", "citation_coverage"):
        applicable = [r[key] for r in rows if r[key] is not None]
        metrics[key] = {"applicable": len(applicable), "score_0": applicable.count(0),
                        "score_1": applicable.count(1), "score_2": applicable.count(2),
                        "mean_0_2": sum(applicable)/len(applicable) if applicable else None}
    negatives = {r["question_id"] for r in pack if r.get("answerable") is False}
    refusal = [r for r in rows if r["question_id"] in negatives]
    return {**check, "metrics": metrics, "unanswerable_reviewed": len(refusal),
            "unanswerable_fully_correct": sum(r["correctness"] == 2 for r in refusal),
            "human_metrics": metrics if check["independent_human_review"] else None,
            "note": "AI source-audit scores, when present, are provisional judgments against AI-authored labels; not a human-verified answer accuracy."}
