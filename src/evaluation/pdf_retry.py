"""Replay a bounded retry policy over frozen development answers and sources."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from src.evaluation.answer_review import review_pack, summarize_reviews
from src.evaluation.pdf_answers import collect_results, load_inputs, read_json, summarize_answers, write_new
from src.evaluation.pdf_benchmark import evidence_coverage, sha256


def content_hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def baseline_inputs(baseline, retrieval):
    baseline, retrieval = Path(baseline), Path(retrieval)
    dataset, requests = load_inputs(retrieval)
    manifest = read_json(baseline / "manifest.json")
    if read_json(baseline / "requests.json") != requests or content_hash(requests) != manifest["requests_sha256"]:
        raise ValueError("Baseline generation inputs differ from frozen retrieval")
    if (baseline / "dataset.json").read_bytes() != (retrieval / "dataset.json").read_bytes():
        raise ValueError("Baseline reference data changed")
    results = collect_results(baseline, requests)
    if len(results) != 45 or any(r["status"] != "succeeded" for r in results):
        raise ValueError("Expected all 45 successful baseline answers")
    return dataset, requests, results, manifest


def verify_retry_run(run, baseline, retrieval, root):
    run, baseline, retrieval, root = map(Path, (run, baseline, retrieval, root))
    manifest = read_json(run / "manifest.json")
    baseline_inputs(baseline, retrieval)
    for name, expected in manifest["inputs_sha256"].items():
        path = (root / name).resolve()
        if root.resolve() not in path.parents or sha256(path.read_bytes()) != expected:
            raise ValueError("Frozen retry input changed")
    for name, expected in manifest["source_code_sha256"].items():
        if sha256((run / "source_snapshot" / name).read_bytes()) != expected:
            raise ValueError("Retry source snapshot changed")
    for name, expected in manifest["artifacts_sha256"].items():
        if Path(name).name != name or sha256((run / name).read_bytes()) != expected:
            raise ValueError("Retry artifact changed")
    requests = read_json(run / "requests.json")
    if len({r["question_id"] for r in requests}) != len(requests):
        raise ValueError("Duplicate retry request")
    # Rebuild the prompt from canonical whole chunks. No prior answer or reference
    # answer may be injected into the follow-up generation input.
    from src.agent.context_window import ByteTokenEstimator
    from src.generation.context_builder import build_qa_prompt
    from src.generation.strict_context_builder import build_strict_context
    dataset = read_json(retrieval / "dataset.json")
    questions = {q["question_id"]: q for q in dataset["questions"]}
    chunks = read_json(retrieval / "chunks.json")
    for request in requests:
        question = questions[request["question_id"]]
        lookup = {c["chunk_id"]: c for c in chunks[question["paper_id"]]}
        selected = [lookup[s["chunk_id"]] for s in request["sources"]]
        context, sources = build_strict_context(selected, max_chars=manifest["policy"]["max_chars"])
        prompt = build_qa_prompt(question["question"], context)
        messages = [{"role": "user", "content": prompt}]
        if (sources != request["sources"] or request["paper_id"] != question["paper_id"]
                or messages != request["messages"] or sha256(prompt.encode()) != request["prompt_sha256"]
                or request["input_estimate"] != ByteTokenEstimator().count(messages)):
            raise ValueError("Retry prompt/source scope changed")
    return manifest, requests


def effective_review_pack(run, baseline, retrieval, sources):
    run, baseline, retrieval = map(Path, (run, baseline, retrieval))
    pack = review_pack(baseline, retrieval, sources)
    requests = read_json(run / "requests.json")
    results = {r["question_id"]: r for r in collect_results(run, requests)}
    req_by_id = {r["question_id"]: r for r in requests}
    chunks = read_json(retrieval / "chunks.json")
    aligns = read_json(retrieval / "alignments.json")
    effective, selections = [], []
    for original in pack:
        row = deepcopy(original)
        qid = row["question_id"]
        result = results.get(qid)
        accept = bool(result and result["status"] == "succeeded" and not result["invalid_citations"])
        if accept:
            req = req_by_id[qid]
            lookup = {c["chunk_id"]: c for c in chunks[row["paper_id"]]}
            selected = [lookup[s["chunk_id"]] for s in req["sources"]]
            row.update(answer=result["answer"], answer_sha256=sha256(result["answer"].encode()),
                status=result["status"], cited_numbers=result["cited_numbers"], invalid_citations=result["invalid_citations"],
                provided_sources=[{"citation": s["citation"], "page_number": s.get("page_number"),
                    "chunk_id": s["chunk_id"], "text": lookup[s["chunk_id"]]["text"]} for s in req["sources"]],
                context_coverage=evidence_coverage(selected, [a for a in aligns if a["question_id"] == qid]) if row["answerable"] else None)
        selections.append({"question_id": qid, "selected": "retry" if accept else "baseline",
            "retry_status": result["status"] if result else "not_requested", "answer_sha256": row["answer_sha256"]})
        effective.append(row)
    return effective, selections


def retry_report(run, baseline, retrieval, sources, root, *, reviews_file=None):
    run, baseline, retrieval = map(Path, (run, baseline, retrieval))
    manifest, requests = verify_retry_run(run, baseline, retrieval, root)
    _, base_requests, base_results, base_manifest = baseline_inputs(baseline, retrieval)
    results = collect_results(run, requests)
    incremental = summarize_answers(results, manifest)
    base_summary = summarize_answers(base_results, base_manifest)
    pack, selections = effective_review_pack(run, baseline, retrieval, sources)
    baseline_pack = review_pack(baseline, retrieval, sources)
    rubric = read_json(Path(root) / "benchmarks/pdf_qa/answer_review_rubric.json")
    base_reviews = read_json(baseline / "ai_reviews.json")
    base_review_summary = summarize_reviews(baseline_pack, base_reviews, rubric)
    reviews = read_json(reviews_file) if reviews_file else None
    review_summary = summarize_reviews(pack, reviews, rubric) if reviews else None
    if review_summary and not review_summary["complete"]:
        raise ValueError("Incomplete effective-answer review")
    comparable = [r for r in pack if r["answerable"]]
    def averages(rows):
        return {k: sum(r["context_coverage"][k] for r in rows)/len(rows) for k in rows[0]["context_coverage"]}
    decisions = read_json(run / "decisions.json")
    report = {"schema_version": "pdf-evidence-retry-report-v1", "run_id": run.name,
        "scope": "Exploratory 45-question development policy comparison; saved control answers reused; independent human review pending.",
        "question_count": len(pack), "baseline_answer_run": baseline.name, "retrieval_run": retrieval.name,
        "policy": manifest["policy"], "model_config": manifest["model_config"],
        "gate_reason_counts": dict(Counter(d["reason"] for d in decisions)),
        "retried_questions": [r["question_id"] for r in requests],
        "retrieval_calls": sum(d["retrieval_calls"] for d in decisions),
        "retrieval_seconds": sum(d.get("seconds", 0.) for d in decisions),
        "incremental_generation": incremental, "baseline_generation": base_summary,
        "total_known_tokens_with_baseline": base_summary["known_returned_usage"]["total_tokens"] + incremental["known_returned_usage"]["total_tokens"],
        "final_answer_selection_counts": dict(Counter(r["selected"] for r in selections)),
        "baseline_context_metrics": averages([r for r in baseline_pack if r["answerable"]]),
        "policy_context_metrics": averages(comparable),
        "baseline_ai_review": base_review_summary, "policy_ai_review": review_summary,
        "independent_human_metrics": None,
        "selections": selections,
        "decisions": [{k: d[k] for k in ("question_id", "reason", "retry", "retrieval_calls", "new_chunk_ids", "dropped_chunk_ids")} for d in decisions],
        "manifest_sha256": sha256((run / "manifest.json").read_bytes()),
        "attempt_result_sha256": {r["question_id"]: sha256((run / "attempts" / r["question_id"] / "result.json").read_bytes()) for r in results if (run / "attempts" / r["question_id"] / "result.json").exists()},
        "quality_note": "Answer-level 0/1/2 AI source-audit ratings are provisional, not a human-verified accuracy or a blind evaluation."}
    if reviews:
        before = {r["question_id"]: r for r in base_reviews["records"]}
        deltas = [r["correctness"]-before[r["question_id"]]["correctness"] for r in reviews["records"]]
        report.update(correctness_changes={"improved": sum(d>0 for d in deltas), "unchanged": sum(d==0 for d in deltas), "worsened": sum(d<0 for d in deltas)},
            review_records=reviews["records"], reviews_sha256=sha256(Path(reviews_file).read_bytes()), rubric_sha256=sha256((Path(root) / "benchmarks/pdf_qa/answer_review_rubric.json").read_bytes()))
    return report, pack
