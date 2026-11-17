"""Development-only generation inputs, durable attempts and honest answer diagnostics."""

import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.agent.context_window import ByteTokenEstimator
from src.evaluation.pdf_benchmark import load_dataset, sha256
from src.generation.context_builder import build_qa_prompt
from src.generation.strict_context_builder import build_strict_context
from src.llm.provider import ModelError


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_inputs(run, variant="anchor_first"):
    """Rebuild prompts from frozen retrieved chunks; gold never enters messages."""
    run = Path(run)
    summary = read_json(run / "summary.json")
    for name, expected in summary["artifacts_sha256"].items():
        if Path(name).name != name or sha256((run / name).read_bytes()) != expected:
            raise ValueError("Retrieval artifact changed")
    dataset = load_dataset(run / "dataset.json")
    if dataset["status"] != "development_pilot" or any(p["split"] != "dev" for p in dataset["papers"]):
        raise ValueError("Only development data may generate answers")
    if variant not in {"top5", "anchor_first"}:
        raise ValueError("Unsupported context variant")
    questions = {q["question_id"]: q for q in dataset["questions"]}
    chunks = read_json(run / "chunks.json")
    records = {r["question_id"]: r for r in read_json(run / "records.json")}
    protocol = read_json(run / "protocol.json")
    requests = [json.loads(line) for line in (run / "generation_requests.jsonl").read_text(encoding="utf-8").splitlines()]
    selected, seen = [], set()
    for request in requests:
        if request["variant"] != variant:
            continue
        qid = request["question_id"]
        if qid in seen or qid not in questions or not re.fullmatch(r"[a-zA-Z0-9_-]+", qid):
            raise ValueError("Duplicate/unknown/unsafe question ID")
        seen.add(qid)
        question, record = questions[qid], records[qid]
        if request["paper_id"] != question["paper_id"] or request["sources"] != record["contexts"][variant]["sources"]:
            raise ValueError("Question/source scope differs from retrieval run")
        lookup = {c["chunk_id"]: c for c in chunks[question["paper_id"]]}
        chosen = [lookup[s["chunk_id"]] for s in request["sources"]]
        if any(c["paper_id"] != s["paper_id"] for c, s in zip(chosen, request["sources"], strict=True)):
            raise ValueError("Cross-document citation")
        context, _ = build_strict_context(chosen, max_chars=protocol["contexts"]["max_chars"])
        prompt = build_qa_prompt(question["question"], context)
        if prompt != request["prompt"] or sha256(prompt.encode("utf-8")) != request["prompt_sha256"]:
            raise ValueError("Prompt differs from frozen retrieval context")
        messages = [{"role": "user", "content": prompt}]
        selected.append({"question_id": qid, "paper_id": question["paper_id"], "variant": variant,
                         "prompt_sha256": request["prompt_sha256"], "messages": messages,
                         "input_estimate": ByteTokenEstimator().count(messages), "sources": request["sources"]})
    if seen != set(questions) or len(selected) != summary["question_count"]:
        raise ValueError("Incomplete generation input population")
    return dataset, selected


def citation_diagnostics(answer, sources):
    # Only syntactic validity. Semantic support is a separate reviewed metric.
    numbers = sorted({int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer)
                      for n in re.findall(r"\d+", group)})
    allowed = {s["citation"] for s in sources}
    return {"cited_numbers": numbers, "invalid_citations": sorted(set(numbers)-allowed),
            "citation_format_valid": bool(numbers) and not (set(numbers)-allowed)}


def attempt_paths(run, qid):
    folder = Path(run) / "attempts" / qid
    return folder / "request.json", folder / "result.json"


def generate_one(run, request, provider):
    intent, output = attempt_paths(run, request["question_id"])
    if intent.exists():
        if output.exists():
            return read_json(output)
        return {"question_id": request["question_id"], "status": "uncertain", "api_attempted": True}
    config = provider.config
    budget = config.context_window-config.max_output_tokens-config.safety_margin
    if request["input_estimate"] > budget:
        raise ValueError("Generation prompt exceeds application input budget")
    before = {"question_id": request["question_id"], "paper_id": request["paper_id"],
              "prompt_sha256": request["prompt_sha256"], "started_at": datetime.now(timezone.utc).isoformat(),
              "input_estimate": request["input_estimate"], "input_budget": budget,
              "count_method": ByteTokenEstimator.method}
    # Durable intent is written BEFORE the request. An interrupted call is not replayed.
    write_new(intent, before)
    started = time.perf_counter()
    try:
        response = provider.complete(request["messages"])
        if response.tool_calls:
            raise ModelError("unexpected_tool_call", "Plain RAG benchmark received a tool call")
        answer = response.content.strip()
        status = "succeeded" if answer and response.finish_reason != "length" else "truncated" if answer else "empty"
        result = {**before, "status": status, "api_attempted": True, "answer": answer,
                  "usage": response.usage, "response_metadata": response.response_metadata,
                  "finish_reason": response.finish_reason,
                  **citation_diagnostics(answer, request["sources"])}
    except Exception as exc:
        result = {**before, "status": "failed", "api_attempted": True,
                  "error_code": exc.code if isinstance(exc, ModelError) else "generation_failed"}
    result["seconds"] = time.perf_counter()-started
    write_new(output, result)
    return result


def collect_results(run, requests):
    output = []
    for request in requests:
        intent, result = attempt_paths(run, request["question_id"])
        if result.exists():
            item = read_json(result)
            if item["prompt_sha256"] != request["prompt_sha256"]:
                raise ValueError("Saved answer belongs to a different prompt")
            output.append(item)
        else:
            output.append({"question_id": request["question_id"], "status": "uncertain" if intent.exists() else "pending",
                           "api_attempted": intent.exists()})
    return output


def summarize_answers(results, manifest):
    from src.evaluation.qasper_benchmark import percentile_summary
    counts = dict(Counter(r["status"] for r in results))
    returned = [r for r in results if r["status"] in {"succeeded", "truncated", "empty"}]
    completed = [r for r in results if r["status"] == "succeeded"]
    usage = {key: sum(r.get("usage", {}).get(key, 0) for r in returned)
             for key in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")}
    usage_complete = len(returned) > 0 and all(all(k in r.get("usage", {}) for k in ("prompt_tokens", "completion_tokens")) for r in returned)
    upper = (usage["prompt_tokens"]*.3+usage["completion_tokens"]*1.2)/1_000_000 if usage_complete else None
    return {"schema_version": "pdf-answer-run-v1", "scope": "Development-only plain RAG generation; semantic answer scores require a separate review.",
            "manifest_sha256": sha256(json.dumps(manifest, sort_keys=True).encode()),
            "question_count": len(results), "status_counts": counts,
            "generation_api_attempts": sum(r["api_attempted"] for r in results), "sdk_max_retries": 0,
            "known_returned_usage": usage, "usage_complete_for_returned_responses": usage_complete,
            "estimated_returned_cost_usd_upper_at_published_peak_rates": upper,
            "cost_note": "Estimate for returned responses only; cache discounts/off-peak may lower it. Failed/uncertain request usage is unknown. Not an account bill.",
            "response_seconds": percentile_summary([r["seconds"] for r in returned]) if returned else None,
            "answers_with_citations": sum(bool(r["cited_numbers"]) for r in completed),
            "answers_with_invalid_citation_numbers": sum(bool(r["invalid_citations"]) for r in completed),
            "answer_quality_metrics": None, "review_status": "pending"}
