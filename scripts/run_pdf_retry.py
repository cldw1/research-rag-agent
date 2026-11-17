"""Prepare or execute one optional evidence retry over the fixed PDF development set."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.context_window import ByteTokenEstimator
from src.evaluation.pdf_answers import attempt_paths, generate_one, read_json, write_new
from src.evaluation.pdf_benchmark import sha256, verify_candidate_release
from src.evaluation.pdf_retry import baseline_inputs, retry_report, verify_retry_run
from src.llm.provider import ModelConfig, OpenAICompatibleProvider
from src.retrieval.evidence_retry import EvidenceRetryPolicy, prepare_evidence_retry


SOURCE_FILES = ["scripts/run_pdf_retry.py", "src/evaluation/pdf_retry.py", "src/retrieval/evidence_retry.py",
    "src/evaluation/pdf_answers.py", "src/evaluation/answer_review.py", "src/llm/provider.py",
    "src/generation/context_builder.py", "src/generation/strict_context_builder.py",
    "src/retrieval/structured_retriever.py", "src/embedding.py"]


def prepare(args):
    verify_candidate_release(ROOT / "benchmarks/pdf_qa")
    dataset, base_requests, base_results, base_manifest = baseline_inputs(args.baseline, args.retrieval_run)
    if (args.retrieval_run / "dataset.json").read_bytes() != (ROOT / "benchmarks/pdf_qa/development.json").read_bytes():
        raise ValueError("Only the unchanged development set is supported")
    protocol_path = args.protocol
    protocol = read_json(protocol_path)
    if args.baseline.name != protocol["baseline_answer_run"] or args.retrieval_run.name != protocol["baseline_retrieval_run"]:
        raise ValueError("Baseline differs from frozen retry protocol")
    policy = EvidenceRetryPolicy(**protocol["policy"])
    run = args.output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    # Experiment copy: database/index access cannot mutate the saved control run.
    shutil.copytree(args.retrieval_run / "catalog", run / "catalog")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from src.embedding import load_embedding_resources
    from src.storage.document_store import LocalDocumentStore
    torch.set_num_threads(8)
    model, _, profile = load_embedding_resources(local_files_only=True)
    if profile.to_dict() != read_json(args.retrieval_run / "summary.json")["embedding_profile"]:
        raise ValueError("Embedding model differs from baseline")
    store = LocalDocumentStore(run / "catalog")
    by_id = {r["question_id"]: r for r in base_results}
    retrievers, requests, decisions = {}, [], []
    try:
        for original in base_requests:
            qid, pid = original["question_id"], original["paper_id"]
            if pid not in retrievers:
                retrievers[pid] = store.load_retriever(original["sources"][0]["paper_id"], model, profile)
            retriever = retrievers[pid]
            question = next(q["question"] for q in dataset["questions"] if q["question_id"] == qid)
            before = time.perf_counter()
            decision = prepare_evidence_retry(retriever, question, retriever.paper_id,
                by_id[qid]["answer"], original["sources"], policy=policy)
            decision.update(question_id=qid, seconds=time.perf_counter()-before)
            if decision["retry"]:
                prompt = decision["prompt"]
                messages = [{"role": "user", "content": prompt}]
                requests.append({"question_id": qid, "paper_id": pid, "variant": "evidence_retry",
                    "sources": decision["sources"], "messages": messages,
                    "input_estimate": ByteTokenEstimator().count(messages), "prompt_sha256": sha256(prompt.encode())})
            decisions.append(decision)
    finally:
        store.close()
    write_new(run / "requests.json", requests)
    write_new(run / "decisions.json", decisions)
    inputs = [protocol_path, ROOT / "benchmarks/pdf_qa/answer_review_rubric.json",
        ROOT / "benchmarks/pdf_qa/development.json", ROOT / "benchmarks/pdf_qa/candidate_manifest.json",
        args.baseline / "manifest.json", args.baseline / "ai_reviews.json", args.baseline / "requests.json",
        args.baseline / "dataset.json", args.retrieval_run / "summary.json"]
    inputs += [args.baseline / "attempts" / r["question_id"] / "result.json" for r in base_results]
    inputs += [args.retrieval_run / name for name in read_json(args.retrieval_run / "summary.json")["artifacts_sha256"]]
    hashes = {}
    for name in SOURCE_FILES:
        content = (ROOT / name).read_bytes()
        hashes[name] = sha256(content)
        snapshot = run / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
    manifest = {"protocol_version": protocol["protocol_version"], "policy": asdict(policy),
        "model_config": base_manifest["model_config"], "thinking": base_manifest["thinking"],
        "transport": {"environment_proxy": args.use_environment_proxy, "follow_redirects": False},
        "price_reference": base_manifest["price_reference"], "baseline_answer_run": args.baseline.name,
        "retrieval_run": args.retrieval_run.name, "source_code_sha256": hashes,
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()) for p in inputs},
        "artifacts_sha256": {name: sha256((run / name).read_bytes()) for name in ("requests.json", "decisions.json")}}
    write_new(run / "manifest.json", manifest)
    print({"run": str(run), "planned_retries": len(requests)}, flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "data/pdf_benchmark/answer_runs/20261115T171727Z-a10a57a1")
    parser.add_argument("--retrieval-run", type=Path, default=ROOT / "data/pdf_benchmark/runs/20261115T165227Z-1bd0349b")
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks/pdf_qa/retry_protocol.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/pdf_benchmark/retry_runs")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-calls", type=int, default=45)
    parser.add_argument("--use-environment-proxy", action="store_true")
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args()
    for name in ("baseline", "retrieval_run", "output_root", "protocol", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if not 0 <= args.max_calls <= 45:
        parser.error("max-calls must be between 0 and 45")
    run = args.resume or prepare(args)
    manifest, requests = verify_retry_run(run, args.baseline, args.retrieval_run, ROOT)
    if args.execute:
        if manifest["transport"]["environment_proxy"] != args.use_environment_proxy:
            parser.error("Proxy configuration differs from frozen run")
        for name, expected in manifest["source_code_sha256"].items():
            if sha256((ROOT / name).read_bytes()) != expected:
                parser.error("Execution code changed; create a new run")
        pending = [r for r in requests if not attempt_paths(run, r["question_id"])[0].exists()]
        pending = pending[:max(0, args.max_calls-(len(requests)-len(pending)))]
        if pending:
            from dotenv import load_dotenv
            from openai import DefaultHttpxClient, OpenAI
            load_dotenv(ROOT / ".env", override=False)
            key = os.getenv("DEEPSEEK_API_KEY", "").strip()
            if not key or any(s in key.lower() for s in ("your_", "dummy", "placeholder")):
                parser.error("Configure DEEPSEEK_API_KEY in the existing environment or .env")
            config = ModelConfig(**manifest["model_config"])
            client = OpenAI(api_key=key, base_url=config.base_url, timeout=60., max_retries=0,
                http_client=DefaultHttpxClient(trust_env=args.use_environment_proxy, follow_redirects=False))
            provider = OpenAICompatibleProvider(config, key, client=client)
            try:
                probe = generate_one(run, pending.pop(0), provider)
                print({"question_id": probe["question_id"], "status": probe["status"]}, flush=True)
                if probe["status"] != "failed":
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        for future in as_completed([pool.submit(generate_one, run, r, provider) for r in pending]):
                            result = future.result()
                            print({"question_id": result["question_id"], "status": result["status"]}, flush=True)
            finally:
                client.close()
    report, pack = retry_report(run, args.baseline, args.retrieval_run, ROOT / "data/pdf_benchmark/sources", ROOT,
        reviews_file=args.reviews)
    write_new(run / ("summary-"+uuid.uuid4().hex[:8]+".json"), report)
    write_new(run / ("review_pack-"+uuid.uuid4().hex[:8]+".json"), pack)
    if args.public_output:
        if report["incremental_generation"]["status_counts"].get("pending", 0) or report["policy_ai_review"] is None:
            parser.error("Public export requires completed attempts and full effective-answer review")
        write_new(args.public_output, report)
    print({"run": str(run), "incremental_status": report["incremental_generation"]["status_counts"],
           "selected": report["final_answer_selection_counts"]}, flush=True)


if __name__ == "__main__":
    main()
