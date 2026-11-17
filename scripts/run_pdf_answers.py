"""Generate the frozen PDF development answers with the existing DeepSeek adapter."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.pdf_answers import (
    attempt_paths, collect_results, generate_one, load_inputs, read_json, summarize_answers, write_new,
)
from src.evaluation.pdf_benchmark import sha256, verify_candidate_release
from src.llm.provider import ModelConfig, OpenAICompatibleProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, default=ROOT / "data/pdf_benchmark/runs/20261115T165227Z-1bd0349b")
    parser.add_argument("--variant", choices=["anchor_first", "top5"], default="anchor_first")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/pdf_benchmark/answer_runs")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--execute", action="store_true", help="Make real billable generation requests; otherwise prepare inputs only")
    parser.add_argument("--max-calls", type=int, default=45)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--use-environment-proxy", action="store_true", help="Explicitly use the local HTTP(S) proxy environment for this benchmark")
    args = parser.parse_args()
    if not 0 <= args.max_calls <= 45:
        parser.error("max-calls must be between 0 and 45")
    verify_candidate_release(ROOT / "benchmarks/pdf_qa")
    if sha256((args.retrieval_run / "dataset.json").read_bytes()) != sha256((ROOT / "benchmarks/pdf_qa/development.json").read_bytes()):
        parser.error("Expected the unchanged frozen development view")
    dataset, requests = load_inputs(args.retrieval_run, args.variant)
    config = ModelConfig(provider="deepseek", model="deepseek-flash", temperature=0., max_output_tokens=800,
                         supports_tools=False, stream=False)
    manifest = {"protocol_version": "pdf-answers-v1", "retrieval_run_id": args.retrieval_run.name,
        "dataset_sha256": sha256((args.retrieval_run / "dataset.json").read_bytes()),
        "requests_sha256": sha256(json.dumps(requests, sort_keys=True, ensure_ascii=False).encode()),
        "question_ids": [r["question_id"] for r in requests], "variant": args.variant,
        "model_config": config.to_dict(), "thinking": "disabled", "sdk_max_retries": 0,
        "transport": {"environment_proxy": args.use_environment_proxy, "follow_redirects": False},
        "price_reference": {"url": "https://api-docs.deepseek.com/quick_start/pricing/", "checked_date": "2026-11-16",
                            "peak_usd_per_million": {"input_cache_miss": .3, "output": 1.2}},
        "source_code_sha256": {name: sha256((ROOT / name).read_bytes()) for name in
            ("scripts/run_pdf_answers.py", "src/evaluation/pdf_answers.py", "src/llm/provider.py", "src/generation/context_builder.py")}}
    if args.resume:
        run = args.resume
        if read_json(run / "manifest.json") != manifest:
            parser.error("Resume inputs/config/code differ; refusing to mix runs")
    else:
        run = args.output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
        run.mkdir(parents=True, exist_ok=False)
        write_new(run / "manifest.json", manifest)
        write_new(run / "requests.json", requests)
        write_new(run / "dataset.json", dataset)
    pending = [r for r in requests if not attempt_paths(run, r["question_id"])[0].exists()]
    existing = len(requests)-len(pending)
    pending = pending[:max(0, args.max_calls-existing)]
    if args.execute and pending:
        from dotenv import load_dotenv
        from openai import DefaultHttpxClient, OpenAI
        load_dotenv(ROOT / ".env", override=False)
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key or any(part in key.lower() for part in ("your_", "placeholder", "dummy")):
            parser.error("Configure DEEPSEEK_API_KEY in the environment or the existing local .env; never paste it into output files")
        client = OpenAI(api_key=key, base_url=config.base_url, timeout=60., max_retries=0,
                        http_client=DefaultHttpxClient(follow_redirects=False, trust_env=args.use_environment_proxy))
        provider = OpenAICompatibleProvider(config, key, client=client)
        print(json.dumps({"run": str(run), "pending": len(pending), "model": config.model}), flush=True)
        try:
            # Validate one call before starting the remaining batch.
            if pending and existing == 0:
                result = generate_one(run, pending.pop(0), provider)
                print(json.dumps({"question_id": result["question_id"], "status": result["status"]}), flush=True)
                if result["status"] == "failed":
                    pending = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(generate_one, run, request, provider) for request in pending]
                for future in as_completed(futures):
                    result = future.result()
                    print(json.dumps({"question_id": result["question_id"], "status": result["status"]}), flush=True)
        finally:
            client.close()
    results = collect_results(run, requests)
    summary = summarize_answers(results, manifest)
    # Preserve previous summaries when resuming; raw attempt records are immutable.
    write_new(run / ("summary-" + uuid.uuid4().hex[:8] + ".json"), summary)
    print(json.dumps({"run": str(run), "status_counts": summary["status_counts"],
                      "generation_api_attempts": summary["generation_api_attempts"]}), flush=True)


if __name__ == "__main__":
    main()
