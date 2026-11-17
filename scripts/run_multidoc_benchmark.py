"""Offline comparison over saved PDF development vectors; no generation calls."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time
import uuid
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.pdf_answers import load_inputs, read_json, write_new
from src.evaluation.pdf_benchmark import sha256, verify_candidate_release, verify_sources
from src.evaluation.multidoc_qa import resolve_cases, score_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, default=ROOT / "data/pdf_benchmark/runs/20261115T165227Z-1bd0349b")
    parser.add_argument("--spec", type=Path, default=ROOT / "benchmarks/multidoc_qa/questions.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks/multidoc_qa/protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_candidate_release(ROOT / "benchmarks/pdf_qa")
    dataset, _ = load_inputs(args.retrieval_run)
    protocol = read_json(args.protocol)
    if protocol != read_json(ROOT / "benchmarks/multidoc_qa/protocol.json"):
        raise ValueError("Unsupported comparison protocol")
    if sha256((args.retrieval_run / "dataset.json").read_bytes()) != protocol["source_dataset_sha256"]:
        raise ValueError("Frozen development dataset changed")
    if args.output.exists():
        raise FileExistsError("Result already exists")
    cases = resolve_cases(read_json(args.spec), dataset)
    verify_sources(dataset, ROOT / "data/pdf_benchmark/sources")
    summary = read_json(args.retrieval_run / "summary.json")
    run = ROOT / "data/pdf_benchmark/multidoc_runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    shutil.copytree(args.retrieval_run / "catalog", run / "catalog")
    inputs = {"dataset.json": args.retrieval_run / "dataset.json", "protocol.json": args.protocol, "questions.json": args.spec}
    for name, path in inputs.items():
        (run / name).write_bytes(path.read_bytes())
    source_names = ["scripts/run_multidoc_benchmark.py", "src/evaluation/multidoc_qa.py", "src/evaluation/pdf_benchmark.py",
        "src/retrieval/multi_document_retriever.py", "src/retrieval/structured_retriever.py", "src/generation/context_builder.py",
        "src/generation/strict_context_builder.py", "src/retrieval/context_expander.py", "src/embedding.py"]
    source_hashes = {}
    for name in source_names:
        content = (ROOT/name).read_bytes()
        source_hashes[name] = sha256(content)
        p = run / "source_snapshot" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from src.embedding import load_embedding_resources
    from src.storage.document_store import LocalDocumentStore
    from src.retrieval.multi_document_retriever import MultiDocumentRetriever
    from src.retrieval.context_expander import expand_anchors_first
    from src.generation.context_builder import build_qa_prompt
    from src.generation.strict_context_builder import build_strict_context
    torch.set_num_threads(8)
    model, counter, profile = load_embedding_resources(local_files_only=True)
    if profile.to_dict() != summary["embedding_profile"]:
        raise ValueError("Embedding identity changed")
    store = LocalDocumentStore(run / "catalog")
    members = {}
    chunks = read_json(args.retrieval_run / "chunks.json")
    alignments = {}
    for a in read_json(args.retrieval_run / "alignments.json"):
        alignments.setdefault(a["question_id"], []).append(a)
    records, requests = [], []
    try:
        for build in summary["builds"]:
            member = store.load_retriever(build["document_id"], model, profile)
            if member.index_id != build["index_id"] or member.chunks != chunks[build["paper_id"]]:
                raise ValueError("Saved document chunks/index changed")
            members[build["paper_id"]] = member
        for case in cases:
            pids = [t["paper_id"] for t in case["targets"]]
            retriever = MultiDocumentRetriever([members[p] for p in pids])
            bindings = {p: members[p].paper_id for p in pids}
            start = time.perf_counter()
            all_hits = retriever.search(case["question"], retriever.paper_id, top_k=len(retriever.chunks), candidate_k=50)
            seconds = time.perf_counter()-start
            variants = {}
            for order in ("global", "balanced"):
                ranked = sorted(all_hits, key=lambda c:c["global_fusion_rank"]) if order == "global" else all_hits
                for context_kind in ("top5", "anchor_first"):
                    key = order+"_"+context_kind
                    candidates = ranked[:5]
                    if context_kind == "anchor_first":
                        candidates = expand_anchors_first(candidates, retriever.chunks, window=1)
                    context, sources = build_strict_context(candidates, max_chars=7400)
                    metrics = score_context(sources, retriever.chunks, case["targets"], bindings, alignments)
                    variants[key] = {"metrics": metrics, "context_chars": len(context), "bge_reference_tokens": counter.count(context),
                        "ranked_top5": [c["chunk_id"] for c in ranked[:5]], "sources": sources}
                    prompt = build_qa_prompt(case["question"], context)
                    requests.append({"question_id": case["question_id"], "variant": key, "prompt": prompt,
                        "prompt_sha256": sha256(prompt.encode()), "sources": sources})
            records.append({"question_id": case["question_id"], "selected_papers": pids, "bindings": bindings,
                "positive_papers": [t["paper_id"] for t in case["targets"] if t["answerable"]],
                "shared_search_seconds": seconds, "variants": variants})
    finally:
        store.close()
    write_new(run / "records.json", records)
    write_new(run / "generation_requests.json", requests)
    fields = ("document_presence", "mean_full_evidence_recall", "all_required_papers_hit", "all_gold_complete")
    aggregate = {}
    for key in protocol["variants"]:
        aggregate[key] = {}
        for field in fields:
            values = [r["variants"][key]["metrics"][field] for r in records if r["variants"][key]["metrics"][field] is not None]
            aggregate[key][field] = {"n": len(values), "mean": sum(values)/len(values) if values else None}
    if any(sha256((ROOT/n).read_bytes()) != h for n,h in source_hashes.items()):
        raise ValueError("Source changed during experiment")
    report = {"schema_version":"multidoc-dev-report-v1", "scope":protocol["scope"], "run_id":run.name,
        "question_count":len(records), "paper_count":len(members), "pure_absence_questions":sum(not r["positive_papers"] for r in records),
        "generation_api_calls":0, "answer_quality":None, "attribution_quality":None, "human_review":"pending",
        "protocol":protocol, "embedding_profile":profile.to_dict(), "source_code_sha256":source_hashes,
        "artifacts_sha256":{n:sha256((run/n).read_bytes()) for n in [*inputs,"records.json","generation_requests.json"]},
        "variants":aggregate, "records":records,
        "note":"Document presence is not relevant evidence coverage. Shared candidates are ranked once per question; no per-variant latency comparison. Raw prompts remain private."}
    write_new(run / "summary.json", report)
    write_new(args.output, report)
    print({"run":str(run), "questions":len(records), "variants":aggregate})


if __name__ == "__main__":
    main()
