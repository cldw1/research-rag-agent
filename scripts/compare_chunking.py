"""Compare adaptive/structure on the complete cached QASPER Validation split."""

import argparse
import json
import os
import platform
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.evaluation.chunking_comparison import SELECTION_RULE, STRATEGIES, summarize_comparison, verify_source_coverage
from src.evaluation.qasper_benchmark import digest, evidence_coverage, percentile_summary, prepare_dataset, retrieval_metrics
from src.retrieval.rank_fusion import fused_order
from scripts.compare_rank_fusion import rank_once
from scripts.run_qasper_benchmark import build_or_restore, file_sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run", type=Path, help="Completed Validation fusion-comparison run")
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    import torch
    from datasets import Dataset
    from src.embedding import load_embedding_resources
    from src.generation.strict_context_builder import build_strict_context
    from src.ingestion.structured_chunker import ChunkConfig, StructuredChunker
    from src.retrieval.context_expander import expand_anchors_first

    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    baseline = json.loads((args.baseline_run / "summary.json").read_text(encoding="utf-8"))
    baseline_path = args.baseline_run / "per_question.jsonl"
    if baseline["protocol"]["split"] != "validation" or file_sha(baseline_path) != baseline["records_sha256"]:
        raise ValueError("Expected a verified Validation baseline")
    if json.loads((args.baseline_run / "protocol.json").read_text(encoding="utf-8")) != baseline["protocol"]:
        raise ValueError("Baseline protocol changed")
    old = [json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lookup = {row["question_id"]: row for row in old}
    if len(lookup) != len(old) or len(old) != baseline["question_count"]:
        raise ValueError("Incomplete baseline population")
    source = args.dataset_file
    if source is None:
        matches = sorted((ROOT / "data/raw/hf_cache").rglob("parquet-validation.arrow"))
        if not matches:
            parser.error("No cached Validation split; supply --dataset-file")
        source = matches[0]
    if file_sha(source) != baseline["protocol"]["source_sha256"]:
        raise ValueError("Validation dataset differs from baseline")
    if source.suffix not in (".arrow", ".parquet"):
        parser.error("Expected Arrow or Parquet")
    dataset = Dataset.from_file(str(source)) if source.suffix == ".arrow" else Dataset.from_parquet(str(source))
    documents, questions, corpus, excluded = prepare_dataset(dataset, "validation")
    if (set(q.question_id for q in questions) != set(lookup)
            or digest(sorted(q.question_id for q in questions)) != baseline["protocol"]["question_ids_sha256"]):
        raise ValueError("Question population changed")
    model, counter, profile = load_embedding_resources(local_files_only=True)
    if profile.to_dict() != baseline["protocol"]["embedding_profile"]:
        raise ValueError("Embedding profile changed")
    model.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
    model.encode(["Fixed warmup input for retrieval."], normalize_embeddings=True, show_progress_bar=False)
    config = ChunkConfig(**baseline["protocol"]["chunk_config"])
    if config.strategy != "adaptive" or config.overlap_chars != 0:
        raise ValueError("Expected default adaptive baseline with zero overlap")
    configs = {strategy: replace(config, strategy=strategy) for strategy in STRATEGIES}
    folder = ROOT / "data/chunking_comparisons" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    cache = ROOT / "data/benchmark_cache"
    cache.mkdir(exist_ok=True)
    code_files = ["scripts/compare_chunking.py", "src/evaluation/chunking_comparison.py",
        "src/evaluation/qasper_benchmark.py", "src/ingestion/structured_chunker.py",
        "src/retrieval/structured_retriever.py", "src/retrieval/context_expander.py",
        "src/generation/context_builder.py", "src/generation/strict_context_builder.py",
        "scripts/compare_rank_fusion.py", "scripts/run_qasper_benchmark.py"]
    protocol = {"version": "qasper-chunking-v1", "dataset": "allenai/qasper", "split": "validation",
        "source_sha256": file_sha(source), "question_ids_sha256": digest(sorted(lookup)),
        "paper_ids_sha256": digest(sorted(d.document_id for d in documents)),
        "baseline_records_sha256": file_sha(baseline_path), "configs": {s: asdict(c) for s, c in configs.items()},
        "embedding_profile": profile.to_dict(), "candidate_k": 50, "rrf_k": 60, "k_values": [1, 5, 10],
        "evidence_char_budget": 7400, "neighbor_window": 1, "selection_rule": SELECTION_RULE,
        "bootstrap_seed": 42, "bootstrap_resamples": 2000, "threads": torch.get_num_threads(),
        "query_timing": "Candidate generation including real BGE query encoding plus RRF ordering; one call per question/strategy shared across BM25/Dense/Hybrid quality scoring; not end-to-end application latency.",
        "index_timing": "Cache restoration and fresh encoding reported separately; unequal cache states are not a cold-build speed comparison.",
        "token_accounting": "BGE tokenizer reference counts including special tokens; not generation-model usage.",
        "gold_policy": "Unchanged union of uniquely aligned paragraph IDs; no answer text enters indexing or queries.",
        "execution_order": "document-major; adaptive/structure order alternates by document index",
        "environment": {"python": platform.python_version(), "os": platform.system(), "torch": torch.__version__, "device": "cpu"},
        "code_sha256": {name: file_sha(ROOT / name) for name in code_files}, "generation_api_calls": 0}
    write_json(folder / "protocol.json", protocol)  # Frozen before chunking/ranking/scoring.
    write_json(folder / "excluded.json", excluded)
    grouped = defaultdict(list)
    for question in questions:
        grouped[question.paper_id].append(question)
    records, papers = [], []
    print("CHUNKING_RUN=" + str(folder), flush=True)
    print("CORPUS=" + json.dumps(corpus), flush=True)
    with (folder / "per_question.jsonl").open("x", encoding="utf-8") as stream:
        for number, document in enumerate(documents):
            order = STRATEGIES if number % 2 == 0 else tuple(reversed(STRATEGIES))
            pending = {q.question_id: {"paper_id": q.paper_id, "question_id": q.question_id,
                       "gold_paragraph_count": len(q.gold), "strategies": {}} for q in grouped[document.document_id]}
            paper = {"paper_id": document.document_id, "strategies": {}}
            lengths = {b.block_id: len(b.text) for b in document.blocks}
            for strategy in order:
                chunk_started = time.perf_counter()
                chunks = StructuredChunker(configs[strategy], counter).chunk(document)
                chunk_seconds = time.perf_counter() - chunk_started
                coverage = verify_source_coverage(document, chunks)
                retriever, build = build_or_restore(document, chunks, model, profile, cache)
                paper["strategies"][strategy] = {"chunk_count": len(chunks), "chunking_seconds": chunk_seconds,
                    "vector_bytes": int(retriever.vectors.nbytes), "chunk_tokens": [c["token_count"] for c in chunks],
                    "chunk_characters": [len(c["text"]) for c in chunks], **coverage, **build}
                for question in grouped[document.document_id]:
                    query_started = time.perf_counter()
                    ranks, _ = rank_once(retriever, question.question)
                    orders = {name: sorted(values, key=values.get) for name, values in ranks.items()}
                    orders["hybrid"] = fused_order(ranks)
                    seconds = time.perf_counter() - query_started
                    ranked = {name: [chunks[i] for i in indices[:10]] for name, indices in orders.items()}
                    values = {"methods": {name: {str(k): retrieval_metrics(chunks, found, question, lengths, k)
                              for k in protocol["k_values"]} for name, found in ranked.items()},
                        "ranks": {name: indices[:10] for name, indices in orders.items()},
                        "candidate_query_seconds": seconds, "contexts": {}}
                    if strategy == "adaptive":
                        previous = lookup[question.question_id]
                        for name, old_name in (("dense", "dense"), ("hybrid", "hybrid_rrf60")):
                            if (values["ranks"][name] != previous["ranks"][old_name]
                                    or values["methods"][name]["5"] != previous["methods"][old_name]):
                                raise ValueError("Adaptive baseline ranking/metrics did not reproduce")
                    ordinal = {c["chunk_id"]: i for i, c in enumerate(chunks)}
                    for variant in ("top5", "anchor_first"):
                        candidates = ranked["hybrid"][:5]
                        if variant == "anchor_first":
                            candidates = expand_anchors_first(candidates, chunks, window=1)
                        context, sources = build_strict_context(candidates, max_chars=7400)
                        chosen = [chunks[ordinal[item["chunk_id"]]] for item in sources]
                        if len(context) > 7400:
                            raise ValueError("Evidence budget exceeded")
                        values["contexts"][variant] = {**evidence_coverage(chosen, question, lengths),
                            "chunks": len(chosen), "characters": len(context), "bge_reference_tokens": counter.count(context)}
                    pending[question.question_id]["strategies"][strategy] = values
                del retriever
            for record in pending.values():
                record["strategies"] = {s: record["strategies"][s] for s in STRATEGIES}
                stream.write(json.dumps(record) + "\n")
                records.append(record)
            stream.flush()
            papers.append(paper)
            if (number + 1) % 10 == 0 or number + 1 == len(documents):
                print(f"PROGRESS papers={number+1}/{len(documents)} questions={len(records)}/{len(questions)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if len(records) != len(questions):
        raise ValueError("Incomplete comparison")
    report = {"protocol": protocol, "corpus": corpus, "question_count": len(records),
        "adaptive_baseline_reproduced": True, "all_source_characters_preserved": True,
        "production_default_unchanged": True, **summarize_comparison(records), "engineering": {}}
    for strategy in STRATEGIES:
        builds = [p["strategies"][strategy] for p in papers]
        report["engineering"][strategy] = {key: sum(p[key] for p in builds) for key in (
            "chunk_count", "chunking_seconds", "vector_bytes", "build_seconds", "encoding_and_index_seconds",
            "cache_hit", "merged_paragraph_chunks")}
        report["engineering"][strategy].update(
            chunk_tokens=percentile_summary([t for p in builds for t in p["chunk_tokens"]]),
            chunk_characters=percentile_summary([t for p in builds for t in p["chunk_characters"]]),
            candidate_query_seconds=percentile_summary([r["strategies"][strategy]["candidate_query_seconds"] for r in records]))
    write_json(folder / "per_paper.json", papers)
    report.update(elapsed_seconds=time.perf_counter()-started,
        artifacts={"per_question_sha256": file_sha(folder / "per_question.jsonl"),
                   "protocol_sha256": file_sha(folder / "protocol.json"),
                   "per_paper_sha256": file_sha(folder / "per_paper.json"), "excluded_sha256": file_sha(folder / "excluded.json")},
        scope="Exploratory Validation comparison motivated by Test case inspection; source coverage, not generated answer accuracy. Different chunk units make chunk-level Precision/nDCG descriptive.")
    write_json(folder / "summary.json", report)
    print("CHUNKING_RESULT=" + str(folder / "summary.json"), flush=True)
    print(json.dumps({"recommended_for_followup": report["recommended_for_followup"],
        "hybrid_at_5": {s: report["results"][s]["methods"]["hybrid"]["5"] for s in STRATEGIES},
        "anchor_first": {s: report["results"][s]["contexts"]["anchor_first"] for s in STRATEGIES}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
