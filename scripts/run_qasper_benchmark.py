"""Run the fixed QASPER evidence retrieval protocol with cached BGE, offline."""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.evaluation.qasper_benchmark import (
    PROTOCOL_VERSION, digest, evidence_coverage, paper_bootstrap, percentile_summary,
    prepare_dataset, retrieval_metrics,
)


def file_sha(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(part)
    return hasher.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def build_or_restore(document, chunks, model, profile, cache):
    import numpy as np
    from src.retrieval.structured_retriever import StructuredDocumentRetriever
    from src.runtime import load_faiss

    faiss = load_faiss()
    identity = digest({"chunks": chunks, "profile": profile.to_dict()})
    array, metadata = cache / (identity + ".npy"), cache / (identity + ".json")
    started = time.perf_counter()
    if array.exists() or metadata.exists():
        manifest = json.loads(metadata.read_text(encoding="utf-8"))
        if manifest["identity"] != identity or file_sha(array) != manifest["vectors_sha256"]:
            raise ValueError("Benchmark cache failed identity/checksum verification")
        vectors = np.load(array, allow_pickle=False)
        index = faiss.IndexFlatIP(profile.dimension)
        index.add(vectors)
        result = StructuredDocumentRetriever.from_saved(document, chunks, model, vectors, index)
        return result, {"cache_hit": True, "build_seconds": time.perf_counter() - started,
                        "encoding_and_index_seconds": 0.0}
    result = StructuredDocumentRetriever(document, chunks, model)
    seconds = time.perf_counter() - started
    with array.open("xb") as stream:
        np.save(stream, result.vectors, allow_pickle=False)
    write_json(metadata, {"identity": identity, "vectors_sha256": file_sha(array)})
    return result, {"cache_hit": False, "build_seconds": seconds,
                    "encoding_and_index_seconds": seconds}


def summarize(records, corpus, protocol, timings, builds, run):
    methods = {}
    for method in protocol["methods"]:
        metrics = {str(k): {key: sum(r["methods"][method][str(k)][key] for r in records) / len(records)
                           for key in records[0]["methods"][method][str(k)]} for k in protocol["k_values"]}
        intervals = {key: paper_bootstrap(records, lambda r, m=method, t=key: r["methods"][m]["5"][t])
                     for key in metrics["5"]}
        methods[method] = {"metrics": metrics, "ci95_at_5": intervals,
                           "query_seconds": percentile_summary(timings[method])}
    paired = {key: paper_bootstrap(records, lambda r, t=key:
              r["methods"]["hybrid"]["5"][t] - r["methods"]["dense"]["5"][t])
              for key in methods["hybrid"]["metrics"]["5"]}
    contexts = {}
    for variant in ("top5", "anchor_first"):
        contexts[variant] = {key: sum(r["contexts"][variant][key] for r in records) / len(records)
                             for key in records[0]["contexts"][variant]}
    return {"protocol": protocol, "corpus": corpus, "question_count": len(records),
            "paper_count": len({r["paper_id"] for r in records}), "methods": methods,
            "paired_hybrid_minus_dense_at_5": paired, "contexts": contexts,
            "timings": {**run, "total_index_build_seconds": sum(b["build_seconds"] for b in builds),
                        "fresh_encoding_and_index_seconds": sum(b["encoding_and_index_seconds"] for b in builds),
                        "cache_hits": sum(b["cache_hit"] for b in builds)},
            "generation_api_calls": 0,
            "scope": "Paper-scoped text evidence retrieval; source coverage and reference evidence F1, not answer generation accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--dataset-file", type=Path, help="Existing QASPER Arrow or Parquet for the selected split")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    import torch
    from datasets import Dataset
    from src.embedding import load_embedding_resources
    from src.ingestion.structured_chunker import ChunkConfig, StructuredChunker
    from src.generation.strict_context_builder import build_strict_context
    from src.retrieval.context_expander import expand_anchors_first

    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    source = args.dataset_file
    if source is None:
        matches = sorted((ROOT / "data" / "raw" / "hf_cache").rglob(f"parquet-{args.split}.arrow"))
        if not matches:
            parser.error("No cached split. Supply an existing official dataset file; automatic downloads are disabled.")
        source = matches[0]
    if source.suffix == ".arrow":
        dataset = Dataset.from_file(str(source))
    elif source.suffix == ".parquet":
        dataset = Dataset.from_parquet(str(source))
    else:
        parser.error("Expected Arrow or Parquet")
    documents, questions, corpus, excluded = prepare_dataset(dataset, args.split)
    if not questions:
        raise ValueError("No eligible text-evidence questions")
    source_hash = file_sha(source)
    run_id = f"qasper-{args.split}-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    output = ROOT / "data" / "benchmark_runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    cache = ROOT / "data" / "benchmark_cache"
    cache.mkdir(parents=True, exist_ok=True)
    model_started = time.perf_counter()
    model, counter, profile = load_embedding_resources(local_files_only=True)
    # Candidate-length checks and evidence accounting tokenize without model forward passes.
    model.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
    model_seconds = time.perf_counter() - model_started
    model.encode(["Fixed warmup input for retrieval."], normalize_embeddings=True, show_progress_bar=False)
    config = ChunkConfig()
    chunk_started = time.perf_counter()
    chunker = StructuredChunker(config, counter)
    chunk_sets = {d.document_id: chunker.chunk(d) for d in documents}
    chunk_seconds = time.perf_counter() - chunk_started
    corpus.update(chunk_count=sum(len(chunks) for chunks in chunk_sets.values()),
                  body_paragraph_count=sum(b.kind == "paragraph" for d in documents for b in d.blocks))
    protocol = {"version": PROTOCOL_VERSION, "dataset": "allenai/qasper", "split": args.split,
        "source_file_sha256": source_hash, "selection": "all uniquely aligned answerable textual-evidence questions",
        "question_ids_sha256": digest(sorted(q.question_id for q in questions)),
        "paper_ids_sha256": digest(sorted(d.document_id for d in documents)),
        "chunk_config": asdict(config), "embedding_profile": profile.to_dict(), "device": "cpu",
        "torch_threads": torch.get_num_threads(), "methods": ["bm25", "dense", "hybrid"],
        "candidate_k_per_method": 50, "rrf_k": 60, "k_values": [1, 5, 10],
        "evidence_char_budget": 7400, "bootstrap_seed": 42, "bootstrap_resamples": 2000,
        "gold_policy": "union of uniquely matched textual paragraph IDs across answerable annotations",
        "evidence_f1_policy": "max paragraph-set F1 across textual references",
        "precision_denominator": "K, including missing slots",
        "query_timing": "actual production search including online query encoding for dense/hybrid; one invocation per question/method",
        "environment": {"python": platform.python_version(), "os": platform.system(), "torch": torch.__version__}}
    write_json(output / "protocol.json", protocol)  # Frozen before any ranking is scored.
    write_json(output / "excluded.json", excluded)
    print("RUN_DIRECTORY=" + str(output), flush=True)
    print("CORPUS=" + json.dumps(corpus), flush=True)
    grouped = defaultdict(list)
    for question in questions:
        grouped[question.paper_id].append(question)
    records, builds, timings = [], [], {method: [] for method in protocol["methods"]}
    with (output / "per_question.jsonl").open("x", encoding="utf-8") as rows:
        for number, document in enumerate(documents, 1):
            chunks = chunk_sets[document.document_id]
            retriever, build = build_or_restore(document, chunks, model, profile, cache)
            builds.append(build)
            lengths = {block.block_id: len(block.text) for block in document.blocks}
            ordinal = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}
            for question in grouped[document.document_id]:
                record = {"paper_id": question.paper_id, "question_id": question.question_id,
                          "gold_paragraph_count": len(question.gold), "methods": {}, "ranks": {}, "contexts": {}}
                hybrid = None
                for method in protocol["methods"]:
                    query_started = time.perf_counter()
                    ranked = retriever.search(question.question, question.paper_id, top_k=10, candidate_k=50, method=method)
                    timings[method].append(time.perf_counter() - query_started)
                    record["methods"][method] = {str(k): retrieval_metrics(chunks, ranked, question, lengths, k)
                                                  for k in protocol["k_values"]}
                    record["ranks"][method] = [ordinal[c["chunk_id"]] for c in ranked]
                    if method == "hybrid":
                        hybrid = ranked
                for variant in ("top5", "anchor_first"):
                    candidates = hybrid[:5] if variant == "top5" else expand_anchors_first(hybrid[:5], chunks, window=1)
                    context, sources = build_strict_context(candidates, max_chars=7400)
                    chosen = [chunks[ordinal[source["chunk_id"]]] for source in sources]
                    record["contexts"][variant] = {**evidence_coverage(chosen, question, lengths),
                        "chunks": len(chosen), "characters": len(context),
                        "bge_reference_tokens": counter.count(context)}
                rows.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows.flush()
                records.append(record)
            if number % 10 == 0 or number == len(documents):
                print(f"PROGRESS papers={number}/{len(documents)} questions={len(records)}/{len(questions)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if len(records) != len(questions):
        raise ValueError("Incomplete evaluation; no summary will be published")
    run = {"model_load_seconds": model_seconds, "chunking_seconds": chunk_seconds,
           "before_aggregation_seconds": time.perf_counter() - started}
    report = summarize(records, corpus, protocol, timings, builds, run)
    report["artifacts"] = {"per_question_jsonl_sha256": file_sha(output / "per_question.jsonl"),
                           "protocol_sha256": file_sha(output / "protocol.json"),
                           "excluded_sha256": file_sha(output / "excluded.json")}
    write_json(output / "summary.json", report)
    with (output / "per_question.csv").open("x", encoding="utf-8", newline="") as stream:
        fields = ["paper_id", "question_id", "method", "k", *next(iter(records[0]["methods"]["bm25"].values()))]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for record in records:
            for method, levels in record["methods"].items():
                for k, metrics in levels.items():
                    writer.writerow({"paper_id": record["paper_id"], "question_id": record["question_id"],
                                     "method": method, "k": k, **metrics})
    print("RESULT=" + str(output / "summary.json"), flush=True)
    print(json.dumps({"questions": len(records), "papers": report["paper_count"],
        "metrics_at_5": {m: v["metrics"]["5"] for m, v in report["methods"].items()},
        "contexts": report["contexts"], "timings": report["timings"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
