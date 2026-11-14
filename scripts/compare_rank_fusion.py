"""Choose fusion candidates on the complete Validation split, with cached BGE."""

import argparse
import csv
import json
import os
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
    digest, paper_bootstrap, prepare_dataset, retrieval_metrics,
)
from src.retrieval.rank_fusion import fused_order


POLICIES = {
    "hybrid_rrf60": {"constant": 60, "weights": {"bm25": 1.0, "dense": 1.0}},
    "hybrid_rrf10": {"constant": 10, "weights": {"bm25": 1.0, "dense": 1.0}},
    "hybrid_dense1_5": {"constant": 60, "weights": {"bm25": 1.0, "dense": 1.5}},
}
SELECTION_RULE = (
    "Validation Recall@5 is primary; require non-decreasing Hit@5 and MRR@5, "
    "and a paper-bootstrap paired Recall@5 95% lower bound above zero. "
    "Otherwise retain hybrid_rrf60. Candidate policies are fixed before scoring."
)


def rank_once(retriever, question):
    import numpy as np
    from src.retrieval.structured_retriever import lexical_tokens
    from src.retrieval.uploaded_pdf_retriever import QUERY_PREFIX

    count = min(len(retriever.chunks), 50)
    lexical = np.asarray(retriever.bm25.get_scores(lexical_tokens(question)))
    ordered = np.argsort(-lexical, kind="stable")[:count]
    if len(retriever.model.tokenizer.encode(QUERY_PREFIX + question, add_special_tokens=True,
                                           truncation=False)) > retriever.model.max_seq_length:
        raise ValueError("Question exceeds the embedding input budget")
    vector = np.ascontiguousarray(retriever.model.encode(
        [QUERY_PREFIX + question], normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False), dtype=np.float32)
    _, dense = retriever.index.search(vector, count)
    return {"bm25": {int(i): r for r, i in enumerate(ordered, 1)},
            "dense": {int(i): r for r, i in enumerate(dense[0], 1) if i >= 0}}, lexical


def summarize_records(records, policies):
    metrics = next(iter(records[0]["methods"].values()))
    results = {}
    for policy in ["dense", *policies]:
        means = {key: sum(r["methods"][policy][key] for r in records) / len(records) for key in metrics}
        if policy == "hybrid_rrf60":
            paired = {}
        else:
            paired = {key: paper_bootstrap(records, lambda r, name=policy, metric=key:
                      r["methods"][name][metric] - r["methods"]["hybrid_rrf60"][metric])
                      for key in ("hit", "recall", "mrr", "ndcg", "evidence_f1")}
        results[policy] = {"metrics_at_5": means, "paired_minus_rrf60": paired}
    baseline = results["hybrid_rrf60"]["metrics_at_5"]
    accepted = [name for name in policies if name != "hybrid_rrf60"
        and results[name]["metrics_at_5"]["hit"] >= baseline["hit"]
        and results[name]["metrics_at_5"]["mrr"] >= baseline["mrr"]
        and results[name]["paired_minus_rrf60"]["recall"]["low"] > 0]
    selected = max(accepted, key=lambda name: results[name]["metrics_at_5"]["recall"]) if accepted else "hybrid_rrf60"
    return results, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dataset-file", type=Path, help="Existing official Validation Arrow or Parquet")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    import numpy as np
    import torch
    from datasets import Dataset
    from src.embedding import load_embedding_resources
    from src.ingestion.structured_chunker import ChunkConfig, StructuredChunker
    from scripts.run_qasper_benchmark import build_or_restore, file_sha, write_json

    torch.set_num_threads(args.threads)
    source = args.dataset_file
    if source is None:
        matches = sorted((ROOT / "data/raw/hf_cache").rglob("parquet-validation.arrow"))
        if not matches:
            parser.error("No cached Validation split; provide --dataset-file. Automatic downloads are disabled.")
        source = matches[0]
    if source.suffix not in (".arrow", ".parquet"):
        parser.error("Expected official Validation Arrow or Parquet")
    dataset = Dataset.from_file(str(source)) if source.suffix == ".arrow" else Dataset.from_parquet(str(source))
    documents, questions, corpus, excluded = prepare_dataset(dataset, "validation")
    if not questions:
        raise ValueError("No eligible Validation questions")
    folder = ROOT / "data/fusion_comparisons" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    cache = ROOT / "data/benchmark_cache"
    cache.mkdir(exist_ok=True)
    model, counter, profile = load_embedding_resources(local_files_only=True)
    model.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
    config = ChunkConfig()
    protocol = {"dataset": "allenai/qasper", "split": "validation", "policies": POLICIES,
        "selection_rule": SELECTION_RULE, "source_sha256": file_sha(source),
        "question_ids_sha256": digest(sorted(q.question_id for q in questions)),
        "chunk_config": asdict(config), "embedding_profile": profile.to_dict(),
        "candidate_k": 50, "k": 5, "bootstrap_seed": 42, "bootstrap_resamples": 2000,
        "query_encoding": "one real BGE query vector shared by all compared fusion policies",
        "threads": torch.get_num_threads(), "generation_api_calls": 0}
    write_json(folder / "protocol.json", protocol)
    write_json(folder / "excluded.json", excluded)
    grouped = defaultdict(list)
    for question in questions:
        grouped[question.paper_id].append(question)
    records, builds, chunk_count = [], [], 0
    start = time.perf_counter()
    print("VALIDATION_RUN=" + str(folder), flush=True)
    print("CORPUS=" + json.dumps(corpus), flush=True)
    with (folder / "per_question.jsonl").open("x", encoding="utf-8") as stream:
        for number, document in enumerate(documents, 1):
            chunks = StructuredChunker(config, counter).chunk(document)
            chunk_count += len(chunks)
            retriever, build = build_or_restore(document, chunks, model, profile, cache)
            builds.append(build)
            lengths = {b.block_id: len(b.text) for b in document.blocks}
            for question in grouped[document.document_id]:
                rankings, lexical = rank_once(retriever, question.question)
                orders = {"dense": sorted(rankings["dense"], key=rankings["dense"].get)}
                orders.update({name: fused_order(rankings, **policy) for name, policy in POLICIES.items()})
                record = {"paper_id": question.paper_id, "question_id": question.question_id,
                    "methods": {name: retrieval_metrics(chunks, [chunks[i] for i in order[:10]], question, lengths, 5)
                                for name, order in orders.items()},
                    "ranks": {name: order[:10] for name, order in orders.items()},
                    "positive_lexical_candidates": int(np.count_nonzero(lexical > 0))}
                records.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
            if number % 10 == 0 or number == len(documents):
                print(f"PROGRESS papers={number}/{len(documents)} questions={len(records)}/{len(questions)} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    if len(records) != len(questions):
        raise ValueError("Incomplete validation run")
    results, selected = summarize_records(records, POLICIES)
    corpus["chunk_count"] = chunk_count
    report = {"protocol": protocol, "corpus": corpus, "question_count": len(records),
        "results": results, "selected_policy": selected,
        "elapsed_seconds": time.perf_counter() - start,
        "cache_hits": sum(b["cache_hit"] for b in builds),
        "records_sha256": file_sha(folder / "per_question.jsonl"),
        "scope": "Parameter comparison and selection on Validation; published Test baseline remains unchanged."}
    write_json(folder / "summary.json", report)
    fields = ["paper_id", "question_id", "policy", *next(iter(records[0]["methods"].values()))]
    with (folder / "per_question.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            for policy, metrics in record["methods"].items():
                writer.writerow({"paper_id": record["paper_id"], "question_id": record["question_id"],
                                 "policy": policy, **metrics})
    print("VALIDATION_RESULT=" + str(folder / "summary.json"), flush=True)
    print(json.dumps({"selected_policy": selected, "results": {k:v["metrics_at_5"] for k,v in results.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
