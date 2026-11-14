"""Reproduce published failures and inspect source coverage and candidate ranks."""

import argparse
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.evaluation.qasper_benchmark import (
    digest, evidence_coverage, paragraph_ids, prepare_dataset,
)
from src.retrieval.rank_fusion import fused_order
from scripts.compare_rank_fusion import rank_once


def failure_groups(records):
    selected = {}
    for record in records:
        hit = {name: record["methods"][name]["5"]["hit"] for name in ("bm25", "dense", "hybrid")}
        if hit["dense"] and not hit["hybrid"]:
            selected[record["question_id"]] = ("fusion_regression", record)
        elif not any(hit.values()):
            selected[record["question_id"]] = ("all_miss", record)
    return selected


def diagnose(chunks, question, baseline, rankings, lexical, category, lengths):
    orders = {name: sorted(ranking, key=ranking.get)[:10] for name, ranking in rankings.items()}
    orders["hybrid"] = fused_order(rankings)[:10]
    if any(orders[name] != baseline["ranks"][name] for name in orders):
        raise ValueError("The published baseline rankings could not be reproduced")
    if evidence_coverage(chunks, question, lengths)["full_paragraph_recall"] != 1.0:
        raise ValueError("Gold source is not completely represented by the chunk set")
    gold = {index for index, chunk in enumerate(chunks) if paragraph_ids(chunk) & question.gold}
    first_dense = min((rank for index, rank in rankings["dense"].items() if index in gold), default=None)
    first_bm25 = min((rank for index, rank in rankings["bm25"].items() if index in gold), default=None)
    flags = {"questions": 1,
        "zero_lexical_branch": int(not any(score != 0 for score in lexical)),
        "no_positive_lexical_branch": int(not any(score > 0 for score in lexical)),
        "no_gold_positive_lexical_score": int(not any(lexical[index] > 0 for index in gold)),
        "zero_score_chunk_in_hybrid_top5": int(any(lexical[index] == 0 for index in orders["hybrid"][:5])),
        "gold_outside_both_candidate_pools": int(first_dense is None and first_bm25 is None),
        "gold_present_in_either_candidate_pool": int(first_dense is not None or first_bm25 is not None),
        "hybrid_hit_at10": int(baseline["methods"]["hybrid"]["10"]["hit"]),
        "any_method_hit_at10": int(any(baseline["methods"][name]["10"]["hit"] for name in orders))}
    if first_dense is not None and first_dense <= 5:
        flags["first_dense_gold_rank_" + str(first_dense)] = 1
    record = {"paper_id": question.paper_id, "question_id": question.question_id, "category": category,
        "chunk_count": len(chunks), "positive_lexical_candidates": sum(score > 0 for score in lexical),
        "zero_lexical_candidates": sum(score == 0 for score in lexical), "gold_chunk_count": len(gold),
        "first_dense_gold_rank": first_dense, "first_bm25_gold_rank": first_bm25,
        "original_hybrid_ordinals": orders["hybrid"]}
    # NumPy comparisons are converted before JSON export, preserving a numeric-only schema.
    record["positive_lexical_candidates"] = int(record["positive_lexical_candidates"])
    record["zero_lexical_candidates"] = int(record["zero_lexical_candidates"])
    return record, flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dataset-file", type=Path, help="The exact Arrow or Parquet used by the baseline")
    args = parser.parse_args()
    import torch
    from datasets import Dataset
    from src.embedding import load_embedding_resources
    from src.ingestion.structured_chunker import ChunkConfig, StructuredChunker
    from scripts.run_qasper_benchmark import build_or_restore, file_sha, write_json

    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    summary = json.loads((args.baseline_run / "summary.json").read_text(encoding="utf-8"))
    source_records = args.baseline_run / "per_question.jsonl"
    if file_sha(source_records) != summary["artifacts"]["per_question_jsonl_sha256"]:
        raise ValueError("Baseline record checksum mismatch")
    baseline = [json.loads(line) for line in source_records.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = failure_groups(baseline)
    split = summary["protocol"]["split"]
    source = args.dataset_file
    if source is None:
        matches = sorted((ROOT / "data/raw/hf_cache").rglob(f"parquet-{split}.arrow"))
        if not matches:
            parser.error("No cached baseline split; provide --dataset-file. Automatic downloads are disabled.")
        source = matches[0]
    if file_sha(source) != summary["protocol"]["source_file_sha256"]:
        raise ValueError("Dataset snapshot differs from baseline")
    if source.suffix not in (".arrow", ".parquet"):
        parser.error("Expected official Arrow or Parquet")
    dataset = Dataset.from_file(str(source)) if source.suffix == ".arrow" else Dataset.from_parquet(str(source))
    documents, questions, corpus, _ = prepare_dataset(dataset, split)
    model, counter, profile = load_embedding_resources(local_files_only=True)
    if profile.to_dict() != summary["protocol"]["embedding_profile"]:
        raise ValueError("Embedding profile differs from baseline")
    model.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
    grouped = defaultdict(list)
    for question in questions:
        if question.question_id in selected:
            grouped[question.paper_id].append(question)
    chunker = StructuredChunker(ChunkConfig(**summary["protocol"]["chunk_config"]), counter)
    counts, records = defaultdict(Counter), []
    cache = ROOT / "data/benchmark_cache"
    for document in documents:
        if document.document_id not in grouped:
            continue
        chunks = chunker.chunk(document)
        identity = digest({"chunks": chunks, "profile": profile.to_dict()})
        if not (cache / (identity + ".npy")).exists():
            raise ValueError("Expected a verified baseline vector cache")
        retriever, _ = build_or_restore(document, chunks, model, profile, cache)
        lengths = {block.block_id: len(block.text) for block in document.blocks}
        for question in grouped[document.document_id]:
            category, prior = selected[question.question_id]
            rankings, lexical = rank_once(retriever, question.question)
            record, flags = diagnose(chunks, question, prior, rankings, lexical, category, lengths)
            records.append(record)
            counts[category].update(flags)
        if len(records) % 20 < len(grouped[document.document_id]):
            print(f"AUDIT_PROGRESS={len(records)}/{len(selected)}", flush=True)
    if len(records) != len(selected):
        raise ValueError("Incomplete diagnostic population")
    report = {"dataset": "allenai/qasper", "split": split, "baseline_questions": len(baseline),
        "diagnosed_questions": len(records), "counts": {k:dict(v) for k,v in counts.items()},
        "source_chunks_completely_cover_all_gold": True, "original_rankings_reproduced": True,
        "baseline_records_sha256": file_sha(source_records),
        "source_file_sha256": file_sha(source), "records": records,
        "scope": "Post-evaluation diagnosis of a fixed published Test baseline; not a new unseen evaluation."}
    output = ROOT / "data/failure_analysis" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", report)
    print("AUDIT_RESULT=" + str(output / "summary.json"), flush=True)
    print(json.dumps(report["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
