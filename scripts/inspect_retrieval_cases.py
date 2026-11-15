"""Inspect six fixed diagnostic cases with original text retained only locally."""

import argparse
import json
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.evaluation.case_inspection import fenced_reference, inspect_rankings, select_cases
from src.evaluation.qasper_benchmark import digest, evidence_coverage, paragraph_ids, prepare_dataset
from scripts.compare_rank_fusion import rank_once


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("--audit", type=Path, default=ROOT / "results/qasper_failure_audit.json")
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--public-output", type=Path, help="Optional numeric-only JSON; an existing file is never overwritten")
    args = parser.parse_args()
    if args.public_output and args.public_output.exists():
        parser.error("Public output already exists")
    import numpy as np
    import torch
    from datasets import Dataset
    from src.embedding import load_embedding_resources
    from src.ingestion.structured_chunker import ChunkConfig, StructuredChunker
    from src.retrieval.structured_retriever import lexical_tokens
    from src.retrieval.uploaded_pdf_retriever import QUERY_PREFIX
    from scripts.run_qasper_benchmark import build_or_restore, file_sha, write_json

    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    summary = json.loads((args.baseline_run / "summary.json").read_text(encoding="utf-8"))
    original = args.baseline_run / "per_question.jsonl"
    if file_sha(original) != summary["artifacts"]["per_question_jsonl_sha256"]:
        raise ValueError("Baseline record checksum mismatch")
    baseline = [json.loads(line) for line in original.read_text(encoding="utf-8").splitlines() if line.strip()]
    lookup = {row["question_id"]: row for row in baseline}
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit["baseline_records_sha256"] != file_sha(original):
        raise ValueError("Audit belongs to another baseline")
    context_rescues = {}
    for category in ("fusion_regression", "all_miss"):
        rows = [lookup[row["question_id"]] for row in audit["records"] if row["category"] == category]
        context_rescues[category] = {"questions": len(rows),
            "anchor_first_any_gold": sum(row["contexts"]["anchor_first"]["recall"] > 0 for row in rows),
            "anchor_first_all_gold": sum(row["contexts"]["anchor_first"]["all_gold_covered"] for row in rows)}
    selection = select_cases(audit["records"], baseline)
    folder = ROOT / "data/case_inspection" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "selection.json", selection)  # Fixed before reading case text.
    source = args.dataset_file
    if source is None:
        matches = sorted((ROOT / "data/raw/hf_cache").rglob("parquet-test.arrow"))
        if not matches:
            parser.error("No cached Test file; provide --dataset-file")
        source = matches[0]
    if file_sha(source) != summary["protocol"]["source_file_sha256"]:
        raise ValueError("Dataset differs from the baseline snapshot")
    if source.suffix not in (".arrow", ".parquet"):
        parser.error("Expected Arrow or Parquet")
    dataset = Dataset.from_file(str(source)) if source.suffix == ".arrow" else Dataset.from_parquet(str(source))
    selected_papers = {item["paper_id"] for item in selection["cases"]}
    abstracts = {str(row["id"]): row["abstract"] for row in dataset if str(row["id"]) in selected_papers}
    documents, questions, _, _ = prepare_dataset(dataset, "test")
    selected = {item["question_id"]: item for item in selection["cases"]}
    grouped = defaultdict(list)
    for question in questions:
        if question.question_id in selected:
            grouped[question.paper_id].append(question)
    model, counter, profile = load_embedding_resources(local_files_only=True)
    if profile.to_dict() != summary["protocol"]["embedding_profile"]:
        raise ValueError("Embedding configuration differs from baseline")
    model.tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True
    chunker = StructuredChunker(ChunkConfig(**summary["protocol"]["chunk_config"]), counter)
    public_cases, local_cases = [], []
    for document in documents:
        if document.document_id not in grouped:
            continue
        chunks = chunker.chunk(document)
        cache = ROOT / "data/benchmark_cache"
        identity = digest({"chunks": chunks, "profile": profile.to_dict()})
        if not (cache / (identity + ".npy")).exists():
            raise ValueError("Expected baseline vector cache")
        retriever, _ = build_or_restore(document, chunks, model, profile, cache)
        lengths = {block.block_id: len(block.text) for block in document.blocks}
        for question in grouped[document.document_id]:
            metadata = selected[question.question_id]
            if evidence_coverage(chunks, question, lengths)["full_paragraph_recall"] != 1:
                raise ValueError("Gold source coverage is incomplete")
            rankings, lexical = rank_once(retriever, question.question)
            vector = np.ascontiguousarray(model.encode([QUERY_PREFIX + question.question],
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
            _, dense_indices = retriever.index.search(vector, len(chunks))
            numeric = inspect_rankings(chunks, question, rankings, lexical,
                                       np.argsort(-lexical, kind="stable"), dense_indices[0])
            for method in ("bm25", "dense"):
                actual = sorted(rankings[method], key=rankings[method].get)[:10]
                if actual != lookup[question.question_id]["ranks"][method]:
                    raise ValueError("Original ranking changed")
            if numeric["hybrid_top10_ordinals"] != lookup[question.question_id]["ranks"]["hybrid"]:
                raise ValueError("Original fusion ranking changed")
            public = {**metadata, "chunk_count": len(chunks), "candidate_k": min(50, len(chunks)),
                      "rrf_constant": 60, "reference_paragraph_sets": [sorted(ref) for ref in question.references],
                      "baseline_context_metrics": lookup[question.question_id]["contexts"], **numeric}
            public_cases.append(public)
            selected_ordinals = sorted(set(numeric["hybrid_top10_ordinals"])
                | {r["chunk_ordinal"] for r in numeric["gold_chunks"]}
                | set(list(rankings["dense"])[:5]) | set(list(rankings["bm25"])[:5]))
            local = {**public, "title": document.title, "question": question.question,
                "abstract": abstracts[question.paper_id],
                "gold_sources": [{"block_id": block.block_id, "section_path": list(block.section_path), "text": block.text}
                                 for block in document.blocks if block.block_id in question.gold],
                "chunk_texts": [{"chunk_ordinal": i, "section": chunks[i]["section_name"], "text": chunks[i]["text"],
                                 "query_term_overlap": sorted(set(lexical_tokens(question.question)) & set(lexical_tokens(chunks[i]["text"]))) }
                                for i in selected_ordinals]}
            local_cases.append(local)
            print("INSPECTED=" + metadata["case"], flush=True)
    if len(public_cases) != len(selection["cases"]):
        raise ValueError("Incomplete case population")
    public_cases.sort(key=lambda case: case["case"])
    local_cases.sort(key=lambda case: case["case"])
    public = {"baseline_records_sha256": file_sha(original), "selection": selection,
        "question_count": len(public_cases), "cases": public_cases,
        "diagnostic_context_rescues": context_rescues,
        "original_rankings_reproduced": True, "gold_source_coverage_verified": True,
        "scope": "Purposefully selected diagnostic cases; case observations do not establish population gains."}
    write_json(folder / "numeric_report.json", public)
    if args.public_output:
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.public_output, public)
    write_json(folder / "source_review.local.json", {"cases": local_cases})
    parts = ["# Local source review", "", "Original questions and reference text; keep this file local.", ""]
    for case in local_cases:
        parts += ["## " + case["case"], "", fenced_reference(case["title"]), "", fenced_reference(case["question"])]
        for source in case["gold_sources"]:
            parts += ["", "Gold " + source["block_id"], "", fenced_reference(source["text"])]
    with (folder / "source_review.local.md").open("x", encoding="utf-8") as stream:
        stream.write("\n\n".join(parts) + "\n")
    print("CASE_DIRECTORY=" + str(folder), flush=True)


if __name__ == "__main__":
    main()
