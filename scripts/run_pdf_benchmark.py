"""Offline real-PDF development diagnostic; prepare prompts, never call an LLM."""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.evaluation.pdf_benchmark import (
    align_evidence, evidence_coverage, load_dataset, retrieval_metrics, sha256, verify_sources,
    verify_candidate_release,
)

SOURCE_FILES = [
    "scripts/run_pdf_benchmark.py", "src/evaluation/pdf_benchmark.py", "src/evaluation/qasper_benchmark.py",
    "src/embedding.py", "src/runtime.py", "src/ingestion/document.py", "src/ingestion/parsers.py",
    "src/ingestion/structured_chunker.py", "src/retrieval/structured_retriever.py",
    "src/retrieval/uploaded_pdf_retriever.py", "src/retrieval/context_expander.py",
    "src/storage/document_store.py", "src/storage/database.py", "src/storage/models.py",
    "src/storage/artifacts.py", "src/storage/embedding_profile.py",
    "src/generation/rag_pipeline.py", "src/generation/context_builder.py", "src/generation/strict_context_builder.py",
]


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def mean_scores(records):
    if not records:
        return None
    return {key: sum(r[key] for r in records)/len(records) for key in records[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/pdf_qa/pilot.json")
    parser.add_argument("--sources", type=Path, default=ROOT / "data/pdf_benchmark/sources")
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks/pdf_qa/protocol.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/pdf_benchmark/runs")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    dataset = load_dataset(args.dataset)
    if dataset["status"] != "development_pilot" or any(p["split"] != "dev" for p in dataset["papers"]):
        parser.error("This runner accepts development pilot papers only; final testing requires reviewed frozen labels.")
    if dataset.get("selection_sha256"):
        verify_candidate_release(args.dataset.parent)
        if args.dataset.name != "development.json":
            parser.error("A candidate release can only score its verified development.json view")
    if protocol["protocol_version"] != "pdf-qa-pilot-v1":
        parser.error("Unsupported protocol")
    # The current production retriever/pipeline implement these fixed parameters.
    if (protocol["retrieval"] != {"methods": ["bm25", "dense", "hybrid"],
            "candidate_k": 50, "rrf_k": 60, "k_values": [1, 5, 10]}
            or protocol["contexts"]["variants"] != ["top5", "anchor_first"]
            or protocol["contexts"]["neighbor_window"] != 1
            or protocol["generation"]["default"] != "disabled"):
        parser.error("Protocol differs from executable retrieval/context parameters")
    native, source_check = verify_sources(dataset, args.sources)
    import torch
    from src.embedding import load_embedding_resources
    from src.evaluation.qasper_benchmark import percentile_summary
    from src.generation.rag_pipeline import ResearchRAGPipeline
    from src.ingestion.parsers import ParseOptions
    from src.ingestion.structured_chunker import ChunkConfig
    from src.storage.document_store import LocalDocumentStore

    run = args.output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    for name, path, expected_content in (("dataset.json", args.dataset, dataset), ("protocol.json", args.protocol, protocol)):
        content = path.read_bytes()
        if json.loads(content) != expected_content:
            raise RuntimeError("Input changed during validation")
        with (run / name).open("xb") as stream:
            stream.write(content)
    source_hashes = {}
    for name in SOURCE_FILES:
        content = (ROOT / name).read_bytes()
        source_hashes[name] = sha256(content)
        snapshot = run / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as stream:
            stream.write(content)
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    model, counter, profile = load_embedding_resources(local_files_only=True)
    expected = protocol["embedding"]
    if (profile.model_name != expected["model"] or profile.dimension != expected["dimension"]
            or not expected["normalize"] or expected["device"] != "cpu"):
        raise ValueError("Embedding profile differs from protocol")
    model_seconds = time.perf_counter()-started
    records, builds, alignments, chunk_sets = [], [], [], {}
    parsing_options = ParseOptions(**protocol["parsing"])
    config = ChunkConfig(**protocol["chunking"])
    store = LocalDocumentStore(run / "catalog")
    try:
        for paper in dataset["papers"]:
            before = time.perf_counter()
            content = (args.sources / paper["filename"]).read_bytes()
            ingested = store.ingest(content, paper["filename"], parsing_options)
            ingest_seconds = time.perf_counter()-before
            before = time.perf_counter()
            built = store.build_index(ingested.document.document_id, model, profile, counter, config)
            build_seconds = time.perf_counter()-before
            retriever = store.load_retriever(ingested.document.document_id, model, profile)
            document = retriever.document
            chunk_sets[paper["paper_id"]] = retriever.chunks
            write_json(run / (paper["paper_id"] + "_parsed.json"), document.to_dict())
            pipeline = ResearchRAGPipeline(retriever, max_chars=protocol["contexts"]["max_chars"])
            builds.append({"paper_id": paper["paper_id"], "document_id": document.document_id,
                "document_version": document.document_version, "index_id": built.index_id,
                "chunks": len(retriever.chunks), "parse_status": document.status,
                "issues": [asdict(issue) for issue in document.issues],
                "ingest_seconds": ingest_seconds, "chunk_embedding_index_seconds": build_seconds})
            for question in (q for q in dataset["questions"] if q["paper_id"] == paper["paper_id"]):
                aligned = align_evidence(question, native[paper["paper_id"]], document)
                alignments.extend({"question_id": question["question_id"], **a} for a in aligned)
                record = {"question_id": question["question_id"], "paper_id": paper["paper_id"],
                          "category": question["category"], "answerable": question["answerable"],
                          "methods": {}, "contexts": {}}
                for method in protocol["retrieval"]["methods"]:
                    before = time.perf_counter()
                    ranked = retriever.search(question["question"], document.document_id,
                                              top_k=10, candidate_k=50, method=method)
                    record["methods"][method] = {"seconds": time.perf_counter()-before,
                        "ranked_chunk_ids": [c["chunk_id"] for c in ranked],
                        "metrics": {str(k): retrieval_metrics(retriever.chunks, ranked, aligned, k)
                                    for k in protocol["retrieval"]["k_values"]} if aligned else None}
                for variant in protocol["contexts"]["variants"]:
                    before = time.perf_counter()
                    prepared = pipeline.prepare(question["question"], document.document_id, method=variant)
                    selected = {s["chunk_id"] for s in prepared["sources"]}
                    chunks = [c for c in retriever.chunks if c["chunk_id"] in selected]
                    record["contexts"][variant] = {"seconds": time.perf_counter()-before,
                        "sources": prepared["sources"], "context_chars": prepared["context_chars"],
                        "bge_reference_tokens": counter.count(prepared["context"]),
                        "metrics": evidence_coverage(chunks, aligned) if aligned else None}
                    # Gold/reference answers are deliberately absent from generation input.
                    with (run / "generation_requests.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"question_id": question["question_id"],
                            "paper_id": paper["paper_id"], "variant": variant,
                            "prompt": prepared["prompt"], "prompt_sha256": sha256(prepared["prompt"].encode("utf-8")),
                            "sources": prepared["sources"]}, ensure_ascii=False) + "\n")
                records.append(record)
            print(json.dumps({"paper_id": paper["paper_id"], "chunks": len(retriever.chunks), "done": True}), flush=True)
    finally:
        store.close()
    answerable = [r for r in records if r["answerable"]]
    summary = {"scope": "Real-PDF development retrieval/context diagnostic; AI labels awaiting human review; no generated answers.",
        "dataset_sha256": sha256((run / "dataset.json").read_bytes()), "protocol_sha256": sha256((run / "protocol.json").read_bytes()),
        "question_count": len(records), "answerable_count": len(answerable),
        "unanswerable_count": len(records)-len(answerable), "paper_count": len(builds),
        "human_review_pending": sum(q["annotation"]["human_review"] == "pending" for q in dataset["questions"]),
        "source_check": source_check, "alignment": dict(Counter(a["status"] for a in alignments)),
        "source_code_sha256": source_hashes,
        "source_snapshot_saved": True,
        "embedding_profile": profile.to_dict(), "chunk_config": asdict(config),
        "parse_options": asdict(parsing_options), "threads": args.threads,
        "python": platform.python_version(),
        "libraries": {n: importlib.metadata.version(n) for n in ("PyMuPDF", "sentence-transformers", "faiss-cpu", "numpy")},
        "builds": builds, "methods": {}, "contexts": {},
        "model_load_seconds": model_seconds, "total_seconds": time.perf_counter()-started,
        "generation_api_calls": 0, "answer_metrics": None,
        "confidence_intervals": None, "confidence_note": f"{len(builds)} development papers; no generalization claim or inferential CI."}
    for method in protocol["retrieval"]["methods"]:
        summary["methods"][method] = {"metrics": {
            str(k): mean_scores([r["methods"][method]["metrics"][str(k)] for r in answerable])
            for k in protocol["retrieval"]["k_values"]},
            "query_seconds": percentile_summary([r["methods"][method]["seconds"] for r in records]),
            "per_paper_at_5": {p["paper_id"]: mean_scores([r["methods"][method]["metrics"]["5"]
                for r in answerable if r["paper_id"] == p["paper_id"]]) for p in dataset["papers"]},
            "per_category_at_5": {category: mean_scores([r["methods"][method]["metrics"]["5"]
                for r in answerable if r["category"] == category])
                for category in sorted({r["category"] for r in answerable})}}
    for variant in protocol["contexts"]["variants"]:
        summary["contexts"][variant] = mean_scores([r["contexts"][variant]["metrics"] for r in answerable])
    write_json(run / "alignments.json", alignments)
    write_json(run / "records.json", records)
    write_json(run / "chunks.json", chunk_sets)
    if any(sha256((ROOT / name).read_bytes()) != value for name, value in source_hashes.items()):
        raise RuntimeError("Source code changed during this run; do not publish mixed-version results")
    summary["artifacts_sha256"] = {name: sha256((run / name).read_bytes()) for name in
        ("dataset.json", "protocol.json", "alignments.json", "records.json", "chunks.json", "generation_requests.jsonl")}
    write_json(run / "summary.json", summary)
    with (run / "answer_review.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        fields = ["question_id", "paper_id", "variant", "question", "answerable", "reference_points",
                  "answer", "correctness_0_2", "faithfulness_0_2", "supported_citations",
                  "total_citations", "supported_claims", "total_claims", "abstention_correct", "reviewer", "notes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for question in dataset["questions"]:
            for variant in protocol["contexts"]["variants"]:
                writer.writerow({k: question[k] for k in ["question_id", "paper_id", "question", "answerable"]} |
                                {"variant": variant, "reference_points": " | ".join(question["reference_points"])})
    print(json.dumps({"run": str(run), "questions": len(records), "generation_api_calls": 0,
                      "alignment": summary["alignment"], "answer_metrics": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
