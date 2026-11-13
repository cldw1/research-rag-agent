"""Local-only BGE integration on synthetic files; no LLM API or downloads."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "v2_bge_smoke.json")
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer
    from scripts.benchmark_document_parsers import synthetic_files
    from src.generation.rag_pipeline import ResearchRAGPipeline
    from src.ingestion.parsers import UnifiedDocumentParser
    from src.ingestion.structured_chunker import ModelTokenCounter, StructuredChunker
    from src.retrieval.structured_retriever import StructuredDocumentRetriever

    start = time.perf_counter()
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu",
        cache_folder=str(ROOT / "models" / "embedding_cache"), local_files_only=True)
    loading_seconds = time.perf_counter() - start
    # Local caches may omit the upstream revision. Record the actual loaded
    # tensors as well, rather than claiming an unknown revision is pinned.
    weights_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        weights_hash.update(name.encode())
        weights_hash.update(str(tuple(tensor.shape)).encode())
        weights_hash.update(str(tensor.dtype).encode())
        weights_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    counter = ModelTokenCounter(model.tokenizer, "BAAI/bge-small-en-v1.5")
    records = []
    for filename, content in synthetic_files().items():
        if filename == "scan.pdf":
            continue
        doc = UnifiedDocumentParser().parse(content, filename)
        chunks = StructuredChunker(counter=counter).chunk(doc)
        start = time.perf_counter()
        retriever = StructuredDocumentRetriever(doc, chunks, model)
        build_seconds = time.perf_counter() - start
        prepared = ResearchRAGPipeline(retriever).prepare("What dataset does the experiment use?", doc.document_id)
        records.append({"filename": filename, "sha256": doc.file_hash, "chunks": len(chunks),
            "vectors": retriever.index.ntotal, "dimension": retriever.vectors.shape[1],
            "max_chunk_tokens": max(c["token_count"] for c in chunks), "build_seconds": build_seconds,
            "evidence_contains_expected_text": "fixed validation dataset" in prepared["context"],
            "source_spans_preserved": all(s.get("source_metadata", {}).get("spans") for s in prepared["sources"])})
    model_config = model[0].auto_model.config
    result = {"python": platform.python_version(), "embedding_model": "BAAI/bge-small-en-v1.5",
              "model_revision": getattr(model_config, "_commit_hash", None),
              "loaded_weights_sha256": weights_hash.hexdigest(),
              "model_config_sha256": hashlib.sha256(model_config.to_json_string().encode()).hexdigest(),
              "libraries": {name: importlib.metadata.version(name) for name in ("sentence-transformers", "faiss-cpu", "PyMuPDF", "numpy")},
              "device": "cpu", "local_files_only": True, "normalized": True, "metric": "inner_product",
              "max_sequence_length": model.max_seq_length, "token_counter": counter.identity,
              "load_seconds": loading_seconds, "records": records,
              "limitations": "Synthetic integration only; no real retrieval-quality improvement or LLM accuracy measured."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, indent=2))
    return 0 if all(r["evidence_contains_expected_text"] and r["source_spans_preserved"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
