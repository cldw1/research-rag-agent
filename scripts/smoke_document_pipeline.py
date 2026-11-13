"""Real synthetic scan -> local OCR -> HybridChunker -> BGE/FAISS -> reopen.

This is a functionality check, not an accuracy/latency benchmark. No LLM calls.
"""

import argparse
import inspect
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def scanned_fixture():
    import pymupdf
    with pymupdf.open() as source, pymupdf.open() as scan:
        page = source.new_page()
        page.insert_text((50, 70), "Research Methods", fontsize=24)
        page.insert_text((50, 115), "The fixed validation dataset measures retrieval recall.", fontsize=16)
        page.insert_text((50, 150), "Evidence must preserve the original document source.", fontsize=16)
        page.insert_text((50, 200), "科研文献需要保留原始来源。", fontname="china-s", fontsize=18)
        page = source.new_page()
        page.insert_text((50, 70), "Results and Limitations", fontsize=24)
        page.insert_text((50, 115), "This generated fixture does not establish model accuracy.", fontsize=15)
        for x in (50, 320, 540):
            page.draw_line((x, 160), (x, 280))
        for y in (160, 200, 240, 280):
            page.draw_line((50, y), (540, y))
        for point, text in [((65, 186), "Metric"), ((335, 186), "Value"), ((65, 226), "Samples"),
                            ((335, 226), "32"), ((65, 266), "Source pages"), ((335, 266), "2")]:
            page.insert_text(point, text, fontsize=16)
        for page in source:
            target = scan.new_page(width=page.rect.width, height=page.rect.height)
            target.insert_image(target.rect, stream=page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).tobytes("png"))
        extract = {"use_ocr": False} if "use_ocr" in inspect.signature(pymupdf.utils.get_text).parameters else {}
        assert all(not page.get_text("text", **extract).strip() for page in scan)
        return scan.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "data" / "copilot_models" / "docling")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "v5_document_smoke.json")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"] = "1", "1"
    from src.embedding import load_embedding_resources
    from src.ingestion.parsers import ParseOptions
    from src.ingestion.structured_chunker import ChunkConfig
    from src.storage.backup import create_backup, restore_backup
    from src.storage.document_store import LocalDocumentStore

    parent = ROOT / "data" / "copilot_v5_smoke"
    parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    root = workspace / "catalog"
    model, counter, profile = load_embedding_resources(local_files_only=True)
    store = LocalDocumentStore(root)
    try:
        document = store.ingest(scanned_fixture(), "synthetic_scan.pdf", ParseOptions(
            backend="docling", ocr=True, ocr_language="eng+chi_sim", docling_timeout=300,
            docling_artifacts_path=str(args.artifacts.resolve()))).document
        pages = {block.location.page_number for block in document.blocks}
        assert pages == {1, 2}, "Both original scan pages must produce source blocks"
        assert "validation" in document.content.lower(), "English OCR must recognize the fixture text"
        assert "科研" in document.content, "Chinese OCR must recognize the fixture text"
        result = store.build_index(document.document_id, model, profile, counter, ChunkConfig(strategy="hybrid"))
        first = store.load_retriever(document.document_id, model, profile).search("fixed validation dataset", document.document_id)
        assert first and all(item["page_number"] in pages for item in first)
        from sqlalchemy import select
        from sqlalchemy.orm import Session
        from src.storage.models import ChunkRecord
        with Session(store.engine) as session:
            chunks = [row.data for row in session.scalars(select(ChunkRecord).where(ChunkRecord.index_id == result.index_id))]
        token_counts = [chunk["token_count"] for chunk in chunks]
        assert max(token_counts) <= 480
        for chunk in chunks:
            assert {span["page_number"] for span in chunk["source_metadata"]["spans"]} == {chunk["page_number"]}
        create_backup(store, workspace / "backup")
    finally:
        store.close()
    reopened = LocalDocumentStore(root)
    try:
        assert reopened.get_document(document.document_id).to_dict() == document.to_dict()
        second = reopened.load_retriever(document.document_id, model, profile).search("fixed validation dataset", document.document_id)
        assert [row["chunk_id"] for row in first] == [row["chunk_id"] for row in second]
    finally:
        reopened.close()
    restore_backup(workspace / "backup", workspace / "restored")
    restored = LocalDocumentStore(workspace / "restored")
    try:
        assert restored.get_document(document.document_id).to_dict() == document.to_dict()
        recovered = restored.load_retriever(document.document_id, model, profile).search("fixed validation dataset", document.document_id)
        assert [row["source_metadata"] for row in recovered] == [row["source_metadata"] for row in first]
    finally:
        restored.close()
    report = {"dataset": "two fixed generated image-only PDF pages with English, Chinese and a table",
        "purpose": "functional integration; not a comparative or quality benchmark", "llm_api_calls": 0,
        "parser": document.source_metadata["engine"], "dependencies": document.source_metadata["dependencies"],
        "ocr_language": document.source_metadata["ocr_language_resolved"], "pages": document.page_count,
        "status": document.status, "issues": [issue.code for issue in document.issues],
        "block_kinds": dict(Counter(block.kind for block in document.blocks)), "chunks": result.chunk_count,
        "maximum_chunk_tokens": max(token_counts), "embedding": profile.to_dict(),
        "english_and_chinese_fixture_recognized": True, "reopen_preserves_sources": True,
        "backup_restore_preserves_sources": True,
        "limits": "Synthetic fixture only; no formula model, paid LLM, real-paper accuracy or performance claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
