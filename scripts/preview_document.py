"""Offline parser/chunker entry point, runnable from PyCharm or PowerShell."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.document import DocumentParseError
from src.ingestion.parsers import ParseOptions, UnifiedDocumentParser
from src.ingestion.structured_chunker import ChunkConfig, StructuredChunker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--strategy", choices=["fixed", "recursive", "structure", "adaptive", "hybrid"], default="adaptive")
    parser.add_argument("--backend", choices=["lightweight", "docling"], default="lightweight")
    parser.add_argument("--artifacts", help="Prepared local Docling model directory")
    parser.add_argument("--formula", action="store_true", help="Enable optional formula recognition model")
    parser.add_argument("--budget", type=int, default=480, help="UTF-8 byte preview budget, not model tokens")
    parser.add_argument("--overlap-chars", type=int, default=0)
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--language", default="eng")
    parser.add_argument("--tessdata")
    parser.add_argument("--encoding", default="auto", choices=["auto", "utf-8-sig", "utf-16", "gb18030"])
    parser.add_argument("--output", type=Path, help="Optional private JSON export; never overwrites an existing file")
    args = parser.parse_args()
    try:
        options = ParseOptions(ocr=args.ocr, ocr_language=args.language, tessdata=args.tessdata, text_encoding=args.encoding,
            backend=args.backend, docling_artifacts_path=args.artifacts, docling_formula=args.formula)
        if args.file.stat().st_size > options.max_bytes:
            raise DocumentParseError("file_too_large", "文件超过大小上限")
        document = UnifiedDocumentParser(options).parse(args.file.read_bytes(), args.file.name)
        config = ChunkConfig(strategy=args.strategy, max_tokens=args.budget, overlap_chars=args.overlap_chars)
        counter = None
        if args.strategy == "hybrid":
            from src.embedding import load_embedding_resources
            _, counter, _ = load_embedding_resources(local_files_only=True)
        chunks = StructuredChunker(config, counter).chunk(document)
        result = {"document": document.to_dict(), "config": asdict(config), "chunks": chunks}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
        print(json.dumps({"type": document.file_type, "status": document.status,
            "pages": document.page_count, "blocks": len(document.blocks), "chunks": len(chunks),
            "counter": chunks[0]["token_count_method"], "issues": [asdict(i) for i in document.issues]},
            ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"{getattr(exc, 'code', 'processing_error')}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
