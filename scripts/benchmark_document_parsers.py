"""Reproducible synthetic parser smoke benchmark; no retrieval-quality claims."""

import argparse
import io
import json
import platform
import statistics
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.document import DocumentParseError
from src.ingestion.parsers import PARSER_VERSION, ParseOptions, UnifiedDocumentParser
from src.ingestion.structured_chunker import ChunkConfig, StructuredChunker


def synthetic_files():
    import pymupdf

    sentence = "The experiment uses a fixed validation dataset and reports recall."
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 50), sentence)
        pdf = doc.tobytes(no_new_id=True)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        with pymupdf.open() as scan:
            scan.new_page().insert_image(page.rect, stream=pixmap.tobytes("png"))
            scanned = scan.tobytes(no_new_id=True)
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w") as docx:
        docx.writestr(zipfile.ZipInfo("[Content_Types].xml", date_time=(2026, 1, 1, 0, 0, 0)), '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        docx.writestr(zipfile.ZipInfo("word/document.xml", date_time=(2026, 1, 1, 0, 0, 0)), '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Methods</w:t></w:r></w:p><w:p><w:r><w:t>' + sentence + '</w:t></w:r></w:p></w:body></w:document>')
    return {"text.pdf": pdf, "scan.pdf": scanned, "word.docx": package.getvalue(),
            "notes.md": ("# Methods\n\n" + sentence).encode(), "notes.txt": sentence.encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "v2_parser_smoke.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--tessdata")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    options = ParseOptions(ocr=args.ocr, tessdata=args.tessdata)
    rows = []
    for name, data in synthetic_files().items():
        durations, row = [], {"filename": name}
        for _ in range(args.repeats):
            start = time.perf_counter()
            try:
                doc = UnifiedDocumentParser(options).parse(data, name)
                row.update(status=doc.status, blocks=len(doc.blocks), pages=doc.page_count,
                           recovered_expected_sentence="fixed validation dataset" in doc.content)
                chunk_results = {}
                for strategy in ("fixed", "recursive", "structure", "adaptive"):
                    t0 = time.perf_counter()
                    chunks = StructuredChunker(ChunkConfig(strategy=strategy)).chunk(doc)
                    chunk_results[strategy] = {"chunks": len(chunks), "seconds": time.perf_counter() - t0,
                                               "budget_unit": "utf8_bytes_preview"}
                row["chunking"] = chunk_results
            except DocumentParseError as exc:
                row.update(status="failed", error_code=exc.code)
            durations.append(time.perf_counter() - start)
        row.update(samples=args.repeats, mean_seconds=statistics.mean(durations),
                   p50_seconds=statistics.median(durations), max_seconds=max(durations),
                   timing_scope="parse_plus_four_chunkers_warm_process")
        rows.append(row)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "dataset": "five deterministic synthetic format fixtures, one document each",
              "python": platform.python_version(), "platform": platform.platform(),
              "parser_version": PARSER_VERSION, "ocr_enabled": args.ocr,
              "limitations": ["Not a retrieval-quality benchmark", "No real OCR accuracy measured without installed language data",
                              "No Docling/MinerU/PaddleOCR timing comparison", "Tiny warm-process samples; no production throughput inference"],
              "records": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
