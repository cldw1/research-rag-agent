"""Verified, section-aware V3 index for the *specific* HMC test PDF.

This module deliberately does not pretend to discover headings for arbitrary PDFs.
It reuses the page/heading alignment already checked in steps 55–68.
All other PDFs should continue through the original UploadedPDFRetriever.
"""

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import pymupdf
from rank_bm25 import BM25Okapi

from src.ingestion.pdf_chunker_v3 import (
    compact,
    extract_pdf_pages,
    split_long_text,
)
from src.retrieval.bm25_retriever import tokenize
from src.retrieval.uploaded_pdf_retriever import (
    MAX_CHUNKS,
    MAX_PAGES,
    MAX_PDF_BYTES,
    UploadedPDFRetriever,
)


VERIFIED_PDF_SHA256 = (
    "9e886059d25db512b670fed4d064d52cd5af41a8510f4d6adfeffff87ec43e4b"
)
VERIFIED_FILENAME = "HandbookChapter5.pdf"
VERIFIED_CHUNK_COUNT = 247
VERIFIED_HEADING_COUNT = 50


def _flatten_page(blocks):
    lines = []
    for block in blocks:
        for raw_line in block.splitlines():
            line = " ".join(raw_line.split())
            if line:
                lines.append(line)
    return lines


def _load_markers(mapping_path):
    path = Path(mapping_path)
    if not path.is_file():
        raise FileNotFoundError(f"缺少已验证的章节映射文件：{path}")
    mapping = json.loads(path.read_text(encoding="utf-8"))

    if mapping.get("filename") != VERIFIED_FILENAME:
        raise ValueError("章节映射对应的 PDF 文件名不正确")
    if mapping.get("integrity_passed") is not True:
        raise ValueError("章节映射报告没有通过文本完整性检查")
    if mapping.get("unmatched_headings"):
        raise ValueError("章节映射包含未匹配的标题")
    if int(mapping.get("chunk_count", -1)) != VERIFIED_CHUNK_COUNT:
        raise ValueError("章节映射中的 V3 Chunk 数量不一致")

    events = mapping.get("matched_events", [])
    if len(events) != VERIFIED_HEADING_COUNT:
        raise ValueError("章节映射中的标题匹配数量不一致")

    # 旧版映射不一定携带校验和，因此始终检查 PDF 文件本身的 SHA256。
    if mapping.get("pdf_hash") and mapping["pdf_hash"] != VERIFIED_PDF_SHA256:
        raise ValueError("章节映射中的 PDF SHA256 不一致")

    markers = {}
    for event in events:
        page = int(event["page"])
        line_index = int(event["line_index"])
        section = str(event["section"]).strip()
        if page < 1 or line_index < 0 or not section:
            raise ValueError("章节映射中出现无效的位置或空标题")
        markers.setdefault(page, {})[line_index] = section

    return markers


def _build_chunks(pages, markers, paper_id, title):
    chunks = []
    active_section = None
    checked_markers = set()

    for page_index, blocks in enumerate(pages):
        page_number = page_index + 1
        lines = _flatten_page(blocks)
        events = markers.get(page_number, {})
        buffer = []
        chunk_index = 0

        for line_index in events:
            if line_index >= len(lines):
                raise ValueError(f"第 {page_number} 页的章节位置超出了文本范围")

        def flush():
            nonlocal buffer, chunk_index
            if not buffer:
                return
            section = active_section or f"Page {page_number}"
            chunk_text = " ".join(buffer)
            chunks.append({
                "chunk_id": (
                    f"{paper_id}_v3_s{page_index:03d}_"
                    f"p{chunk_index:03d}_c000"
                ),
                "paper_id": paper_id,
                "title": title,
                "section_name": section,
                "section_index": page_index,
                "paragraph_index": chunk_index,
                "piece_index": 0,
                "page_number": page_number,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "is_reference": section.casefold() in {"references", "bibliography"},
            })
            chunk_index += 1
            buffer = []

        for line_index, line in enumerate(lines):
            if line_index in events:
                flush()
                active_section = events[line_index]
                checked_markers.add((page_number, line_index))

            for piece in split_long_text(line, limit=1000):
                candidate = " ".join(buffer + [piece])
                current_length = len(" ".join(buffer))
                if buffer and (
                    len(candidate) > 1000
                    or (current_length >= 450 and len(candidate) > 800)
                ):
                    flush()
                buffer.append(piece)
        flush()

    expected_markers = {
        (page, idx)
        for page, page_events in markers.items()
        for idx in page_events
    }
    if checked_markers != expected_markers:
        raise ValueError("有章节映射落在 PDF 实际页码之外")

    for page_number, blocks in enumerate(pages, start=1):
        original = compact("".join(blocks))
        rebuilt = compact("".join(
            c["text"] for c in chunks if c["page_number"] == page_number
        ))
        if original != rebuilt:
            raise ValueError(f"第 {page_number} 页 V3 文本完整性检查失败")

    if len(chunks) != VERIFIED_CHUNK_COUNT:
        raise ValueError(
            f"V3 Chunk 数量错误：期望 {VERIFIED_CHUNK_COUNT}，实际 {len(chunks)}"
        )
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(f"Chunk 数量超过限制 {MAX_CHUNKS}")
    return chunks


class VerifiedV3PDFRetriever(UploadedPDFRetriever):
    """Keep UploadedPDFRetriever.get_paper_data() and .search() unchanged."""

    has_verified_sections = True
    chunk_mode = "verified_v3"

    def __init__(self, pdf_bytes, filename, model, mapping_path):
        if not isinstance(pdf_bytes, bytes):
            raise ValueError("PDF 内容必须是 bytes")
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise ValueError("PDF 不能超过 20 MB")
        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError("文件不是有效的 PDF 格式")
        if filename != VERIFIED_FILENAME:
            raise ValueError("验证过的 V3 映射仅适用于 HandbookChapter5.pdf")

        digest = hashlib.sha256(pdf_bytes).hexdigest()
        if digest != VERIFIED_PDF_SHA256:
            raise ValueError("PDF 内容校验未通过：不能复用此文件的章节映射")

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.needs_pass:
                raise ValueError("暂不支持需要密码的 PDF")
            page_count = len(doc)
        if page_count > MAX_PAGES:
            raise ValueError(f"PDF 不能超过 {MAX_PAGES} 页")

        markers = _load_markers(mapping_path)
        pages = extract_pdf_pages(pdf_bytes)
        if len(pages) != page_count:
            raise ValueError("PDF 解析页数不一致")

        self.model = model
        self.paper_id = "upload_" + digest[:12]
        self.title = Path(filename).stem
        self.page_count = page_count
        self.chunks = _build_chunks(pages, markers, self.paper_id, self.title)

        tokenized = [tokenize(c["text"]) for c in self.chunks]
        if any(not t for t in tokenized):
            raise ValueError("V3 存在空词项 Chunk，请检查映射")
        self.bm25 = BM25Okapi(tokenized)

        vectors = model.encode(
            [c["text"] for c in self.chunks],
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)
        self.paper_rows = {self.paper_id: list(range(len(self.chunks)))}

        print(
            f"已验证 V3 索引：{self.page_count} 页，"
            f"{len(self.chunks)} Chunk，FAISS={self.index.ntotal}"
        )
