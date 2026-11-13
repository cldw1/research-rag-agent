"""Trace existing adaptive chunks using cached BGE tokenization, without downloads."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedding import MODEL_NAME
from src.ingestion.parsers import UnifiedDocumentParser
from src.ingestion.structured_chunker import ChunkConfig, ModelTokenCounter, StructuredChunker


def fixture():
    """Synthetic research text; numeric values illustrate syntax, not research results."""
    short = ["We record each observation before analysis.", "We retain the original measurement for inspection."]
    long = " ".join(
        f"For trial {i:02d}, x_i=0.123, y_i=0.456, n=120 and p<0.05. The procedure stores the measurement."
        for i in range(40)
    )
    text = "\n\n".join([
        "# Methods", *short, long,
        "| Symbol | Meaning |\n| --- | --- |\n| n | Sample size |\n| x | Measurement |",
        "```python\nvalues = [1, 2, 3]\nmean = sum(values) / len(values)\n```",
        "## Limitations", "This fixture does not measure retrieval quality.",
    ])
    return text, short, long


def check_sources(document, chunks, config):
    """Verify exact parsed-block slices, full coverage and budget limits."""
    blocks = {block.block_id: block for block in document.blocks}
    covered = {block_id: set() for block_id in blocks}
    for chunk in chunks:
        spans = chunk["source_metadata"]["spans"]
        fragments = []
        locations = set()
        for span in spans:
            block = blocks[span["block_id"]]
            start, end = span["char_start"], span["char_end"]
            assert 0 <= start < end <= len(block.text)
            assert span["page_number"] == block.location.page_number
            assert chunk["section_path"] == list(block.section_path)
            covered[block.block_id].update(range(start, end))
            fragments.append(block.text[start:end])
            locations.add((span["page_number"], block.section_path))
        assert chunk["text"] == "\n\n".join(fragments)
        assert len(locations) == 1
        assert chunk["token_count"] <= config.max_tokens
        assert len(chunk["text"]) <= config.max_chars
    assert all(covered[key] == set(range(len(block.text))) for key, block in blocks.items())
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


def demonstrate(counter):
    text, short, long = fixture()
    parser = UnifiedDocumentParser()
    document = parser.parse(text.encode("utf-8"), "synthetic_methods.md")
    config = ChunkConfig()
    chunker = StructuredChunker(config, counter)
    chunks = chunker.chunk(document)
    check_sources(document, chunks, config)
    assert chunks == chunker.chunk(document)
    assert all(chunk["page_number"] is None for chunk in chunks)
    merged = next(chunk for chunk in chunks if chunk["text"] == "\n\n".join(short))
    long_block = next(block for block in document.blocks if block.text == long)
    pieces = [chunk for chunk in chunks if any(
        span["block_id"] == long_block.block_id for span in chunk["source_metadata"]["spans"])]
    assert len(pieces) > 1
    atomic = [chunk for chunk in chunks if any(
        kind in ("table", "code", "formula") for kind in chunk["source_metadata"]["block_kinds"])]
    assert all(len(chunk["source_metadata"]["spans"]) == 1 for chunk in atomic)
    assert all(not chunk["source_metadata"]["atomic_split_required"] for chunk in atomic)

    # Exercise actual PDF parsing too: page numbers come from generated PDF pages.
    import pymupdf

    with pymupdf.open() as pdf:
        for message in ["First page research paragraph.", "Second page research paragraph."]:
            page = pdf.new_page()
            page.insert_text((50, 50), message)
        pdf_bytes = pdf.tobytes()
    pdf_document = parser.parse(pdf_bytes, "synthetic_pages.pdf")
    pdf_chunks = chunker.chunk(pdf_document)
    check_sources(pdf_document, pdf_chunks, config)
    assert [chunk["page_number"] for chunk in pdf_chunks] == [1, 2]
    assert all(span["bbox"] for chunk in pdf_chunks for span in chunk["source_metadata"]["spans"])

    # Atomic blocks have an explicit fallback rather than unbounded embedding input.
    oversize = parser.parse(("```python\n" + "print(1)\n" * 150 + "```\n").encode(), "oversize_code.md")
    oversize_chunks = chunker.chunk(oversize)
    check_sources(oversize, oversize_chunks, config)
    assert len(oversize_chunks) > 1
    assert all(chunk["source_metadata"]["atomic_split_required"] for chunk in oversize_chunks)

    changed = StructuredChunker(replace(config, max_tokens=240), counter).chunk(document)
    assert not ({chunk["chunk_id"] for chunk in chunks} & {chunk["chunk_id"] for chunk in changed})
    return {
        "fixture": "fixed synthetic Markdown, two generated PDF pages and oversized code",
        "api_calls": 0, "embedding_weights_loaded": False, "downloads": False,
        "tokenizer": MODEL_NAME, "count_method": counter.identity, "config": asdict(config),
        "source_blocks": len(document.blocks), "chunk_count": len(chunks),
        "short_paragraphs_merged": len(merged["source_metadata"]["spans"]),
        "merged_paragraph_tokens": merged["token_count"],
        "merged_source_spans": merged["source_metadata"]["spans"],
        "long_paragraph_chars": len(long), "long_paragraph_pieces": len(pieces),
        "long_piece_ranges": [[span["char_start"], span["char_end"]]
                              for chunk in pieces for span in chunk["source_metadata"]["spans"]],
        "maximum_chunk_tokens": max(chunk["token_count"] for chunk in chunks),
        "maximum_chunk_chars": max(len(chunk["text"]) for chunk in chunks),
        "small_atomic_blocks_kept_separate": len(atomic),
        "oversized_code_pieces": len(oversize_chunks), "oversized_code_flagged": True,
        "pdf_chunk_pages": [chunk["page_number"] for chunk in pdf_chunks],
        "pdf_bounding_boxes_preserved": True, "markdown_pages_are_none": True,
        "exact_parsed_source_coverage": True, "same_config_same_chunks": True,
        "changed_config_disjoint_chunk_ids": True,
        "chunks": [{"ordinal": i, "section": chunk["section_name"],
                    "kinds": chunk["source_metadata"]["block_kinds"],
                    "tokens": chunk["token_count"], "chars": len(chunk["text"]),
                    "span_count": len(chunk["source_metadata"]["spans"])} for i, chunk in enumerate(chunks)],
        "limits": "Source coverage is relative to parsed blocks; no retrieval/answer quality measurement.",
    }


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--output", type=Path, help="Optional JSON report; refuses to overwrite")
    args = arguments.parse_args()
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, cache_dir=str(ROOT / "models" / "embedding_cache"), local_files_only=True)
    except OSError:
        arguments.exit(2, "缺少可用的本地 BGE Tokenizer 缓存；本演示不会下载模型。\n")
    report = demonstrate(ModelTokenCounter(tokenizer, MODEL_NAME))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
