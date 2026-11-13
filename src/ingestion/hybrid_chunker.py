"""Docling HybridChunker on unified blocks, using the actual embedding tokenizer.

Serialized tables/headings are derived text. Provenance intentionally locates
the complete source blocks; it never pretends transformed text has exact offsets.
"""

import hashlib
import importlib.metadata
import json
from dataclasses import asdict, replace
from itertools import groupby

from src.ingestion.structured_chunker import StructuredChunker


class DoclingHybridChunker:
    version = "docling-hybrid-adapter-1.0"

    def __init__(self, config, counter):
        self.config, self.counter = config, counter
        if not hasattr(counter, "tokenizer"):
            raise ValueError("HybridChunker 需要 Embedding 的真实 tokenizer；请先加载本地 Embedding 模型")

    @staticmethod
    def _table(block, table_type, cell_type):
        if block.metadata.get("table_data"):
            return table_type.model_validate(block.metadata["table_data"])
        rows = block.metadata.get("rows")
        if rows is None:
            # A Markdown table is kept as source text unless a real cell model
            # was supplied by the parser; never guess complex merge semantics.
            return None
        columns = max((len(row) for row in rows), default=0)
        cells = [cell_type(text=str(text), start_row_offset_idx=r, end_row_offset_idx=r + 1,
                    start_col_offset_idx=c, end_col_offset_idx=c + 1, column_header=r == 0)
                 for r, row in enumerate(rows) for c, text in enumerate(row)]
        return table_type(num_rows=len(rows), num_cols=columns, table_cells=cells) if columns else None

    def chunk(self, document):
        try:
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
            from docling_core.types.doc import DoclingDocument, DocItemLabel, TableData, TableCell
        except ImportError:
            raise ValueError("HybridChunker 依赖未安装，请安装 requirements-docling.txt") from None
        native_version = importlib.metadata.version("docling-core")
        signature = {"version": self.version, "docling_core": native_version,
            "document_version": document.document_version, "config": asdict(self.config),
            "counter": self.counter.identity, "provenance": "source_block"}
        index_version = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        tokenizer = HuggingFaceTokenizer(tokenizer=self.counter.tokenizer, max_tokens=self.config.max_tokens)
        engine = HybridChunker(tokenizer=tokenizer, merge_peers=True)
        limiter = StructuredChunker(replace(self.config, strategy="recursive"), self.counter)
        chunks = []
        groups = groupby(document.blocks, key=lambda block: (block.location.page_number, block.section_path))
        for section_index, ((page, path), source_blocks) in enumerate(groups):
            source_blocks = list(source_blocks)
            native = DoclingDocument(name=document.title or "Research document")
            mapping = {}
            for level, heading in enumerate(path, 1):
                native.add_heading(text=heading, level=min(level, 6))
            # Headings are supplied as context; heading-only documents still
            # produce chunks rather than an empty index.
            body = [block for block in source_blocks if block.kind != "heading"] or source_blocks
            for block in body:
                table = self._table(block, TableData, TableCell) if block.kind == "table" else None
                if table is not None:
                    item = native.add_table(data=table)
                else:
                    label = DocItemLabel.TEXT
                    if block.kind == "code":
                        label = DocItemLabel.CODE
                    elif block.kind == "formula":
                        label = DocItemLabel.FORMULA
                    item = native.add_text(label=label, text=block.text)
                mapping[item.self_ref] = block
            for raw in engine.chunk(dl_doc=native):
                originals = [mapping[item.self_ref] for item in raw.meta.doc_items if item.self_ref in mapping]
                if not originals:
                    raise ValueError("HybridChunker 输出缺少可追溯来源，未构建索引")
                rendered = engine.contextualize(raw)
                for start, end in limiter._ranges(rendered):
                    if len(chunks) >= self.config.max_chunks:
                        raise ValueError("Chunk 数量超过配置上限")
                    text = rendered[start:end]
                    spans = [{"block_id": block.block_id, "char_start": 0, "char_end": len(block.text),
                              **asdict(block.location)} for block in originals]
                    number = len(chunks)
                    chunks.append({"chunk_id": f"{document.document_id}:{index_version}:{number:06d}",
                        "document_id": document.document_id, "paper_id": document.document_id,
                        "document_version": document.document_version, "index_version": index_version,
                        "title": document.title, "parent_chunk_id": None, "text": text,
                        "section_path": list(path), "section_name": " / ".join(path) or (f"Page {page}" if page else "Document body"),
                        "section_index": section_index, "paragraph_index": originals[0].location.paragraph_index,
                        "piece_index": number, "page_number": page,
                        "token_count": self.counter.count(text), "token_count_method": self.counter.identity,
                        "char_count": len(text), "chunking_strategy": "hybrid", "chunking_version": self.version + ":" + native_version,
                        "source_metadata": {"filename": document.filename, "spans": spans,
                            "block_kinds": [block.kind for block in originals], "span_granularity": "source_block",
                            "context_headings": list(raw.meta.headings or []),
                            "derived_text": True, "docling_core_version": native_version,
                            "atomic_split_required": any(block.kind in ("table", "formula", "code") for block in originals)
                                and (start > 0 or end < len(rendered) or any(self.counter.count(block.text) > self.config.max_tokens for block in originals))}})
        if not chunks:
            raise ValueError("HybridChunker 没有产生可索引内容")
        return chunks
