"""Four selectable strategies with exact source spans and explicit budgets.

The default byte counter is a conservative preview budget, NOT a model token
count. Production embeddings must supply the embedding model's tokenizer.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Protocol

from src.ingestion.document import StructuredDocument


class TokenCounter(Protocol):
    identity: str
    def count(self, text: str) -> int: ...


class ByteBudgetCounter:
    identity = "utf8-bytes-preview-v1"

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))


class ModelTokenCounter:
    def __init__(self, tokenizer, model_name: str):
        self.tokenizer = tokenizer
        backend = getattr(tokenizer, "backend_tokenizer", None)
        configuration = json.loads(backend.to_str()) if backend else {}
        # Ignore transient padding/truncation state, retain normalization and rules.
        configuration.pop("padding", None)
        configuration.pop("truncation", None)
        signature = {"vocab": tokenizer.get_vocab(), "backend": configuration,
                     "special_tokens": tokenizer.special_tokens_map,
                     "options": {key: getattr(tokenizer, key, None) for key in (
                         "do_lower_case", "strip_accents", "tokenize_chinese_chars",
                         "clean_up_tokenization_spaces", "model_max_length")}}
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()
        self.identity = f"tokenizer-v2:{model_name}:{digest}:special-tokens-included"

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str = "adaptive"
    max_tokens: int = 480
    max_chars: int = 1200
    overlap_chars: int = 0
    max_chunks: int = 1500

    def __post_init__(self):
        if self.strategy not in ("fixed", "recursive", "structure", "adaptive", "hybrid"):
            raise ValueError("策略应为 fixed、recursive、structure、adaptive 或 hybrid")
        if min(self.max_tokens, self.max_chars, self.max_chunks) < 1:
            raise ValueError("分块预算必须大于 0")
        if not 0 <= self.overlap_chars < self.max_chars:
            raise ValueError("overlap_chars 必须为非负数且小于 max_chars")
        if self.strategy == "hybrid" and self.overlap_chars:
            raise ValueError("HybridChunker 使用结构和 Token 边界，不支持字符 overlap")


class Chunker(Protocol):
    def chunk(self, document: StructuredDocument) -> list[dict]: ...


class StructuredChunker:
    version = "chunker-1.0"

    def __init__(self, config: ChunkConfig | None = None, counter: TokenCounter | None = None):
        self.config = config or ChunkConfig()
        self.counter = counter or ByteBudgetCounter()

    def _ranges(self, text):
        """Return bounded slices of the original text; never decode token fragments."""
        start = 0
        while start < len(text):
            end = min(len(text), start + self.config.max_chars)
            if self.counter.count(text[start:end]) > self.config.max_tokens:
                lo, hi = start, end
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self.counter.count(text[start:mid]) <= self.config.max_tokens:
                        lo = mid
                    else:
                        hi = mid - 1
                end = lo
            if end <= start:
                raise ValueError("Token 预算不足以容纳一个字符及模型特殊 Token")
            if end < len(text) and self.config.strategy != "fixed":
                candidate = text[start:end]
                # Prefer paragraph, then sentence, then word boundaries; Chinese included.
                for pattern in (r"\n\s*\n", r"(?<=[.!?。！？])\s*", r"\s+"):
                    cuts = [m.end() for m in re.finditer(pattern, candidate) if m.end() >= len(candidate) // 2]
                    if cuts:
                        end = start + cuts[-1]
                        break
            # Token lengths need not be monotonic at word boundaries: verify final slice.
            while end > start and self.counter.count(text[start:end]) > self.config.max_tokens:
                end -= 1
            if end <= start:
                raise ValueError("分块无法满足 Token 预算")
            yield start, end
            if end == len(text):
                break
            start = max(start + 1, end - self.config.overlap_chars)

    def chunk(self, document):
        if self.config.strategy == "hybrid":
            from src.ingestion.hybrid_chunker import DoclingHybridChunker
            return DoclingHybridChunker(self.config, self.counter).chunk(document)
        config = self.config
        signature = {"version": self.version, "config": asdict(config), "counter": self.counter.identity}
        index_version = hashlib.sha256(json.dumps({
            "document_version": document.document_version, **signature,
        }, sort_keys=True).encode()).hexdigest()
        chunks, sections, pending = [], {}, None

        def emit(record):
            if record is None:
                return
            if len(chunks) >= config.max_chunks:
                raise ValueError("Chunk 数量超过配置上限")
            number = len(chunks)
            record.update(
                chunk_id=f"{document.document_id}:{index_version}:{number:06d}",
                document_id=document.document_id, paper_id=document.document_id,
                document_version=document.document_version, index_version=index_version,
                title=document.title, parent_chunk_id=None,
                token_count=self.counter.count(record["text"]), token_count_method=self.counter.identity,
                chunking_strategy=config.strategy, chunking_version=self.version,
                char_count=len(record["text"]),
            )
            chunks.append(record)

        for block in document.blocks:
            section_key = (block.location.page_number, block.section_path)
            section_index = sections.setdefault(section_key, len(sections))
            atomic = block.kind in ("table", "formula", "code")
            for piece_index, (start, end) in enumerate(self._ranges(block.text)):
                text = block.text[start:end]
                record = {
                    "text": text, "section_path": list(block.section_path),
                    "section_name": " / ".join(block.section_path) or (
                        f"Page {block.location.page_number}" if block.location.page_number else "Document body"
                    ),
                    "section_index": section_index, "paragraph_index": block.location.paragraph_index,
                    "piece_index": piece_index, "page_number": block.location.page_number,
                    "source_metadata": {
                        "filename": document.filename, "block_kinds": [block.kind],
                        "atomic_split_required": atomic and (start > 0 or end < len(block.text)),
                        "spans": [{"block_id": block.block_id, "char_start": start, "char_end": end,
                                   **asdict(block.location)}],
                    },
                }
                # Recursive greedily packs peers; adaptive only merges short ones.
                # Neither crosses a page/heading or merges table/formula/code blocks.
                can_merge = (
                    config.strategy in ("recursive", "adaptive") and pending is not None
                    and block.kind in ("paragraph", "list")
                    and all(k in ("paragraph", "list") for k in pending["source_metadata"]["block_kinds"])
                    and pending["section_index"] == section_index
                    and len(pending["text"] + "\n\n" + text) <= config.max_chars
                    and self.counter.count(pending["text"] + "\n\n" + text) <= config.max_tokens
                    and not config.overlap_chars
                    and (config.strategy == "recursive" or (
                        self.counter.count(pending["text"]) <= config.max_tokens // 2
                        and self.counter.count(text) <= config.max_tokens // 2
                    ))
                )
                if can_merge:
                    pending["text"] += "\n\n" + text
                    pending["source_metadata"]["spans"].extend(record["source_metadata"]["spans"])
                    pending["source_metadata"]["block_kinds"].append(block.kind)
                else:
                    emit(pending)
                    pending = record
        emit(pending)
        return chunks
