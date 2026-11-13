"""Source-preserving contracts shared by all 2.0 parsers and chunkers."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class SourceLocation:
    # None means the source has no stable page numbering (DOCX/MD/TXT).
    page_number: int | None = None
    paragraph_index: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class DocumentBlock:
    block_id: str
    kind: Literal["paragraph", "heading", "table", "code", "formula", "list"]
    text: str
    section_path: tuple[str, ...]
    location: SourceLocation
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParseIssue:
    code: str
    message: str
    page_number: int | None = None


@dataclass(frozen=True)
class StructuredDocument:
    document_id: str
    filename: str
    file_type: str
    file_hash: str
    file_size: int
    title: str
    author: str
    parser_version: str
    document_version: str
    blocks: tuple[DocumentBlock, ...]
    page_count: int | None = None
    status: Literal["ready", "partial"] = "ready"
    issues: tuple[ParseIssue, ...] = ()
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "StructuredDocument":
        record = dict(value)
        blocks = []
        for block in record["blocks"]:
            item = dict(block)
            location = dict(item["location"])
            if location.get("bbox") is not None:
                location["bbox"] = tuple(location["bbox"])
            item["location"] = SourceLocation(**location)
            item["section_path"] = tuple(item["section_path"])
            blocks.append(DocumentBlock(**item))
        record["blocks"] = tuple(blocks)
        record["issues"] = tuple(ParseIssue(**issue) for issue in record["issues"])
        return cls(**record)


class DocumentParseError(ValueError):
    """Stable error code plus a user-facing message; no document text in errors."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)
