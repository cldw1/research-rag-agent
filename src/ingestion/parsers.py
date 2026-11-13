"""Bounded, offline PDF / DOCX / Markdown / TXT parsing.

This deliberately implements a modest OOXML reader, not Word pagination or a
general layout model. No links, macros, external relationships or embedded
objects are executed. PDF OCR is explicit and uses installed Tesseract data.
"""

import hashlib
import io
import inspect
import importlib.metadata
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Protocol
from xml.etree import ElementTree as ET

from src.ingestion.document import (
    DocumentBlock, DocumentParseError, ParseIssue, SourceLocation, StructuredDocument,
)


PARSER_VERSION = "structured-1.0"
SUPPORTED_TYPES = ("pdf", "docx", "md", "txt")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


@dataclass(frozen=True)
class ParseOptions:
    max_bytes: int = 20 * 1024 * 1024
    max_pages: int = 80
    max_blocks: int = 10000
    max_archive_bytes: int = 80 * 1024 * 1024
    ocr: bool = False
    ocr_language: str = "eng"
    ocr_dpi: int = 200
    tessdata: str | None = None
    min_text_chars: int = 12
    text_encoding: str = "auto"
    backend: str = "lightweight"
    docling_artifacts_path: str | None = None
    docling_timeout: int = 120
    docling_formula: bool = False

    def __post_init__(self):
        if min(self.max_bytes, self.max_pages, self.max_blocks, self.max_archive_bytes) < 1:
            raise ValueError("文档处理上限必须大于 0")
        if not 72 <= self.ocr_dpi <= 300 or self.min_text_chars < 1:
            raise ValueError("OCR DPI 应为 72–300，文本阈值必须大于 0")
        if not re.fullmatch(r"[a-zA-Z0-9_]+(?:\+[a-zA-Z0-9_]+)*", self.ocr_language):
            raise ValueError("OCR 语言配置无效")
        if self.text_encoding not in ("auto", "utf-8-sig", "utf-16", "gb18030"):
            raise ValueError("不支持的文本编码")
        if self.backend not in ("lightweight", "docling"):
            raise ValueError("解析器应为 lightweight 或 docling")
        if not 10 <= self.docling_timeout <= 600:
            raise ValueError("Docling 处理超时应为 10–600 秒")


def parser_dependencies(options, file_type):
    """Cache keys change when an optional parsing engine is installed/upgraded."""
    names = ["PyMuPDF"] if file_type == "pdf" else []
    if options.backend == "docling" and file_type == "pdf":
        names += ["docling", "docling-slim", "docling-core", "docling-ibm-models", "rapidocr", "onnxruntime"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if options.backend == "docling" and options.docling_artifacts_path:
        manifest = Path(options.docling_artifacts_path) / "model_manifest.json"
        if manifest.is_file():
            versions["model_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return versions


class DocumentParser(Protocol):
    def parse(self, content: bytes, filename: str) -> StructuredDocument: ...


class UnifiedDocumentParser:
    def __init__(self, options: ParseOptions | None = None,
                 progress: Callable[[int, int], None] | None = None):
        self.options = options or ParseOptions()
        self.progress = progress

    def parse(self, content: bytes, filename: str) -> StructuredDocument:
        if not isinstance(content, bytes):
            raise DocumentParseError("invalid_input", "文件内容必须是 bytes")
        if not content:
            raise DocumentParseError("empty_file", "文件为空")
        if len(content) > self.options.max_bytes:
            raise DocumentParseError("file_too_large", "文件超过配置的大小上限")
        # Both slash conventions are stripped, even when running on Linux.
        filename = PurePosixPath(str(filename).replace("\\", "/")).name
        file_type = PurePosixPath(filename).suffix.lower().lstrip(".")
        if file_type not in SUPPORTED_TYPES:
            raise DocumentParseError("unsupported_type", "支持 PDF、DOCX、MD 和 TXT")
        is_pdf = content.startswith(b"%PDF")
        is_zip = content.startswith(b"PK")
        if (file_type == "pdf" and not is_pdf) or (file_type == "docx" and not is_zip):
            raise DocumentParseError("type_mismatch", "文件内容与扩展名不一致")
        if file_type in ("md", "txt") and (is_pdf or is_zip or content.startswith(b"\x89PNG")):
            raise DocumentParseError("type_mismatch", "文件内容与文本扩展名不一致")

        digest = hashlib.sha256(content).hexdigest()
        blocks, issues, metadata = [], [], {}
        title, author, page_count = PurePosixPath(filename).stem, "", None
        if file_type == "pdf":
            if self.options.backend == "docling":
                from src.ingestion.docling_parser import parse_pdf
                blocks, issues, metadata, page_count, pdf_title, author = parse_pdf(content, self.options, self.progress)
            else:
                blocks, issues, metadata, page_count, pdf_title, author = self._pdf(content)
            title = pdf_title or title
        elif file_type == "docx":
            blocks, issues, metadata, docx_title, author = self._docx(content)
            title = docx_title or title
        else:
            text, encoding = self._decode(content)
            metadata["encoding"] = encoding
            blocks, issues = self._text(text, markdown=file_type == "md")
            if file_type == "md":
                title = next((b.text for b in blocks if b.kind == "heading"), title)
        if len(blocks) > self.options.max_blocks:
            raise DocumentParseError("too_many_blocks", "文档结构块数量超过上限")
        if not any(b.text.strip() for b in blocks):
            if any(issue.code == "ocr_failed" for issue in issues):
                raise DocumentParseError("ocr_failed", "OCR 未能提取文字，请检查语言数据配置与扫描质量")
            if self.options.backend == "docling" and file_type == "pdf":
                raise DocumentParseError("docling_no_text", "Docling 未获得可用文本，请检查本地模型、OCR 设置及扫描质量")
            raise DocumentParseError("no_text", "没有可用文本；扫描 PDF 请启用 OCR 并配置语言数据")
        # Include parser dependencies/settings, so incompatible reparses cannot share chunks.
        version = hashlib.sha256(json.dumps({
            "hash": digest, "parser": PARSER_VERSION, "options": asdict(self.options),
            "engine": metadata.get("engine"),
            "dependencies": parser_dependencies(self.options, file_type),
            "blocks": [asdict(b) for b in blocks], "issues": [asdict(i) for i in issues],
        }, sort_keys=True).encode()).hexdigest()
        return StructuredDocument(
            document_id="doc_" + digest, filename=filename, file_type=file_type,
            file_hash=digest, file_size=len(content), title=title, author=author,
            parser_version=PARSER_VERSION, document_version=version,
            blocks=tuple(blocks), issues=tuple(issues), page_count=page_count,
            status="partial" if issues else "ready", source_metadata=metadata,
        )

    def _decode(self, content):
        candidates = [self.options.text_encoding]
        if candidates == ["auto"]:
            candidates = ["utf-16"] if content.startswith((b"\xff\xfe", b"\xfe\xff")) else ["utf-8-sig"]
        for encoding in candidates:
            try:
                text = content.decode(encoding)
            except UnicodeError:
                continue
            if "\x00" not in text and not any(ord(c) < 32 and c not in "\r\n\t\f" for c in text):
                return text.replace("\r\n", "\n").replace("\r", "\n"), encoding
        raise DocumentParseError("invalid_encoding", "文本编码无效；请转为 UTF-8，或显式选择 GB18030")

    def _text(self, text, markdown):
        blocks, issues, path = [], [], []
        buffer, first_line, fence, fenced_kind = [], 1, None, "code"

        def flush(kind="paragraph", end=None):
            if buffer and "\n".join(buffer).strip():
                blocks.append(DocumentBlock(
                    f"b{len(blocks):06d}", kind, "\n".join(buffer).strip(), tuple(path),
                    SourceLocation(paragraph_index=len(blocks), line_start=first_line,
                                   line_end=end),
                ))
            buffer.clear()

        lines = text.splitlines()
        for number, line in enumerate(lines, 1):
            marker = re.match(r"^\s*(`{3,}|~{3,})", line) if markdown else None
            formula = markdown and line.strip() == "$$"
            if fence:
                buffer.append(line)
                closes = (fence == "$$" and formula) or (
                    marker and fence != "$$" and marker[1][0] == fence[0] and len(marker[1]) >= len(fence)
                )
                if closes:
                    flush(fenced_kind, number)
                    fence = None
                continue
            if marker or formula:
                flush(end=number - 1)
                fence, fenced_kind = (marker[1], "code") if marker else ("$$", "formula")
                first_line = number
                buffer.append(line)
                continue
            heading = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line) if markdown else None
            if heading:
                flush(end=number - 1)
                level, name = len(heading[1]), heading[2]
                path = path[:level - 1] + [name]
                blocks.append(DocumentBlock(
                    f"b{len(blocks):06d}", "heading", name, tuple(path),
                    SourceLocation(paragraph_index=len(blocks), line_start=number, line_end=number),
                    {"heading_level": level},
                ))
            elif not line.strip():
                kind = "table" if markdown and len(buffer) > 1 and re.match(r"^\s*\|?\s*:?-{3,}", buffer[1]) else "paragraph"
                flush(kind, number - 1)
            else:
                if not buffer:
                    first_line = number
                buffer.append(line)
            if len(blocks) > self.options.max_blocks:
                raise DocumentParseError("too_many_blocks", "文档结构块数量超过上限")
        kind = fenced_kind if fence else ("table" if markdown and len(buffer) > 1 and re.match(r"^\s*\|?\s*:?-{3,}", buffer[1]) else "paragraph")
        flush(kind, len(lines))
        if fence:
            issues.append(ParseIssue("unclosed_fence", "代码块或公式块未闭合，已保留原文"))
        return blocks, issues

    def _pdf(self, content):
        import pymupdf

        blocks, issues = [], []
        metadata = {"engine": "pymupdf-" + pymupdf.__version__, "ocr_pages": [],
                    "reading_order": "coordinate_sort_not_layout_reconstruction"}
        # API capabilities vary between the installed release and the latest docs.
        # Older releases do not perform automatic OCR and reject use_ocr entirely.
        extract_options = {"sort": True}
        if "use_ocr" in inspect.signature(pymupdf.utils.get_text).parameters:
            extract_options["use_ocr"] = False
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise DocumentParseError("corrupt_pdf", "PDF 无法打开或已损坏") from exc
        with doc:
            if doc.needs_pass:
                raise DocumentParseError("encrypted_pdf", "暂不支持需要密码的 PDF")
            if len(doc) > self.options.max_pages:
                raise DocumentParseError("too_many_pages", "PDF 页数超过配置上限")
            info = doc.metadata or {}
            for i in range(len(doc)):
                try:
                    page = doc[i]
                    # Explicitly disable PyMuPDF 2.x automatic OCR; apply our policy below.
                    records = page.get_text("blocks", **extract_options)
                    text = "".join(str(b[4]) for b in records if len(b) >= 7 and b[6] == 0)
                    usable = sum(c.isalnum() for c in text) >= self.options.min_text_chars and text.count("\ufffd") <= max(1, len(text) // 10)
                    via_ocr = False
                    if not usable:
                        if not self.options.ocr:
                            issues.append(ParseIssue("ocr_required", "此页文本不足，未启用 OCR", i + 1))
                        else:
                            try:
                                tp = page.get_textpage_ocr(language=self.options.ocr_language,
                                    dpi=self.options.ocr_dpi, full=True, tessdata=self.options.tessdata)
                                records = page.get_text("blocks", textpage=tp, **extract_options)
                                via_ocr = True
                                metadata["ocr_pages"].append(i + 1)
                                recognized = "".join(str(b[4]) for b in records if len(b) >= 7 and b[6] == 0)
                                if not any(c.isalnum() for c in recognized):
                                    issues.append(ParseIssue("ocr_empty", "OCR 未识别到文字，请复核扫描质量", i + 1))
                            except Exception:
                                issues.append(ParseIssue("ocr_failed", "OCR 失败，请检查语言数据和扫描质量", i + 1))
                    for paragraph, record in enumerate(records):
                        if len(record) < 7 or record[6] != 0 or not str(record[4]).strip():
                            continue
                        blocks.append(DocumentBlock(
                            f"b{len(blocks):06d}", "paragraph", str(record[4]).strip(), (),
                            SourceLocation(page_number=i + 1, paragraph_index=paragraph,
                                           bbox=tuple(float(v) for v in record[:4])),
                            {"ocr": via_ocr},
                        ))
                        if len(blocks) > self.options.max_blocks:
                            raise DocumentParseError("too_many_blocks", "文档结构块数量超过上限")
                except DocumentParseError:
                    raise
                except Exception:
                    issues.append(ParseIssue("page_failed", "页面提取失败", i + 1))
                if self.progress:
                    self.progress(i + 1, len(doc))
            return blocks, issues, metadata, len(doc), info.get("title", ""), info.get("author", "")

    def _docx(self, content):
        blocks, issues, path = [], [], []
        metadata = {"engine": "stdlib-ooxml-1", "limitations": [
            "no_pagination", "no_layout_reconstruction", "no_image_ocr", "no_tracked_changes",
        ]}
        title, author = "", ""
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                entries = archive.infolist()
                names = [entry.filename for entry in entries]
                if len(entries) > 4000 or sum(e.file_size for e in entries) > self.options.max_archive_bytes:
                    raise DocumentParseError("archive_too_large", "DOCX 解压大小或成员数量超过上限")
                if len(names) != len(set(names)) or any(e.flag_bits & 1 for e in entries):
                    raise DocumentParseError("unsafe_archive", "DOCX 存在重复或加密成员")
                if "word/document.xml" not in names or "[Content_Types].xml" not in names:
                    raise DocumentParseError("type_mismatch", "ZIP 文件不是有效的 DOCX")

                def xml(name):
                    raw = archive.read(name)
                    # Word documents do not need DTDs; reject entity expansion inputs.
                    normalized = raw.replace(b"\x00", b"").upper()
                    if b"<!DOCTYPE" in normalized or b"<!ENTITY" in normalized:
                        raise DocumentParseError("unsafe_xml", "DOCX XML 包含不支持的实体定义")
                    return ET.fromstring(raw)

                root = xml("word/document.xml")
                skipped_parts = [name for name in names if re.match(
                    r"word/(?:footnotes|endnotes|comments|header\d*|footer\d*)\.xml$", name)]
                if skipped_parts:
                    issues.append(ParseIssue("unsupported_parts", "文档包含未提取的页眉、页脚、脚注或批注；请查看原文"))
                    metadata["skipped_parts"] = skipped_parts
                if any(n.tag in (W + "ins", W + "del") for n in root.iter()):
                    issues.append(ParseIssue("tracked_changes", "文档包含修订标记，未恢复 Word 的最终显示视图"))
                styles = {}
                if "word/styles.xml" in names:
                    for style in xml("word/styles.xml").findall(W + "style"):
                        name = style.find(W + "name")
                        styles[style.get(W + "styleId")] = name.get(W + "val", "") if name is not None else ""
                if "docProps/core.xml" in names:
                    core = xml("docProps/core.xml")
                    title = core.findtext("{http://purl.org/dc/elements/1.1/}title", "")
                    author = core.findtext("{http://purl.org/dc/elements/1.1/}creator", "")
                body = root.find(W + "body")
                if body is None:
                    raise DocumentParseError("corrupt_docx", "DOCX 缺少正文")

                def paragraph_text(node):
                    return "".join((n.text or "") if n.tag == W + "t" else
                                   "\t" if n.tag == W + "tab" else "\n"
                                   for n in node.iter() if n.tag in (W + "t", W + "tab", W + "br"))

                for ordinal, node in enumerate(body):
                    kind, extra = "paragraph", {}
                    if node.tag == W + "p":
                        text = paragraph_text(node)
                        style = node.find(W + "pPr/" + W + "pStyle")
                        style_id = style.get(W + "val", "") if style is not None else ""
                        outline = node.find(W + "pPr/" + W + "outlineLvl")
                        match = re.search(r"(?:heading|标题)\s*([1-6])", styles.get(style_id, style_id), re.I)
                        level = int(match[1]) if match else None
                        if outline is not None and outline.get(W + "val", "").isdigit():
                            value = int(outline.get(W + "val")) + 1
                            level = value if value <= 6 else level
                        if level and text.strip():
                            kind = "heading"
                            path = path[:level - 1] + [text.strip()]
                            extra["heading_level"] = level
                        elif node.find(W + "pPr/" + W + "numPr") is not None:
                            kind = "list"
                    elif node.tag == W + "tbl":
                        rows = [[paragraph_text(cell).strip() for cell in row.findall(W + "tc")]
                                for row in node.findall(W + "tr")]
                        text, kind, extra = "\n".join(" | ".join(row) for row in rows), "table", {"rows": rows}
                    else:
                        if node.tag != W + "sectPr" and any(n.tag == W + "t" for n in node.iter()):
                            issues.append(ParseIssue("unsupported_block", "部分 Word 内容控件未提取，请查看原文"))
                        continue
                    if any(n.tag.endswith("}oMath") for n in node.iter()):
                        issues.append(ParseIssue("omml_not_rendered", "Word 数学公式未转换为 LaTeX，已标记对应段落"))
                        extra["contains_omml"] = True
                        text += "\n[Word mathematical object: inspect original document]"
                    if any(n.tag.endswith("}drawing") for n in node.iter()):
                        issues.append(ParseIssue("image_not_parsed", "Word 图片未执行 OCR，需查看原文"))
                    if text.strip():
                        blocks.append(DocumentBlock(f"b{len(blocks):06d}", kind, text.strip(),
                            tuple(path), SourceLocation(paragraph_index=ordinal), extra))
                    if len(blocks) > self.options.max_blocks:
                        raise DocumentParseError("too_many_blocks", "文档结构块数量超过上限")
        except DocumentParseError:
            raise
        except (zipfile.BadZipFile, ET.ParseError, KeyError, RuntimeError, OSError, ValueError, LookupError) as exc:
            raise DocumentParseError("corrupt_docx", "DOCX 文件损坏或无法解析") from exc
        return blocks, issues, metadata, title, author
