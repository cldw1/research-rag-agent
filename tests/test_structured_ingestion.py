"""Real file-format parsing, source conservation and version isolation, offline."""

import io
import zipfile

import pymupdf
import pytest

from src.ingestion.document import DocumentParseError
from src.ingestion.parsers import ParseOptions, UnifiedDocumentParser
from src.ingestion.structured_chunker import ChunkConfig, StructuredChunker


def make_pdf(texts):
    with pymupdf.open() as pdf:
        for text in texts:
            page = pdf.new_page()
            if text:
                page.insert_text((50, 50), text)
        return pdf.tobytes()


def make_docx(body, *, styles="", document_xml=None, extras=None):
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as archive:
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        archive.writestr("word/document.xml", document_xml or '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + body + '</w:body></w:document>')
        if styles:
            archive.writestr("word/styles.xml", styles)
        for name, value in (extras or {}).items():
            archive.writestr(name, value)
    return result.getvalue()


def test_pdf_preserves_real_page_and_bounding_box_and_reports_missing_text():
    document = UnifiedDocumentParser().parse(make_pdf(["A real research paragraph.", "", "Another page with usable text."]), "paper.pdf")
    assert document.status == "partial" and document.page_count == 3
    assert [b.location.page_number for b in document.blocks] == [1, 3]
    assert all(b.location.bbox and b.location.bbox[0] >= 0 for b in document.blocks)
    assert [(i.code, i.page_number) for i in document.issues] == [("ocr_required", 2)]


def test_pdf_no_ocr_ever_runs_without_opt_in(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("OCR must be opt-in")
    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", forbidden)
    document = UnifiedDocumentParser().parse(make_pdf(["A real usable PDF paragraph.", ""]), "paper.pdf")
    assert document.issues[0].code == "ocr_required"


def test_missing_ocr_dependency_is_page_local_and_no_raw_exception_leaks(monkeypatch):
    def missing(*args, **kwargs):
        raise RuntimeError("private file path / secret")
    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", missing)
    document = UnifiedDocumentParser(ParseOptions(ocr=True)).parse(make_pdf(["A real usable PDF paragraph.", ""]), "scan.pdf")
    assert document.status == "partial" and document.issues[0].code == "ocr_failed"
    assert "secret" not in str(document.to_dict())


def test_ocr_page_results_keep_page_and_coordinates(monkeypatch):
    calls = []
    original = pymupdf.Page.get_text
    def ocr(page, **kwargs):
        calls.append(kwargs)
        return "test-textpage"
    def text(page, *args, **kwargs):
        if kwargs.get("textpage") == "test-textpage":
            return [(1, 2, 100, 40, "Recognized scan text.", 0, 0)]
        return original(page, *args, **kwargs)
    monkeypatch.setattr(pymupdf.Page, "get_textpage_ocr", ocr)
    monkeypatch.setattr(pymupdf.Page, "get_text", text)
    document = UnifiedDocumentParser(ParseOptions(ocr=True, ocr_language="eng+chi_sim")).parse(
        make_pdf(["A real usable PDF paragraph.", ""]), "scan.pdf")
    assert len(calls) == 1 and calls[0]["language"] == "eng+chi_sim"
    assert document.blocks[-1].location.page_number == 2
    assert document.blocks[-1].metadata["ocr"] and document.status == "ready"


def test_encrypted_pdf_and_page_limit_and_corrupt_pdf_have_codes():
    with pymupdf.open() as pdf:
        pdf.new_page()
        encrypted = pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="reader")
    for content, options, code in [
        (encrypted, ParseOptions(), "encrypted_pdf"),
        (make_pdf(["", ""]), ParseOptions(max_pages=1), "too_many_pages"),
        (b"%PDF-corrupt", ParseOptions(), "corrupt_pdf"),
    ]:
        with pytest.raises(DocumentParseError) as error:
            UnifiedDocumentParser(options).parse(content, "x.pdf")
        assert error.value.code == code


def test_docx_heading_table_order_metadata_and_absence_of_fictional_pages():
    body = '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Methods</w:t></w:r></w:p><w:p><w:r><w:t>First paragraph</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Metric</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr></w:tbl><w:p><w:r><w:t>After table</w:t></w:r></w:p>'
    document = UnifiedDocumentParser().parse(make_docx(body, extras={"docProps/core.xml":
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Study</dc:title><dc:creator>Researcher</dc:creator></cp:coreProperties>'}), "study.docx")
    assert document.title == "Study" and document.author == "Researcher"
    assert [b.kind for b in document.blocks] == ["heading", "paragraph", "table", "paragraph"]
    assert document.blocks[2].metadata["rows"] == [["Metric", "Value"]]
    assert all(b.location.page_number is None for b in document.blocks)
    assert document.blocks[-1].section_path == ("Methods",)
    assert [b.location.paragraph_index for b in document.blocks] == [0, 1, 2, 3]


def test_docx_omml_and_images_are_reported_instead_of_silently_disappearing():
    body = '<w:p xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:oMath><m:r><m:t>x=1</m:t></m:r></m:oMath><w:r><w:drawing/></w:r></w:p>'
    document = UnifiedDocumentParser().parse(make_docx(body), "formula.docx")
    assert document.status == "partial"
    assert {i.code for i in document.issues} == {"omml_not_rendered", "image_not_parsed"}
    assert document.blocks[0].metadata["contains_omml"]


@pytest.mark.parametrize("xml", [
    '<!DOCTYPE x [<!ENTITY x "expand">]><x>&x;</x>',
    '<!DOCTYPE x [<!ENTITY x "expand">]><x>&x;</x>'.encode("utf-16"),
])
def test_docx_rejects_entity_expansion(xml):
    with pytest.raises(DocumentParseError) as error:
        UnifiedDocumentParser().parse(make_docx("", document_xml=xml), "unsafe.docx")
    assert error.value.code == "unsafe_xml"


def test_docx_archive_resource_limit():
    with pytest.raises(DocumentParseError) as error:
        UnifiedDocumentParser(ParseOptions(max_archive_bytes=40)).parse(make_docx(""), "large.docx")
    assert error.value.code == "archive_too_large"


def test_docx_unsupported_parts_and_content_are_reported():
    body = '<w:p><w:r><w:t>Usable ordinary paragraph</w:t></w:r></w:p><w:sdt><w:sdtContent><w:p><w:r><w:t>Content control text</w:t></w:r></w:p></w:sdtContent></w:sdt>'
    document = UnifiedDocumentParser().parse(make_docx(body, extras={"word/footnotes.xml": "<footnotes/>"}), "x.docx")
    assert document.status == "partial"
    assert {i.code for i in document.issues} == {"unsupported_parts", "unsupported_block"}


def test_markdown_headings_fences_tables_formulas_and_lines():
    content = "# Paper\n\nIntro.\n\n## Methods\n\n```python\n# not a heading\nx = 1\n```\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n$$\nx = y\n$$\n"
    document = UnifiedDocumentParser().parse(content.encode(), "paper.md")
    assert [b.kind for b in document.blocks] == ["heading", "paragraph", "heading", "code", "table", "formula"]
    code = document.blocks[3]
    assert code.section_path == ("Paper", "Methods")
    assert (code.location.line_start, code.location.line_end) == (7, 10)
    assert all(b.location.page_number is None for b in document.blocks)


def test_non_utf8_requires_explicit_encoding_and_unicode_is_preserved():
    data = "科研文献\n\n统计方法。".encode("gb18030")
    with pytest.raises(DocumentParseError, match="编码"):
        UnifiedDocumentParser().parse(data, "x.txt")
    document = UnifiedDocumentParser(ParseOptions(text_encoding="gb18030")).parse(data, "x.txt")
    assert "科研文献" in document.content
    utf16 = UnifiedDocumentParser().parse("中文测试".encode("utf-16"), "x.txt")
    assert utf16.content == "中文测试"


@pytest.mark.parametrize("data,name,code", [
    (b"", "x.txt", "empty_file"), (b"ordinary text", "x.pdf", "type_mismatch"),
    (b"%PDF-test", "x.md", "type_mismatch"), (b"PK-test", "x.txt", "type_mismatch"),
    (b"abc", "x.exe", "unsupported_type"), (b"\x00binary", "x.txt", "invalid_encoding"),
])
def test_file_validation(data, name, code):
    with pytest.raises(DocumentParseError) as error:
        UnifiedDocumentParser().parse(data, name)
    assert error.value.code == code


def test_file_and_block_limits():
    with pytest.raises(DocumentParseError) as error:
        UnifiedDocumentParser(ParseOptions(max_bytes=3)).parse(b"1234", "x.txt")
    assert error.value.code == "file_too_large"
    with pytest.raises(DocumentParseError) as error:
        UnifiedDocumentParser(ParseOptions(max_blocks=1)).parse(b"first\n\nsecond", "x.txt")
    assert error.value.code == "too_many_blocks"


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "structure", "adaptive"])
@pytest.mark.parametrize("overlap", [0, 5])
def test_chunks_reconstruct_exact_sources_obey_budget_and_cover_all_text(strategy, overlap):
    text = "# Study\n\n" + "中文统计。 Sentence about probability.\n" * 30 + "\nFinal paragraph."
    document = UnifiedDocumentParser().parse(text.encode(), "source.md")
    chunks = StructuredChunker(ChunkConfig(strategy=strategy, max_tokens=80, max_chars=100, overlap_chars=overlap)).chunk(document)
    blocks = {b.block_id: b for b in document.blocks}
    covered = {b.block_id: set() for b in document.blocks}
    for chunk in chunks:
        fragments = []
        assert chunk["token_count"] <= 80 and len(chunk["text"]) <= 100
        assert chunk["page_number"] is None
        for span in chunk["source_metadata"]["spans"]:
            fragments.append(blocks[span["block_id"]].text[span["char_start"]:span["char_end"]])
            covered[span["block_id"]].update(range(span["char_start"], span["char_end"]))
        assert chunk["text"] == "\n\n".join(fragments)
    assert all(covered[b.block_id] == set(range(len(b.text))) for b in document.blocks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_adaptive_merge_rules_never_mix_pages_headings_or_atomic_blocks():
    doc = UnifiedDocumentParser().parse(b"# A\n\nOne.\n\nTwo.\n\n```\ncode\n```\n\n## B\n\nOther.", "x.md")
    chunks = StructuredChunker(ChunkConfig(max_tokens=80)).chunk(doc)
    assert any(c["text"] == "One.\n\nTwo." for c in chunks)
    assert all(len(c["source_metadata"]["block_kinds"]) == 1 for c in chunks if "code" in c["source_metadata"]["block_kinds"])
    pdf = UnifiedDocumentParser().parse(make_pdf(["First page paragraph.", "Second page paragraph."]), "x.pdf")
    assert [c["page_number"] for c in StructuredChunker().chunk(pdf)] == [1, 2]


def test_atomic_blocks_exceeding_budget_are_flagged_and_bounded():
    doc = UnifiedDocumentParser().parse(("```\n" + "print(1)\n" * 50 + "```\n").encode(), "x.md")
    chunks = StructuredChunker(ChunkConfig(max_tokens=40)).chunk(doc)
    assert len(chunks) > 1 and all(c["source_metadata"]["atomic_split_required"] for c in chunks)


def test_ids_deduplicate_bytes_but_never_mix_parser_or_chunker_versions():
    content = b"A paragraph with several research words."
    a = UnifiedDocumentParser().parse(content, "a.txt")
    b = UnifiedDocumentParser().parse(content, "renamed.txt")
    c = UnifiedDocumentParser(ParseOptions(min_text_chars=20)).parse(content, "a.txt")
    assert a.document_id == b.document_id == c.document_id
    assert a.document_version == b.document_version != c.document_version
    normal = StructuredChunker().chunk(a)
    changed = StructuredChunker(ChunkConfig(max_tokens=32)).chunk(a)
    assert not ({x["chunk_id"] for x in normal} & {x["chunk_id"] for x in changed})
    assert normal == StructuredChunker().chunk(a)


def test_budget_and_counting_configuration_cannot_be_silently_ignored():
    doc = UnifiedDocumentParser().parse("中文字".encode(), "x.txt")
    with pytest.raises(ValueError, match="预算"):
        StructuredChunker(ChunkConfig(max_tokens=1)).chunk(doc)
    with pytest.raises(ValueError, match="数量"):
        StructuredChunker(ChunkConfig(max_tokens=3, max_chunks=1)).chunk(doc)
    with pytest.raises(ValueError):
        ChunkConfig(overlap_chars=1200)


def test_recursive_packs_peers_but_structure_keeps_paragraphs():
    doc = UnifiedDocumentParser().parse(b"first paragraph\n\nsecond paragraph", "x.txt")
    recursive = StructuredChunker(ChunkConfig(strategy="recursive")).chunk(doc)
    structure = StructuredChunker(ChunkConfig(strategy="structure")).chunk(doc)
    assert len(recursive) == 1 and len(structure) == 2


def test_benchmark_fixture_bytes_are_reproducible():
    from scripts.benchmark_document_parsers import synthetic_files
    assert synthetic_files() == synthetic_files()
