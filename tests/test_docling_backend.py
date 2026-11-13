"""Local adapter tests, independent of model downloads or external documents."""

from types import SimpleNamespace

import pytest

from src.ingestion.document import DocumentParseError
from src.ingestion.docling_parser import convert_page, rapidocr_language, top_left_bbox
from src.ingestion.parsers import ParseOptions, UnifiedDocumentParser, parser_dependencies
from tests.test_structured_ingestion import make_pdf


def native_page():
    box = SimpleNamespace(model_dump=lambda **kwargs: {"l": 10, "t": 700, "r": 300, "b": 650, "coord_origin": "BOTTOMLEFT"})
    prov = SimpleNamespace(page_no=1, bbox=box, charspan=(0, 30))
    values = [SimpleNamespace(label="section_header", self_ref="#/texts/0", level=1,
                            text="Methods", prov=[prov]),
              SimpleNamespace(label="text", self_ref="#/texts/1", text="Fixed validation evidence.", prov=[prov])]
    return SimpleNamespace(iterate_items=lambda: iter((value, 0) for value in values))


def test_docling_native_order_coordinates_and_original_page_mapping():
    blocks, issues, figures, path, _ = convert_page(native_page(), 7, 800, (), 10, ParseOptions())
    assert [block.kind for block in blocks] == ["heading", "paragraph"]
    assert blocks[1].block_id == "b000011" and blocks[1].location.page_number == 7
    assert blocks[1].location.bbox == (10, 100, 300, 150)
    assert blocks[1].section_path == ("Methods",) and not issues and not figures
    assert blocks[1].metadata["provenance"][0]["charspan"] == [0, 30]


def test_docling_page_failure_is_visible_and_keeps_later_real_pages(monkeypatch):
    from src.ingestion import docling_parser
    class Converter:
        calls = 0
        def convert(self, source, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("do not echo a private path or authenticated exception")
            return SimpleNamespace(status="success", document=native_page())
    converter = Converter()
    monkeypatch.setattr(docling_parser, "load_converter", lambda *args: (converter, lambda **kwargs: kwargs))
    progress = []
    doc = UnifiedDocumentParser(ParseOptions(backend="docling"), progress=lambda done, total: progress.append((done, total))).parse(
        make_pdf(["Usable text layer for each page."] * 3), "paper.pdf")
    assert doc.status == "partial" and doc.page_count == 3
    assert {block.location.page_number for block in doc.blocks} == {1, 3}
    assert [(issue.code, issue.page_number) for issue in doc.issues] == [("page_failed", 2)]
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert "private path" not in str(doc.to_dict())


def test_missing_docling_dependency_is_a_clear_global_error(monkeypatch):
    from src.ingestion import docling_parser
    def missing(*args):
        raise DocumentParseError("docling_unavailable", "Install optional parser dependencies")
    monkeypatch.setattr(docling_parser, "load_converter", missing)
    with pytest.raises(DocumentParseError) as error:
        UnifiedDocumentParser(ParseOptions(backend="docling")).parse(make_pdf(["A usable text layer."]), "x.pdf")
    assert error.value.code == "docling_unavailable"


def test_docling_page_limit_is_checked_before_loading_models(monkeypatch):
    from src.ingestion import docling_parser
    def forbidden(*args):
        raise AssertionError("Models must not load before preflight checks")
    monkeypatch.setattr(docling_parser, "load_converter", forbidden)
    with pytest.raises(DocumentParseError) as error:
        UnifiedDocumentParser(ParseOptions(backend="docling", max_pages=1)).parse(make_pdf(["", ""]), "x.pdf")
    assert error.value.code == "too_many_pages"


def test_ocr_language_resolution_does_not_silently_drop_chinese():
    assert rapidocr_language("eng+chi_sim") == "iso:zh-Hans"
    assert rapidocr_language("eng") == "iso:en"
    assert rapidocr_language("chi_tra+eng") == "iso:zh-Hant"
    with pytest.raises(DocumentParseError):
        rapidocr_language("unsupported")


def test_model_manifest_changes_parser_cache_identity(tmp_path):
    options = ParseOptions(backend="docling", docling_artifacts_path=str(tmp_path))
    (tmp_path / "model_manifest.json").write_text('{"revision":1}', encoding="utf-8")
    first = parser_dependencies(options, "pdf")
    (tmp_path / "model_manifest.json").write_text('{"revision":2}', encoding="utf-8")
    assert first != parser_dependencies(options, "pdf")
    assert "model_manifest_sha256" not in parser_dependencies(ParseOptions(), "pdf")


def test_non_finite_coordinates_are_rejected():
    box = SimpleNamespace(model_dump=lambda **kwargs: {"l": float("nan"), "t": 1, "r": 2, "b": 3})
    with pytest.raises(ValueError):
        top_left_bbox(SimpleNamespace(bbox=box), 800)


def test_native_table_serialization_preserves_cells_and_page():
    pytest.importorskip("docling_core")
    from docling_core.types.doc import DoclingDocument, TableData, TableCell
    native = DoclingDocument(name="table")
    native.add_table(data=TableData(num_rows=1, num_cols=2, table_cells=[
        TableCell(text="Samples", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1),
        TableCell(text="32", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2)]))
    blocks, issues, _, _, _ = convert_page(native, 3, 800, (), 0, ParseOptions())
    assert blocks[0].kind == "table" and "32" in blocks[0].text
    assert blocks[0].metadata["table_data"]["num_cols"] == 2
    assert blocks[0].location.page_number == 3 and not issues
