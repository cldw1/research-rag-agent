"""Optional local Docling/RapidOCR PDF backend; page failures stay page-local."""

import io
import inspect
import math
import time

from src.ingestion.document import DocumentBlock, DocumentParseError, ParseIssue, SourceLocation


def rapidocr_language(language):
    codes = set(language.split("+"))
    if not codes <= {"eng", "chi_sim", "chi_tra"}:
        raise DocumentParseError("ocr_language_unsupported", "此 Docling 入口支持 eng、chi_sim、chi_tra 及其组合")
    # RapidOCR takes one recognizer language. Its current Chinese multilingual
    # recognizer also covers English; never let the engine silently drop Chinese
    # by passing an unsupported multi-language list beginning with English.
    return "iso:zh-Hans" if "chi_sim" in codes else "iso:zh-Hant" if "chi_tra" in codes else "iso:en"


def load_converter(options, force_full_page_ocr):
    try:
        from docling.datamodel.base_models import DocumentStream, InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
        from docling.datamodel.object_detection_engine_options import OnnxRuntimeObjectDetectionEngineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        raise DocumentParseError("docling_unavailable", "Docling 依赖未安装，请安装 requirements-docling.txt") from None
    lang = [rapidocr_language(options.ocr_language)]
    pipeline = PdfPipelineOptions(
        do_ocr=options.ocr, do_table_structure=True,
        do_formula_enrichment=options.docling_formula,
        do_picture_description=False, do_picture_classification=False,
        enable_remote_services=False, allow_external_plugins=False,
        accelerator_options={"device": "cpu", "num_threads": 4},
        document_timeout=float(options.docling_timeout),
        artifacts_path=options.docling_artifacts_path,
        ocr_options=RapidOcrOptions(lang=lang, backend="onnxruntime", scale=options.ocr_dpi / 72,
                                  force_full_page_ocr=force_full_page_ocr),
    )
    pipeline.layout_options.engine_options = OnnxRuntimeObjectDetectionEngineOptions()
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)})
    return converter, DocumentStream


def top_left_bbox(provenance, page_height):
    box = getattr(provenance, "bbox", None)
    if box is None:
        return None
    values = box.model_dump(mode="json") if hasattr(box, "model_dump") else dict(box)
    left, top, right, bottom = [float(values[key]) for key in ("l", "t", "r", "b")]
    origin = str(values.get("coord_origin", "TOPLEFT")).upper().replace("_", "")
    if "BOTTOMLEFT" in origin:
        top, bottom = page_height - top, page_height - bottom
    result = left, min(top, bottom), right, max(top, bottom)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Non-finite provenance")
    return result


def convert_page(native, page_number, height, section_path, offset, options):
    """Consume native reading order; preserve item IDs and all source provenance."""
    blocks, issues, figures = [], [], []
    path, title = list(section_path), ""
    for item, _ in native.iterate_items():
        label = getattr(item, "label", None)
        if label is None:
            continue  # Structural group, not an actual source element.
        label = str(getattr(label, "value", label)).lower()
        identifier = str(getattr(item, "self_ref", ""))
        sources = []
        for prov in getattr(item, "prov", []) or []:
            # Each conversion receives exactly one original page. Page 1 maps
            # explicitly to that page; no Word-style pagination is inferred.
            if getattr(prov, "page_no", 1) != 1:
                raise ValueError("Unexpected page in single-page conversion")
            box = top_left_bbox(prov, height)
            sources.append({"page_number": page_number, "bbox": list(box) if box else None,
                            "charspan": list(getattr(prov, "charspan", ()) or ())})
        bbox = tuple(sources[0]["bbox"]) if sources and sources[0]["bbox"] else None
        extra = {"docling_ref": identifier, "docling_label": label, "provenance": sources}
        text, kind = str(getattr(item, "text", "") or ""), "paragraph"
        if label == "picture":
            caption = item.caption_text(native) if hasattr(item, "caption_text") else ""
            figures.append({"figure_id": identifier, "caption": caption, "page_number": page_number, "bbox": list(bbox) if bbox else None})
            text = caption
            extra["figure_reference"] = True
        elif label == "table":
            data = item.data.model_dump(mode="json")
            if len(data.get("table_cells", [])) > 50000:
                raise DocumentParseError("table_too_large", "表格单元格数量超过上限")
            text, kind = item.export_to_markdown(doc=native), "table"
            extra["table_data"] = data
        elif label in ("title", "section_header"):
            kind = "heading"
            level = 1 if label == "title" else max(1, min(6, int(getattr(item, "level", 1))))
            path = path[:level - 1] + [text.strip()]
            extra["heading_level"] = level
            if label == "title" and not title:
                title = text.strip()
        elif label == "list_item":
            kind = "list"
        elif label == "code":
            kind = "code"
        elif label == "formula":
            kind = "formula"
            extra["formula_enrichment"] = options.docling_formula
            if not options.docling_formula:
                issues.append(ParseIssue("formula_not_enriched", "公式仅保留当前提取内容，未启用公式识别模型", page_number))
        if not text.strip():
            continue
        if offset + len(blocks) >= options.max_blocks:
            raise DocumentParseError("too_many_blocks", "文档结构块数量超过上限")
        blocks.append(DocumentBlock(f"b{offset + len(blocks):06d}", kind, text.strip(), tuple(path),
            SourceLocation(page_number=page_number, paragraph_index=len(blocks), bbox=bbox), extra))
    return blocks, issues, figures, tuple(path), title


def parse_pdf(content, options, progress=None):
    import pymupdf
    from src.ingestion.parsers import parser_dependencies

    try:
        pdf = pymupdf.open(stream=content, filetype="pdf")
    except Exception:
        raise DocumentParseError("corrupt_pdf", "PDF 无法打开或已损坏") from None
    with pdf:
        if pdf.needs_pass:
            raise DocumentParseError("encrypted_pdf", "暂不支持需要密码的 PDF")
        if len(pdf) > options.max_pages:
            raise DocumentParseError("too_many_pages", "PDF 页数超过配置上限")
        blocks, issues, converters, path = [], [], {}, ()
        info, started = pdf.metadata or {}, time.monotonic()
        title = info.get("title", "")
        metadata = {"engine": "docling-rapidocr-v1", "dependencies": parser_dependencies(options, "pdf"),
            "reading_order": "docling_layout", "coordinate_system": "points_top_left",
            "ocr_enabled_pages": [], "text_layer_pages": [], "figures": [],
            "ocr_language_resolved": rapidocr_language(options.ocr_language),
            "formula_enrichment": options.docling_formula, "page_processing": []}
        for index in range(len(pdf)):
            if time.monotonic() - started > options.docling_timeout:
                for remaining in range(index, len(pdf)):
                    issues.append(ParseIssue("document_timeout", "文档处理预算耗尽，此页未解析", remaining + 1))
                break
            page = pdf[index]
            try:
                if page.rect.width * page.rect.height * (options.ocr_dpi / 72) ** 2 > 24000000:
                    issues.append(ParseIssue("page_too_large", "页面渲染像素超过上限，请降低 DPI 或拆分原文", index + 1))
                    continue
                text = page.get_text("text", sort=True, **({"use_ocr": False} if "use_ocr" in inspect.signature(pymupdf.utils.get_text).parameters else {}))
                usable = sum(char.isalnum() for char in text) >= options.min_text_chars and text.count("\ufffd") <= max(1, len(text) // 10)
                if usable:
                    metadata["text_layer_pages"].append(index + 1)
                if not usable and not options.ocr:
                    issues.append(ParseIssue("ocr_required", "此页缺少有效文本层，未启用 OCR", index + 1))
                force = options.ocr and not usable
                if force not in converters:
                    converters[force] = load_converter(options, force)
                converter, stream_type = converters[force]
                with pymupdf.open() as single:
                    single.insert_pdf(pdf, from_page=index, to_page=index)
                    stream = stream_type(name="page.pdf", stream=io.BytesIO(single.tobytes()))
                result = converter.convert(stream, raises_on_error=False, max_num_pages=1, max_file_size=options.max_bytes)
                state = str(getattr(getattr(result, "status", ""), "value", getattr(result, "status", ""))).lower()
                if state not in ("success", "partial_success"):
                    issues.append(ParseIssue("page_failed", "Docling 页面处理未完成，请检查模型文件及扫描质量", index + 1))
                    continue
                new, warnings, figures, path, detected_title = convert_page(result.document, index + 1, float(page.rect.height), path, len(blocks), options)
                blocks.extend(new)
                issues.extend(warnings)
                metadata["figures"].extend(figures)
                title = title or detected_title
                if options.ocr:
                    metadata["ocr_enabled_pages"].append(index + 1)
                metadata["page_processing"].append({"page_number": index + 1, "status": state,
                    "text_layer_usable": usable, "force_full_page_ocr": force})
                if state == "partial_success":
                    issues.append(ParseIssue("page_partial", "Docling 报告此页部分处理成功，请查看原文", index + 1))
                if not new:
                    issues.append(ParseIssue("ocr_empty" if options.ocr else "page_empty", "此页没有提取到可用内容，请查看原文", index + 1))
            except DocumentParseError:
                raise  # Missing dependencies and hard resource limits cannot be hidden as page failures.
            except Exception:
                issues.append(ParseIssue("page_failed", "页面处理失败，请检查模型文件和扫描质量", index + 1))
            finally:
                if progress:
                    progress(index + 1, len(pdf))
        return blocks, issues, metadata, len(pdf), title, info.get("author", "")
