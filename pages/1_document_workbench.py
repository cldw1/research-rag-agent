"""Document library, structured parsing, chunk previews and persistent indexes."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.document import DocumentParseError
from src.ingestion.parsers import ParseOptions
from src.ingestion.structured_chunker import ChunkConfig, StructuredChunker
from src.storage.artifacts import StorageError
from src.storage.document_store import LocalDocumentStore, default_data_root

load_dotenv(ROOT / ".env", override=False)


@st.cache_resource
def get_store(root):
    return LocalDocumentStore(root)


@st.cache_resource
def get_embedding():
    from src.embedding import load_embedding_resources
    return load_embedding_resources()


st.set_page_config(page_title="Research RAG Agent · 文档工作台", page_icon="📚", layout="wide")
st.title("科研文档工作台")
st.caption("上传和管理科研文献，检查分块与原文来源，保存并恢复文档索引。")
try:
    store = get_store(str(default_data_root().resolve()))
except Exception:
    st.error("文档库无法打开，请检查 SQLAlchemy 依赖、目录权限及数据库结构版本。")
    st.stop()

with st.sidebar:
    st.header("文档与分块")
    uploads = st.file_uploader("上传文献", type=["pdf", "docx", "md", "txt"], accept_multiple_files=True)
    backend = st.selectbox("PDF 解析器", ["lightweight", "docling"],
        format_func=lambda value: "轻量文本解析 / Tesseract" if value == "lightweight" else "Docling 版面识别 / RapidOCR")
    ocr = st.checkbox("为缺少有效文本的 PDF 页面启用 OCR", value=backend == "docling", key="ocr_" + backend)
    language = st.selectbox("OCR 语言", ["eng", "eng+chi_sim", "chi_sim"], disabled=not ocr)
    tessdata = st.text_input("Tesseract tessdata 路径（可选）", disabled=not ocr or backend == "docling")
    artifacts, formula, parse_timeout = None, False, 120
    if backend == "docling":
        artifacts = st.text_input("Docling 本地模型目录（可选）") or None
        formula = st.checkbox("启用公式识别（额外模型，处理较慢）", value=False)
        parse_timeout = st.number_input("文档处理时限（秒）", min_value=10, max_value=600, value=120, step=10)
        st.caption("PDF 将识别版面和表格；扫描及图片区域使用 RapidOCR。首次使用需准备模型文件，其他格式沿用现有解析器。")
    encoding = st.selectbox("文本编码", ["auto", "utf-8-sig", "utf-16", "gb18030"])
    strategy = st.selectbox("分块策略", ["adaptive", "hybrid", "structure", "recursive", "fixed"])
    budget = st.number_input("分块预算", min_value=16, max_value=480, value=480, step=16)
    chars = st.number_input("字符上限", min_value=64, max_value=4000, value=1200, step=64)
    overlap = st.number_input("重叠字符数", min_value=0, max_value=min(256, int(chars) - 1), value=0, disabled=strategy == "hybrid")
    if strategy == "hybrid":
        overlap = 0
        st.caption("Hybrid 按文档结构和 Embedding 模型的 Token 预算分块；预览需加载本地 Embedding。来源定位到原始结构块。")
    else:
        st.caption("预览使用 UTF-8 字节预算；构建 BGE 索引时以模型真实 Tokenizer 重新分块。")
    parse_clicked = st.button("保存并解析上传文档", type="primary", disabled=not uploads)
    st.divider()
    search_name = st.text_input("搜索已保存文档")
    filter_type = st.selectbox("文档类型", ["全部", "pdf", "docx", "md", "txt"])
    show_deleted = st.checkbox("显示已移出文档库的文件")

try:
    options = ParseOptions(ocr=ocr, ocr_language=language, tessdata=tessdata.strip() or None,
                           text_encoding=encoding, backend=backend,
                           docling_artifacts_path=artifacts, docling_formula=formula,
                           docling_timeout=int(parse_timeout))
except ValueError as exc:
    st.error(str(exc))
    st.stop()
if parse_clicked:
    progress = st.progress(0.0, text="开始解析")
    for index, upload in enumerate(uploads):
        try:
            if upload.size > options.max_bytes:
                raise DocumentParseError("file_too_large", "文件超过 20 MB 上限")
            content = upload.getvalue()
            result = store.ingest(content, upload.name, options, progress=lambda done, total:
                progress.progress((index + done / total) / len(uploads), text=f"{upload.name} · {done}/{total} 页"))
            if result.reused:
                st.info(f"{upload.name}：相同内容和解析配置已保存，已复用结果。")
            else:
                st.success(f"{upload.name}：已保存解析结果。")
        except (DocumentParseError, StorageError) as exc:
            st.error(f"{upload.name} · {exc.code}：{exc}")
        except Exception:
            st.error(f"{upload.name}：解析未完成，请检查文件或配置。")
        progress.progress((index + 1) / len(uploads), text=f"已处理 {index + 1}/{len(uploads)} 个文件")

with st.expander("最近的处理任务与恢复"):
    st.dataframe(store.list_jobs(), hide_index=True)
    st.caption("处理任务当前在前台同步执行；进程退出后可恢复过期的任务状态，再重新提交处理。")
    if st.button("标记已超时的中断任务"):
        st.info(f"已恢复 {store.recover_expired_jobs()} 个过期任务的状态。")

records = [record for record in store.list_documents(include_deleted=show_deleted)
           if search_name.lower() in record["filename"].lower()
           and (filter_type == "全部" or record["file_type"] == filter_type)]
if not records:
    st.info("文档库中暂无匹配文件。上传 PDF、Word、Markdown 或 TXT 后点击“保存并解析上传文档”。无需 API Key。")
    st.stop()

catalog = {record["document_id"]: record for record in records}
status_labels = {"ready": "解析完成", "partial": "部分解析", "failed": "解析失败", "processing": "处理中", "deleted": "已移出"}
selected = st.selectbox("当前文档", list(catalog), format_func=lambda key:
    f"{catalog[key]['filename']} · {status_labels.get(catalog[key]['status'], catalog[key]['status'])} · {catalog[key]['chunk_count']} 个已索引 Chunk")
record = catalog[selected]
if record["deleted_at"]:
    st.info("文档已移出检索范围，原始文件及历史版本保留。恢复后可重新激活或构建索引。")
    if st.button("恢复到文档库"):
        store.restore_document(selected)
        st.rerun()
    st.stop()
actions = st.columns(2)
with actions[0]:
    if st.button("按当前解析设置重新解析"):
        try:
            store.reparse(selected, options)
            st.rerun()
        except (DocumentParseError, StorageError) as exc:
            st.error(f"{exc.code}：{exc}")
        except Exception:
            st.error("重新解析未完成，已有可用版本保留。")
with actions[1]:
    if st.button("移出文档库（可恢复）"):
        store.delete_document(selected)
        st.session_state.pop("v2_workbench_index", None)
        st.rerun()
try:
    document = store.get_document(selected)
except StorageError as exc:
    st.error(f"{exc.code}：{exc}")
    st.stop()
config = ChunkConfig(strategy=strategy, max_tokens=int(budget), max_chars=int(chars), overlap_chars=int(overlap))
try:
    preview_counter = None
    if strategy == "hybrid":
        with st.spinner("加载分块所需的本地 Tokenizer…"):
            _, preview_counter, _ = get_embedding()
    chunks = StructuredChunker(config, preview_counter).chunk(document)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

left, right = st.columns([3, 2])
with left:
    st.subheader(document.title)
    st.caption(f"{document.file_type.upper()} · {document.file_size / 1024:.1f} KB · {len(document.blocks)} 个结构块 · {len(chunks)} 个预览 Chunk")
    if document.file_type == "pdf":
        layout = document.source_metadata.get("engine") == "docling-rapidocr-v1"
        st.caption(f"真实 PDF 页数：{document.page_count}。" + ("已使用 Docling 版面识别，复杂表格和公式请结合原文复核。" if layout else "按坐标排序，尚未重建双栏、表格或公式结构。"))
    if document.file_type == "docx":
        st.caption("Word 使用 OOXML 基础解析，按段落定位；不提供原始页码，不执行嵌入对象。")
    if document.issues:
        st.warning("文档存在解析提示，请复核后使用。")
        st.dataframe([asdict(issue) for issue in document.issues], hide_index=True)
    tab_blocks, tab_chunks = st.tabs(["解析结果", "分块与来源"])
    with tab_blocks:
        st.dataframe([{"类型": b.kind, "章节": " / ".join(b.section_path),
                       "页码": b.location.page_number, "段落": b.location.paragraph_index,
                       "文本预览": b.text[:180]} for b in document.blocks], hide_index=True)
    with tab_chunks:
        number = st.number_input("查看 Chunk", min_value=1, max_value=len(chunks), value=1)
        chunk = chunks[int(number) - 1]
        st.code(chunk["text"], language=None)
        st.json({k: v for k, v in chunk.items() if k != "text"}, expanded=False)
        if any(c["source_metadata"]["atomic_split_required"] for c in chunks):
            st.warning("部分表格、公式或代码块超出预算，已拆分并保留来源范围；请查看原文。")
    st.download_button("导出解析与预览分块 JSON", json.dumps({
        "document": document.to_dict(), "config": asdict(config), "chunks": chunks,
    }, ensure_ascii=False, indent=2), file_name="research_document_preview.json", mime="application/json")

with right:
    st.subheader("检索证据预览")
    st.caption("沿用 BGE-small-en-v1.5、BM25、FAISS 与 RRF。英文 Embedding 的中文检索效果尚未验证。")
    info = store.active_index_info(selected)
    if info and (info["parse_id"] != document.document_version or info["chunk_config"] != asdict(config)):
        st.info("当前预览设置与已保存索引不同。现有索引仍可检索；构建成功后切换为新版本。")
    force = st.checkbox("强制重新编码（忽略可复用索引）", value=False)
    if st.button("构建或复用持久化索引"):
        try:
            with st.spinner("加载 Embedding 并构建索引…"):
                model, counter, profile = get_embedding()
                result = store.build_index(selected, model, profile, counter, config, force=force)
                retriever = store.load_retriever(selected, model, profile)
                st.session_state["v2_workbench_index"] = (selected, result.index_id, retriever)
                info = store.active_index_info(selected)
                st.success(f"{'复用' if result.reused else '保存'}索引完成：{result.chunk_count} 个 Chunk。")
        except StorageError as exc:
            st.error(f"{exc.code}：{exc}")
        except Exception:
            st.error("索引构建失败，请检查本地 BGE 缓存、FAISS 环境和分块预算。")
    if info and st.button("恢复已保存索引（不重复编码文档）"):
        try:
            with st.spinner("校验文件并恢复索引…"):
                model, counter, profile = get_embedding()
                retriever = store.load_retriever(selected, model, profile)
                st.session_state["v2_workbench_index"] = (selected, info["index_id"], retriever)
                st.success("索引已恢复，文档没有重新解析或编码。")
        except StorageError as exc:
            st.error(f"{exc.code}：{exc}")
        except Exception:
            st.error("索引恢复未完成，请检查本地 Embedding 模型及文件状态。")
    built = st.session_state.get("v2_workbench_index")
    if built and info and built[:2] == (selected, info["index_id"]):
        method = st.selectbox("检索方法", ["hybrid", "bm25", "dense"])
        query = st.text_input("输入文献检索问题")
        if st.button("检索原文证据", disabled=not query.strip()):
            try:
                results = built[2].search(query, document.document_id, method=method)
                for item in results:
                    with st.expander(f"#{item['rank']} · {item['section_name']} · 段落 {item['paragraph_index']}", expanded=item["rank"] == 1):
                        st.code(item["text"], language=None)
                        st.json(item["source_metadata"], expanded=False)
            except StorageError as exc:
                st.error(f"{exc.code}：{exc}")
            except Exception:
                st.error("检索未完成，请重新构建当前文档索引后重试。")
    else:
        st.info("构建索引或恢复已保存索引后，可比较 BM25、Dense 和 Hybrid。无需重新上传文档。")
