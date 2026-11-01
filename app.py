
import os
import json
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 项目路径和环境变量
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(
    PROJECT_ROOT / ".env",
    override=False
)

from scripts.demo_agent_live_dedup import (
    ResearchAgent,
    MODEL,
    MAX_TOTAL_TOOL_CALLS
)

from src.retrieval.validation_paper_scoped_retriever import (
    ValidationPaperScopedRetriever
)

from src.retrieval.uploaded_pdf_retriever import (
    MAX_PDF_BYTES,
    UploadedPDFRetriever
)

from src.retrieval.verified_v3_pdf_retriever import (
    VERIFIED_PDF_SHA256,
    VerifiedV3PDFRetriever
)

from src.retrieval.strategy_proxy import (
    RetrievalStrategyProxy
)


VERIFIED_MAPPING_PATH = (
    PROJECT_ROOT / "results" / "pdf_v3_section_mapping_report.json"
)


# ==========================================
# 2. 页面基础设置
# ==========================================

st.set_page_config(
    page_title="Research RAG Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Research RAG Agent")

st.caption(
    "Hybrid Retrieval + Anchor-First + "
    "DeepSeek Tool Calling"
)


# ==========================================
# 3. DeepSeek 客户端
# ==========================================

api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
if api_key == "your_deepseek_api_key_here":
    api_key = ""

if api_key and "deepseek_client" not in st.session_state:

    st.session_state["deepseek_client"] = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=1
    )

client = st.session_state.get("deepseek_client") if api_key else None
if client is None:
    st.info("尚未配置 DEEPSEEK_API_KEY：可以建立 PDF 索引，问答前请在 .env 中配置并重启应用。")


# ==========================================
# 4. 缓存模型及验证集检索器
# ==========================================

@st.cache_resource(
    show_spinner="正在加载 QASPER 检索器..."
)
def load_validation_retriever():

    return ValidationPaperScopedRetriever()


@st.cache_resource(
    show_spinner="正在加载 BGE Embedding 模型..."
)
def load_embedding_model():

    return SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        device="cpu",
        cache_folder=str(
            PROJECT_ROOT
            / "models"
            / "embedding_cache"
        )
    )


# ==========================================
# 5. 创建 Agent
# ==========================================

def create_agent(retriever, paper_id, is_pdf, strategy="hybrid"):

    # 只有验证过章节映射的 PDF 才允许启用章节重排序。
    # 将代理传给原 ResearchAgent，不改变 PaperSearchTool 的参数接口。
    if is_pdf:
        effective_retriever = RetrievalStrategyProxy(
            retriever=retriever,
            strategy=strategy
        )
    else:
        effective_retriever = retriever

    st.session_state["agent_retrieval_proxy"] = (
        effective_retriever
        if isinstance(effective_retriever, RetrievalStrategyProxy)
        else None
    )

    agent = ResearchAgent(
        client=client,
        retriever=effective_retriever,
        paper_id=paper_id
    )

    if is_pdf:
        if getattr(retriever, "has_verified_sections", False):
            agent.messages[0]["content"] += (
                " The uploaded document has verified section headings "
                "from an exact-match PDF-specific mapping. "
                "Quote only section titles present in retrieved passages. "
                "Always cite the retrieved passage citation numbers. "
                "PDF page numbers remain available."
            )
        else:
            agent.messages[0]["content"] += (
                " The uploaded document is indexed by page number. "
                "Page 1, Page 2 and similar labels are not verified "
                "section headings. For content questions, prefer "
                "search_research_paper. Do not invent section titles. "
                "When citing PDF text, use the retrieved passage "
                "citation numbers."
            )

    return agent


def reset_conversation():

    st.session_state["agent"] = create_agent(
        retriever=active_retriever,
        paper_id=active_paper_id,
        is_pdf=is_pdf_mode,
        strategy=selected_retrieval_strategy
    )

    st.session_state["chat_history"] = []


# ==========================================
# 6. 选择论文来源
# ==========================================

with st.sidebar:

    st.header("📖 论文管理")

    source_mode = st.radio(
        "选择论文来源",
        options=[
            "QASPER 验证集论文",
            "上传自己的 PDF"
        ],
        key="source_mode"
    )


# 以下变量会根据不同模式赋值
active_retriever = None
active_paper_id = None
active_title = ""
active_key = ""
selected_retrieval_strategy = "hybrid"
is_pdf_mode = (
    source_mode == "上传自己的 PDF"
)


# ==========================================
# 7. 模式一：QASPER 验证集论文
# ==========================================

if not is_pdf_mode:

    try:
        active_retriever = load_validation_retriever()
    except (FileNotFoundError, ValueError, RuntimeError, OSError):
        st.error("QASPER 索引或模型尚未准备好，或索引文件不一致。请按 README 构建，或在左侧切换到上传 PDF。")
        st.code("python scripts/prepare_validation_sample.py\npython scripts/build_validation_faiss.py", language="bash")
        st.stop()

    paper_ids = sorted(
        str(paper_id)
        for paper_id in active_retriever.paper_rows
    )

    if not paper_ids:

        st.error("QASPER 索引没有可用论文。")
        st.stop()

    paper_titles = {}

    for chunk in active_retriever.chunks:

        pid = str(chunk["paper_id"])

        if pid not in paper_titles:

            paper_titles[pid] = (
                chunk.get("title", "")
            )

    default_paper_id = (
        "1705.01214"
        if "1705.01214" in paper_ids
        else paper_ids[0]
    )

    with st.sidebar:

        st.caption(
            "当前使用 QASPER Validation 索引"
        )

        selected_paper_id = st.selectbox(
            "选择研究论文",
            options=paper_ids,
            index=paper_ids.index(
                default_paper_id
            ),
            format_func=lambda pid: (
                f"{pid} - "
                f"{paper_titles.get(pid, '')}"
            ),
            key="qasper_paper"
        )

    active_paper_id = selected_paper_id

    active_title = paper_titles.get(
        selected_paper_id,
        ""
    )

    active_key = (
        f"qasper:{active_paper_id}"
    )


# ==========================================
# 8. 模式二：用户上传 PDF
# ==========================================

else:

    with st.sidebar:

        uploaded_file = st.file_uploader(
            "上传自己的 PDF 论文",
            type=["pdf"],
            accept_multiple_files=False,
            key="uploaded_pdf_file"
        )

        st.caption(
            "最大 20 MB、80 页、1500 个 Chunk。"
        )

    if uploaded_file is None:

        st.info(
            "请在左侧上传 PDF，"
            "然后点击「建立检索索引」。"
        )

        st.stop()

    if uploaded_file.size > MAX_PDF_BYTES:
        st.error("PDF 文件不能超过 20 MB，请上传较小的文件。")
        st.stop()

    pdf_bytes = uploaded_file.getvalue()
    if not pdf_bytes.startswith(b"%PDF"):
        st.error("文件不是有效的 PDF 格式，请检查上传文件。")
        st.stop()

    pdf_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    # 检查当前上传文件是否已完成索引
    indexed_hash = st.session_state.get(
        "indexed_pdf_hash"
    )

    pdf_retriever = st.session_state.get(
        "uploaded_pdf_retriever"
    )

    # 只对已验证的 HMC PDF 使用固定 V3 章节映射。
    # 其他 PDF 继续使用原有 Page X 分块，避免伪造章节标题。
    use_verified_v3 = (
        pdf_hash == VERIFIED_PDF_SHA256
        and uploaded_file.name == "HandbookChapter5.pdf"
        and VERIFIED_MAPPING_PATH.is_file()
    )

    index_ready = (
        pdf_retriever is not None
        and indexed_hash == pdf_hash
        and bool(getattr(pdf_retriever, "has_verified_sections", False))
        == use_verified_v3
    )

    # --------------------------------------
    # 8.1 尚未建立索引
    # --------------------------------------

    if not index_ready:

        st.subheader(
            "📄 建立 PDF 检索索引"
        )

        st.write(
            "文件：",
            uploaded_file.name
        )

        st.write(
            "大小：",
            f"{len(pdf_bytes) / 1024 / 1024:.2f} MB"
        )

        if st.button(
            "开始解析 PDF 并建立索引",
            type="primary"
        ):

            try:

                with st.spinner(
                    "正在提取文本、生成向量并建立索引..."
                ):

                    model = load_embedding_model()

                    if use_verified_v3:
                        new_retriever = VerifiedV3PDFRetriever(
                            pdf_bytes=pdf_bytes,
                            filename=uploaded_file.name,
                            model=model,
                            mapping_path=VERIFIED_MAPPING_PATH
                        )
                    else:
                        new_retriever = UploadedPDFRetriever(
                            pdf_bytes=pdf_bytes,
                            filename=uploaded_file.name,
                            model=model
                        )

                # 只在构建成功后更新状态
                st.session_state[
                    "uploaded_pdf_retriever"
                ] = new_retriever

                st.session_state[
                    "indexed_pdf_hash"
                ] = pdf_hash

                st.success(
                    "PDF 检索索引构建成功！"
                )

                st.rerun()

            except Exception as exc:

                st.error(
                    f"PDF 处理失败：{exc}"
                )

        st.stop()

    # --------------------------------------
    # 8.2 PDF 已建立索引
    # --------------------------------------

    active_retriever = pdf_retriever

    active_paper_id = (
        pdf_retriever.paper_id
    )

    active_title = pdf_retriever.title

    active_key = f"pdf:{pdf_hash}"

    # 已验证 V3 才开放章节感知策略；旧 PDF 仍保持 Hybrid。
    with st.sidebar:
        st.divider()
        st.subheader("🔎 检索策略")
        if getattr(active_retriever, "has_verified_sections", False):
            selected_retrieval_strategy = st.radio(
                "当前策略",
                options=["hybrid", "section_aware"],
                index=0,
                format_func=lambda value: (
                    "Hybrid（默认）"
                    if value == "hybrid"
                    else "Section-Aware（实验性）"
                ),
                key="uploaded_pdf_retrieval_strategy"
            )
            st.caption("已验证的 V3 章节索引；切换策略会重置当前聊天。")
        else:
            selected_retrieval_strategy = "hybrid"
            if pdf_hash == VERIFIED_PDF_SHA256 and not VERIFIED_MAPPING_PATH.is_file():
                st.warning("未找到 results/pdf_v3_section_mapping_report.json，当前使用旧版 Hybrid。")
            else:
                st.caption("这份 PDF 尚无经过验证的章节映射，使用 Hybrid。")

    active_key += f":strategy={selected_retrieval_strategy}"


# ==========================================
# 9. 来源变化时，创建新的 Agent
# ==========================================

if (
    "agent" not in st.session_state
    or st.session_state.get("active_key")
    != active_key
):

    reset_conversation()

    st.session_state["active_key"] = (
        active_key
    )


agent = st.session_state["agent"]


# ==========================================
# 10. 当前论文信息
# ==========================================

paper_data = active_retriever.get_paper_data(
    active_paper_id
)

all_chunks = paper_data["chunks"]

chunk_lookup = {
    chunk["chunk_id"]: chunk
    for chunk in all_chunks
}

with st.sidebar:

    st.divider()

    st.subheader("当前论文")

    st.write(active_title)

    st.caption(
        f"Paper ID：{active_paper_id}"
    )

    st.caption(
        f"Chunk 数量：{len(all_chunks)}"
    )

    if is_pdf_mode:

        st.caption(
            f"PDF 页数："
            f"{active_retriever.page_count}"
        )

    st.divider()

    st.subheader("Agent 配置")

    st.caption(f"LLM：{MODEL}")

    st.caption(
        "Embedding：BGE-small-en-v1.5"
    )

    st.caption(
        "Retrieval："
        + (
            "Section-Aware（实验性，V3）"
            if selected_retrieval_strategy == "section_aware"
            else "BM25 + Dense + RRF（默认）"
        )
    )

    st.caption(
        "Expansion：Anchor-First"
    )

    st.caption(
        "每题最多执行 2 次工具调用"
    )

    st.divider()

    clear_clicked = st.button(
        "🧹 清空当前对话",
        use_container_width=True
    )

    if clear_clicked:

        reset_conversation()
        st.rerun()


# ==========================================
# 11. 整理 JSON 导出数据
# ==========================================

def prepare_export_data():

    records = []

    for index, record in enumerate(
        st.session_state["chat_history"],
        start=1
    ):

        result = record["result"]

        tool_records = []

        for call in result.get(
            "tool_calls", []
        ):

            tool_records.append({
                "tool_name": call.get(
                    "tool_name", ""
                ),
                "arguments": call.get(
                    "arguments", {}
                ),
                "new_chunks": call.get(
                    "new_chunks", 0
                ),
                "reused_chunks": call.get(
                    "reused_chunks", 0
                )
            })

        cited_sources = []

        for source in result.get(
            "cited_sources", []
        ):

            source_record = dict(source)

            chunk = chunk_lookup.get(
                source["chunk_id"]
            )

            if chunk is not None and is_pdf_mode:

                source_record["page_number"] = (
                    chunk.get("page_number")
                )

            cited_sources.append(
                source_record
            )

        records.append({
            "turn": index,
            "question": record["question"],
            "answer": result["answer"],
            "retrieval_strategy": result.get(
                "retrieval_strategy", selected_retrieval_strategy
            ),
            "strategy_search_calls": result.get("strategy_search_calls", 0),
            "tool_calls": tool_records,
            "cited_numbers": result.get(
                "cited_numbers", []
            ),
            "cited_sources": cited_sources,
            "invalid_citations": result.get(
                "invalid_citations", []
            )
        })

    return {
        "project": "Research RAG Agent",
        "source_mode": source_mode,
        "model": MODEL,
        "paper_id": active_paper_id,
        "title": active_title,
        "retrieval_strategy": selected_retrieval_strategy,
        "verified_v3": bool(getattr(active_retriever, "has_verified_sections", False)),
        "conversation_count": len(records),
        "records": records
    }


# ==========================================
# 12. 数据导出区域
# ==========================================

export_data = prepare_export_data()

export_json = json.dumps(
    export_data,
    ensure_ascii=False,
    indent=2
).encode("utf-8")

with st.sidebar:

    st.subheader("📥 数据导出")

    st.download_button(
        label="下载聊天记录 JSON",
        data=export_json,
        file_name=(
            f"research_agent_"
            f"{active_paper_id}.json"
        ),
        mime="application/json",
        disabled=(
            export_data["conversation_count"] == 0
        ),
        use_container_width=True
    )

    st.caption(
        f"当前聊天记录："
        f"{export_data['conversation_count']} 轮"
    )

    if is_pdf_mode:

        st.caption(
            "上传文件和索引暂存于当前会话，"
            "不会自动保存到项目目录。"
        )


# ==========================================
# 13. 展示工具调用与引用原文
# ==========================================

def render_answer_details(result):

    tool_calls = result.get(
        "tool_calls", []
    )

    cited_sources = result.get(
        "cited_sources", []
    )

    st.caption(
        f"工具调用：{len(tool_calls)} 次"
        f" | 引用来源：{len(cited_sources)} 个"
    )
    if "retrieval_strategy" in result:
        st.caption(
            f"检索策略：{result['retrieval_strategy']}"
            f" | 检索代理调用：{result.get('strategy_search_calls', 0)} 次"
        )

    # --------------------------------------
    # 13.1 异常引用
    # --------------------------------------

    if result.get("invalid_citations"):

        st.warning(
            "存在无效引用编号："
            f"{result['invalid_citations']}"
        )

    if result.get("placeholder_found"):

        st.warning(
            "模型曾生成 [n] 占位符，"
            "已从显示回答中清理。"
        )

    if result.get(
        "search_answer_missing_citations"
    ):

        st.warning(
            "Agent 已检索原文，"
            "但回答没有数字引用。"
        )

    # --------------------------------------
    # 13.2 工具调用记录
    # --------------------------------------

    with st.expander(
        "🔧 查看 Agent 工具调用过程"
    ):

        for index, call in enumerate(
            tool_calls,
            start=1
        ):

            st.markdown(
                f"**第 {index} 次："
                f"{call['tool_name']}**"
            )

            st.json(
                call.get("arguments", {})
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "新增 Chunk",
                    call.get(
                        "new_chunks", 0
                    )
                )

            with col2:

                st.metric(
                    "复用 Chunk",
                    call.get(
                        "reused_chunks", 0
                    )
                )

            st.divider()

    # --------------------------------------
    # 13.3 引用原文
    # --------------------------------------

    if cited_sources:

        with st.expander(
            "📚 查看引用的论文原文"
        ):

            for source in cited_sources:

                number = source["citation"]

                chunk_id = source["chunk_id"]

                chunk = chunk_lookup.get(
                    chunk_id
                )

                st.markdown(
                    f"**引用 [{number}]**"
                )

                if chunk is None:

                    st.warning(
                        "没有找到对应 Chunk。"
                    )
                    continue

                if is_pdf_mode:

                    st.caption(
                        f"PDF 第 {chunk['page_number']} 页"
                    )
                    if getattr(active_retriever, "has_verified_sections", False):
                        st.caption(
                            "章节：" + chunk.get("section_name", "")
                        )

                else:

                    st.caption(
                        f"章节："
                        f"{chunk.get('section_name', '')}"
                    )

                st.caption(
                    f"Chunk ID：{chunk_id}"
                )

                st.write(
                    chunk["text"]
                )

                st.divider()


# ==========================================
# 14. 主界面文档信息
# ==========================================

if is_pdf_mode:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "PDF 页数",
            active_retriever.page_count
        )

    with col2:
        st.metric(
            "Chunk 数量",
            len(all_chunks)
        )

    with col3:
        st.metric(
            "FAISS 向量数",
            active_retriever.index.ntotal
        )

else:

    st.caption(
        "当前使用 QASPER Validation "
        "独立验证集论文。"
    )

st.caption(
    f"当前文档：{active_title}"
)

st.divider()


# ==========================================
# 15. 渲染已有聊天消息
# ==========================================

for record in st.session_state[
    "chat_history"
]:

    with st.chat_message("user"):

        st.markdown(
            record["question"]
        )

    with st.chat_message("assistant"):

        st.markdown(
            record["result"]["answer"]
        )

        render_answer_details(
            record["result"]
        )


# ==========================================
# 16. 无聊天记录时展示提示
# ==========================================

if not st.session_state[
    "chat_history"
]:

    st.info(
        "论文已经加载完成。"
        "你可以询问研究方法、实验设计、"
        "评估指标和核心结论。"
    )

    if is_pdf_mode:

        if getattr(active_retriever, "has_verified_sections", False):
            st.caption(
                "当前 PDF 使用已验证的 V3 章节结构；"
                "Section-Aware 为可选实验策略。"
            )
        else:
            st.caption(
                "当前 PDF 按页码组织内容，"
                "页码不等于真实章节标题。"
            )

        st.code(
            "What are the main concepts "
            "discussed in this document?\n"
            "Summarize the main ideas "
            "with citations.",
            language=None
        )

    else:

        st.code(
            "What are the main sections "
            "of this paper?\n"
            "What evaluation metrics are used?\n"
            "What methods are proposed?",
            language=None
        )


# ==========================================
# 17. 用户输入
# ==========================================

question = st.chat_input(
    "请输入你想了解的论文问题...",
    disabled=client is None,
    max_chars=2000
)


# ==========================================
# 18. 执行 Research Agent
# ==========================================

if question:

    if (
        agent.total_tool_calls
        >= MAX_TOTAL_TOOL_CALLS
    ):

        st.warning(
            "当前会话已达到工具调用上限。"
            "请清空当前对话后继续。"
        )

    else:

        try:

            with st.spinner(
                "Agent 正在检索论文并生成答案..."
            ):

                proxy = st.session_state.get("agent_retrieval_proxy")
                search_count_before = (
                    proxy.search_call_count if proxy is not None else 0
                )
                result = agent.ask(question)
                search_count_after = (
                    proxy.search_call_count if proxy is not None else 0
                )

            # 如果 Agent 调用了论文搜索工具，但没有经过策略代理，
            # 明确提示需要检查 ResearchRAGPipeline 的内部调用路径。
            search_tool_used = any(
                item.get("tool_name") == "search_research_paper"
                for item in result.get("tool_calls", [])
            )
            if (
                is_pdf_mode
                and search_tool_used
                and search_count_after == search_count_before
            ):
                st.warning(
                    "本轮 Agent 调用了论文搜索工具，但未经过检索策略代理。"
                    "请检查 ResearchRAGPipeline.prepare() 的搜索调用路径；"
                    "不能据此认定 Section-Aware 已生效。"
                )

            result["retrieval_strategy"] = selected_retrieval_strategy
            result["strategy_search_calls"] = (
                search_count_after - search_count_before
            )

            # 保存当前问答
            st.session_state[
                "chat_history"
            ].append({
                "question": question,
                "result": result
            })

            # 刷新页面，显示最新回答
            # 并同步更新 JSON 下载按钮
            st.rerun()

        except Exception as exc:

            st.error(
                f"Agent 执行失败：{exc}"
            )

            # 避免不完整工具消息影响下次调用
            reset_conversation()

            st.warning(
                "当前 Agent 会话已重置。"
                "论文索引仍然保留。"
            )
