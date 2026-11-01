
import hashlib
import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 项目路径
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(
    PROJECT_ROOT / ".env",
    override=False
)

from src.retrieval.uploaded_pdf_retriever import (
    UploadedPDFRetriever
)

from scripts.demo_agent_live_dedup import (
    ResearchAgent,
    MODEL,
    MAX_TOTAL_TOOL_CALLS
)


# ==========================================
# 2. 页面设置
# ==========================================

st.set_page_config(
    page_title="PDF Research Agent",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Research Agent")

st.caption(
    "上传自己的 PDF，使用 Hybrid Retrieval "
    "+ Anchor-First + DeepSeek Agent 进行问答"
)


# ==========================================
# 3. 加载 DeepSeek 配置
# ==========================================

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    st.error(
        "没有找到 DEEPSEEK_API_KEY，"
        "请检查项目根目录的 .env。"
    )
    st.stop()


# ==========================================
# 4. 缓存 BGE Embedding 模型
# ==========================================

@st.cache_resource
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
# 5. 初始化 DeepSeek 客户端
# ==========================================

if "pdf_client" not in st.session_state:

    st.session_state["pdf_client"] = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

client = st.session_state["pdf_client"]


# ==========================================
# 6. 创建新 Agent 会话
# ==========================================

def create_pdf_agent(retriever):

    agent = ResearchAgent(
        client=client,
        retriever=retriever,
        paper_id=retriever.paper_id
    )

    # 上传 PDF 按页组织，不是真正识别章节
    agent.messages[0]["content"] += (
        " This uploaded PDF is indexed by page. "
        "Its Page 1, Page 2 labels are not "
        "verified section headings. "
        "Do not present page labels as actual "
        "section titles. "
        "For questions about content, use "
        "search_research_paper. "
        "If asked about the document structure, "
        "do not invent section headings."
    )

    return agent


def reset_chat(retriever):

    st.session_state["pdf_agent"] = (
        create_pdf_agent(retriever)
    )

    st.session_state["pdf_chat_history"] = []


# ==========================================
# 7. 侧边栏 PDF 上传
# ==========================================

with st.sidebar:

    st.header("📄 PDF 论文管理")

    uploaded_file = st.file_uploader(
        "上传自己的 PDF 论文",
        type=["pdf"],
        accept_multiple_files=False
    )

    st.caption(
        "当前支持文本型 PDF，最大 20 MB、"
        "80 页、1500 个 Chunk。"
    )

    st.divider()

    st.markdown("**Agent 配置**")

    st.caption(f"LLM：{MODEL}")

    st.caption(
        "Embedding：BGE-small-en-v1.5"
    )

    st.caption(
        "Retrieval：BM25 + Dense + RRF"
    )

    st.caption(
        "Expansion：Anchor-First"
    )

    st.caption(
        "每道问题最多执行 2 次工具调用"
    )


# ==========================================
# 8. 检查上传文件
# ==========================================

if uploaded_file is None:

    st.info(
        "请在左侧上传 PDF，"
        "然后建立检索索引。"
    )

    st.stop()


pdf_bytes = uploaded_file.getvalue()

current_hash = hashlib.sha256(
    pdf_bytes
).hexdigest()


# ==========================================
# 9. 判断当前 PDF 是否已经建立索引
# ==========================================

indexed_hash = st.session_state.get(
    "indexed_pdf_hash"
)

retriever = st.session_state.get(
    "uploaded_pdf_retriever"
)

index_ready = (
    retriever is not None
    and indexed_hash == current_hash
)


if not index_ready:

    st.subheader("第一步：建立 PDF 检索索引")

    st.write(
        "文件名称：",
        uploaded_file.name
    )

    st.write(
        "文件大小：",
        f"{len(pdf_bytes) / 1024 / 1024:.2f} MB"
    )

    if st.button(
        "开始解析 PDF 并建立索引",
        type="primary"
    ):

        try:

            with st.spinner(
                "正在解析 PDF、生成向量并建立索引..."
            ):

                model = load_embedding_model()

                new_retriever = UploadedPDFRetriever(
                    pdf_bytes=pdf_bytes,
                    filename=uploaded_file.name,
                    model=model
                )

                # 只在整个构建成功后更新状态
                st.session_state[
                    "uploaded_pdf_retriever"
                ] = new_retriever

                st.session_state[
                    "indexed_pdf_hash"
                ] = current_hash

                reset_chat(new_retriever)

            st.success(
                "PDF 已完成索引构建！"
            )

            st.rerun()

        except Exception as exc:

            st.error(
                "PDF 处理失败："
                + str(exc)
            )

    st.stop()


# ==========================================
# 10. 当前 PDF 已经建立索引
# ==========================================

retriever = st.session_state[
    "uploaded_pdf_retriever"
]

if "pdf_agent" not in st.session_state:

    reset_chat(retriever)

agent = st.session_state["pdf_agent"]


# ==========================================
# 11. 侧边栏：论文信息与重置
# ==========================================

with st.sidebar:

    st.subheader("当前论文")

    st.write(retriever.title)

    st.caption(
        f"论文 ID：{retriever.paper_id}"
    )

    st.caption(
        f"页数：{retriever.page_count}"
    )

    st.caption(
        f"Chunk：{len(retriever.chunks)}"
    )

    st.divider()

    if st.button(
        "🧹 清空对话",
        use_container_width=True
    ):

        reset_chat(retriever)
        st.rerun()


# ==========================================
# 12. 主界面显示 PDF 索引信息
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "PDF 页数",
        retriever.page_count
    )

with col2:
    st.metric(
        "Chunk 数量",
        len(retriever.chunks)
    )

with col3:
    st.metric(
        "FAISS 向量",
        retriever.index.ntotal
    )


st.caption(
    f"当前文档：{retriever.title}"
)

st.divider()


# ==========================================
# 13. 显示引用与工具调用信息
# ==========================================

def render_details(result):

    calls = result.get(
        "tool_calls", []
    )

    sources = result.get(
        "cited_sources", []
    )

    st.caption(
        f"工具调用：{len(calls)} 次"
        f" | 引用：{len(sources)} 个"
    )

    if result.get("invalid_citations"):

        st.warning(
            "存在无效引用："
            + str(result["invalid_citations"])
        )

    if result.get(
        "search_answer_missing_citations"
    ):

        st.warning(
            "本次进行了检索，但回答没有数字引用。"
        )

    # --------------------------------------
    # Agent 调用过程
    # --------------------------------------

    with st.expander(
        "🔧 查看 Agent 工具调用过程"
    ):

        for index, call in enumerate(
            calls,
            start=1
        ):

            st.markdown(
                f"**第 {index} 次："
                f"{call['tool_name']}**"
            )

            st.json(
                call.get("arguments", {})
            )

            st.write(
                "新增 Chunk：",
                call.get("new_chunks", 0)
            )

            st.write(
                "复用 Chunk：",
                call.get("reused_chunks", 0)
            )

            st.divider()

    # --------------------------------------
    # 引用原文
    # --------------------------------------

    if sources:

        with st.expander(
            "📚 查看引用的 PDF 原文"
        ):

            chunk_lookup = {
                chunk["chunk_id"]: chunk
                for chunk in retriever.chunks
            }

            for source in sources:

                number = source["citation"]
                chunk_id = source["chunk_id"]

                st.markdown(
                    f"**引用 [{number}]**"
                )

                chunk = chunk_lookup.get(
                    chunk_id
                )

                if chunk is None:

                    st.warning(
                        "没有找到对应的原文。"
                    )

                    continue

                st.caption(
                    f"PDF 第 {chunk['page_number']} 页"
                )

                st.caption(
                    f"Chunk ID：{chunk_id}"
                )

                st.write(
                    chunk["text"]
                )

                st.divider()


# ==========================================
# 14. 渲染聊天历史
# ==========================================

for record in st.session_state[
    "pdf_chat_history"
]:

    with st.chat_message("user"):

        st.markdown(
            record["question"]
        )

    with st.chat_message("assistant"):

        st.markdown(
            record["result"]["answer"]
        )

        render_details(
            record["result"]
        )


# ==========================================
# 15. 没有聊天记录时显示提示
# ==========================================

if not st.session_state["pdf_chat_history"]:

    st.info(
        "你的 PDF 已经完成检索索引构建，"
        "现在可以直接向 Research Agent 提问。"
    )

    st.markdown("**示例问题：**")

    st.code(
        "What is the main topic of this document?\n"
        "What methods are discussed in this chapter?\n"
        "Summarize the main ideas with citations.",
        language=None
    )

    st.caption(
        "注意：PDF 的 Page 标签是页码，"
        "不代表自动识别的章节标题。"
    )


# ==========================================
# 16. JSON 导出
# ==========================================

export_records = []

for record in st.session_state[
    "pdf_chat_history"
]:

    result = record["result"]

    export_records.append({
        "question": record["question"],
        "answer": result["answer"],
        "tool_calls": [
            {
                "tool_name": call["tool_name"],
                "arguments": call.get(
                    "arguments", {}
                ),
                "new_chunks": call.get(
                    "new_chunks", 0
                ),
                "reused_chunks": call.get(
                    "reused_chunks", 0
                )
            }
            for call in result.get(
                "tool_calls", []
            )
        ],
        "cited_numbers": result.get(
            "cited_numbers", []
        ),
        "cited_sources": result.get(
            "cited_sources", []
        ),
        "invalid_citations": result.get(
            "invalid_citations", []
        )
    })


with st.sidebar:

    st.divider()

    st.subheader("📥 数据导出")

    export_json = json.dumps(
        {
            "document": retriever.title,
            "paper_id": retriever.paper_id,
            "model": MODEL,
            "records": export_records
        },
        ensure_ascii=False,
        indent=2
    ).encode("utf-8")

    st.download_button(
        label="下载 PDF 问答记录 JSON",
        data=export_json,
        file_name=(
            f"{retriever.paper_id}_chat.json"
        ),
        mime="application/json",
        disabled=(len(export_records) == 0),
        use_container_width=True
    )


# ==========================================
# 17. 用户输入
# ==========================================

question = st.chat_input(
    "向上传的 PDF 提问..."
)


# ==========================================
# 18. 执行真正的 PDF Research Agent
# ==========================================

if question:

    if (
        agent.total_tool_calls
        >= MAX_TOTAL_TOOL_CALLS
    ):

        st.warning(
            "当前会话已达到工具调用上限，"
            "请清空对话后继续。"
        )

    else:

        try:

            with st.spinner(
                "Agent 正在检索 PDF 并生成答案..."
            ):

                result = agent.ask(question)

            st.session_state[
                "pdf_chat_history"
            ].append({
                "question": question,
                "result": result
            })

            st.rerun()

        except Exception as exc:

            st.error(
                "Agent 执行失败："
                + str(exc)
            )

            # 出错时避免保留不完整的工具消息。
            reset_chat(retriever)

            st.warning(
                "Agent 会话已重置，"
                "但上传 PDF 的检索索引仍然保留。"
            )
