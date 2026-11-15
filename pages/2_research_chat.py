"""Persistent chat on the existing versioned document library."""

import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from src.chat_service import ResearchChatService
from src.llm.provider import ModelConfig, ModelError
from src.llm.factory import CATALOG, create_provider
from src.generation_task import GenerationPool
from src.storage.artifacts import StorageError
from src.storage.conversations import ConversationStore
from src.storage.database import catalog_version, upgrade_chat_schema, upgrade_multidoc_schema
from src.storage.document_store import LocalDocumentStore, default_data_root


st.set_page_config(page_title="Research RAG Agent · 科研对话", page_icon="💬", layout="wide")
st.title("科研文献对话")
st.caption("围绕已索引文献提问；历史对话和引用来源长期保存。API Key 仅保留在当前服务内存/会话。")


@st.cache_resource
def load_store(root):
    return LocalDocumentStore(root)


@st.cache_resource
def embedding_resources():
    from src.embedding import load_embedding_resources
    return load_embedding_resources()


@st.cache_resource
def generation_pool():
    return GenerationPool(max_workers=2)


try:
    documents = load_store(str(default_data_root().resolve()))
except Exception:
    st.error("文档库无法打开，请检查依赖、目录权限及数据库结构。")
    st.stop()
if catalog_version(documents.engine) == 1:
    st.info("已有文档库需要新增聊天表。升级前将完整备份，文档和索引保留。")
    if st.button("备份并启用聊天存储", type="primary"):
        try:
            with st.spinner("备份文档与索引并升级…"):
                upgrade_chat_schema(documents)
            st.rerun()
        except Exception:
            st.error("升级未完成；请检查备份目录权限和原文件/索引完整性。现有文档数据未删除。")
    st.stop()
conversations = ConversationStore(documents)
schema_version = catalog_version(documents.engine)
if schema_version == 2:
    st.info("当前文档库可继续单文档对话。启用多文档会话前将完整备份，原会话与引用保留。")
    if st.button("备份并启用多文档会话"):
        try:
            with st.spinner("备份并扩展会话文档关联…"):
                upgrade_multidoc_schema(documents)
            st.rerun()
        except Exception:
            st.error("多文档升级未完成，请检查备份与索引文件；可继续使用现有单文档会话。")

with st.sidebar:
    st.header("生成模型")
    provider_name = st.selectbox("供应商", list(CATALOG))
    preset = CATALOG[provider_name]
    if provider_name == "DeepSeek":
        model = st.selectbox("模型", ["deepseek-flash", "deepseek-v4-pro"])
    else:
        model = st.text_input("模型名称", value=preset["model"], key="model_" + preset["provider"])
    base_url = st.text_input("Base URL", value=preset["base_url"], key="url_" + preset["provider"],
                             disabled=preset["provider"] not in ("openai_compatible", "bailian"))
    if preset["provider"] == "openai_compatible":
        tools = st.checkbox("服务支持 Tool Calling", value=False)
        st.caption("未勾选时采用普通 RAG。此入口只适用于兼容 Chat Completions 的服务。")
    else:
        tools = True
    if preset["provider"] == "bailian":
        st.caption("Base URL、Key 和模型权限必须匹配同一地域/工作空间。")
    key = st.text_input("API Key", value=os.getenv(preset["key_env"], "") if preset["key_env"] else "",
                        type="password", key=f"generation_key_{provider_name}")
    if key == "your_deepseek_api_key_here":
        key = ""
    st.caption("提问时会把选中的历史和检索证据发送至该模型服务；切换服务后使用当前填写的 Key。")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.2, 0.1, disabled=preset["provider"] in ("openai", "anthropic"))
    if preset["provider"] in ("openai", "anthropic"):
        st.caption("此原生适配模式不发送温度参数，避免与推理模型参数约束冲突。")
    streaming = st.checkbox("流式输出", value=True)
    output = st.number_input("最大输出 Token", min_value=16, max_value=32768, value=1500)
    context = st.number_input("应用上下文预算", min_value=4096, max_value=1048576, value=32768, step=1024)
    st.caption("当前按 UTF-8 序列化字节估算输入，包含工具和证据；会额外预留输出及安全余量。")
    config = None
    try:
        config = ModelConfig(provider=preset["provider"],
            model=model, base_url=base_url, temperature=temperature, max_output_tokens=int(output),
            context_window=int(context), supports_tools=tools, stream=streaming)
    except ModelError as exc:
        st.warning(str(exc))

    st.divider()
    indexed = {item["document_id"]: item for item in documents.list_documents() if item["index_id"]}
    selected_documents = st.multiselect("新会话使用的文档（最多 5 篇）", list(indexed),
        default=list(indexed)[:1], max_selections=5 if schema_version >= 3 else 1,
        format_func=lambda value: indexed[value]["filename"] + " · " + value[-8:]) if indexed else []
    if st.button("新建会话", disabled=not selected_documents or config is None):
        st.session_state["research_conversation_id"] = conversations.create(selected_documents, config)
        st.rerun()
    st.caption("请先在文档工作台上传并建立索引；从原应用的页面导航可进入工作台。")

    saved = {item["id"]: item for item in conversations.list()}
    selected_id = st.selectbox("历史会话", list(saved), format_func=lambda value: saved[value]["title"],
                              key="research_conversation_id") if saved else None
    if st.button("恢复过期生成任务的状态"):
        conversations.recover_expired()
        st.rerun()

if not selected_id:
    st.info("先在工作台保存并索引文档，再在左侧新建会话。浏览历史无需 API Key。")
    st.stop()
details = conversations.details(selected_id)
selected_names = {item["document_id"]: item["filename"] for item in details["documents"]}
st.caption("当前会话文档范围：" + "、".join(selected_names.values()))
if len(selected_names) > 1:
    st.caption("多文档检索使用统一排名并按论文分配证据；比较结论请结合各篇原文引用。")
task = st.session_state.get("research_generation_task")
busy = task is not None and task.snapshot()["status"] == "running"
notice = st.session_state.pop("research_generation_notice", None)
if notice:
    st.info(notice)
try:
    conversations.details(selected_id, require_active=True)
    scope_active = True
except StorageError:
    scope_active = False
    st.warning("此会话绑定的索引已不再活动。可查看原有历史；继续提问请基于当前索引新建会话。")

with st.sidebar:
    with st.form("rename_" + selected_id):
        title = st.text_input("会话名称", value=details["title"])
        if st.form_submit_button("保存名称"):
            try:
                conversations.rename(selected_id, title)
                st.rerun()
            except StorageError as exc:
                st.error(str(exc))
    if st.button("删除此会话"):
        conversations.delete(selected_id)
        st.session_state.pop("research_conversation_id", None)
        st.rerun()
    st.download_button("导出 Markdown", conversations.export_markdown(selected_id),
                       file_name="research_conversation.md", mime="text/markdown")

all_sources = {source["citation"]: source for source in conversations.sources(selected_id)}
history = conversations.history(selected_id)
for turn in history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn["status"] != "completed":
            st.info(f"本轮尚无已保存回答（{turn['status']}）。失败或中断的工具序列不会进入后续模型历史。")
            continue
        result = turn["result"]
        st.markdown(result["answer"].replace("![", "\\!["))  # No automatic external image fetch from generated text.
        if result.get("invalid_citations") or result.get("missing_citations"):
            st.warning("本轮存在缺失或未注册的引用，请核对原文。编号有效也不等于结论已经通过事实核验。")
        if result.get("truncated"):
            st.warning("模型因输出上限停止，回答可能不完整。")
        if len(selected_names) > 1 and result.get("missing_document_ids"):
            missing = "、".join(selected_names.get(did, did) for did in result["missing_document_ids"])
            st.warning("本次工具证据尚未覆盖：" + missing + "。涉及这些文档的结论需补充取证。")
        for number in result["cited_numbers"]:
            source = all_sources.get(number)
            if source:
                location = f"PDF 第 {source['page_number']} 页" if source.get("page_number") else source.get("section", "原文段落")
                filename = selected_names.get(source["paper_id"], source.get("title", ""))
                with st.expander(f"[{number}] {filename} · {source.get('title', '')} · {location}"):
                    st.code(source["text"], language=None)
                    st.json(source.get("source_metadata", {}), expanded=False)
        with st.expander("本轮上下文和用量"):
            st.json({"context": result["context_selections"], "usage": result["request_usage"],
                     "model_calls": result["model_calls"], "tool_calls": result["tool_calls"]}, expanded=False)

if not key:
    st.info("填写当前服务的 API Key 后可继续提问；已有历史仍可查看和导出。")


@st.fragment(run_every=0.5)
def show_generation():
    active = st.session_state.get("research_generation_task")
    if active is None:
        return
    snapshot = active.snapshot()
    if snapshot["status"] == "running":
        st.caption(snapshot["phase"])
        if active.conversation_id == selected_id and snapshot["text"]:
            st.markdown(snapshot["text"].replace("![", "\\![") + " ▌")
        elif active.conversation_id != selected_id:
            st.info("另一会话仍在生成；可返回该会话查看或停止。")
        if st.button("停止生成", key="stop_generation"):
            active.cancel()
            st.rerun()
    else:
        message = "回答已保存。" if snapshot["status"] == "completed" else snapshot["error"] or snapshot["phase"]
        st.session_state["research_generation_notice"] = message
        st.session_state.pop("research_generation_task", None)
        st.rerun()


if task is not None:
    show_generation()

enabled = bool(key and config is not None and scope_active and not busy)
latest_completed = next((turn for turn in reversed(history) if turn["status"] == "completed"), None)
latest_failed = next((turn for turn in reversed(history) if turn["status"] in ("failed", "cancelled", "interrupted")), None)
retry_question = None
retry_columns = st.columns(2)
with retry_columns[0]:
    if st.button("重新生成最近回答", disabled=not enabled or latest_completed is None):
        retry_question = latest_completed["question"]
with retry_columns[1]:
    if st.button("重试最近未完成问题", disabled=not enabled or latest_failed is None):
        retry_question = latest_failed["question"]
st.caption("重新生成保留原回答并追加新回合；重试使用当前模型设置和新的请求 ID。")
question = st.chat_input("向选中的文献提问，或比较它们的方法与结论", disabled=not enabled) or retry_question
if question:
    provider = None
    try:
        with st.spinner("加载本地检索资源…"):
            model, counter, profile = embedding_resources()
            retriever = documents.load_scoped_retriever([item["document_id"] for item in details["documents"]], model, profile)
            provider = create_provider(config, key)
        # Freeze all arguments: Streamlit reruns must not retarget a running job.
        def work(token, callback, service=ResearchChatService(conversations), cid=selected_id,
                 prompt=question, llm=provider, source=retriever, request=uuid.uuid4().hex):
            try:
                return service.ask(cid, prompt, llm, source, request, cancellation=token, on_event=callback)
            finally:
                llm.close()
        st.session_state["research_generation_task"] = generation_pool().start(selected_id, work)
        provider = None  # Ownership transferred to the worker.
        st.rerun()
    except (StorageError, ModelError) as exc:
        st.error(str(exc))
    except Exception:
        st.error("本轮生成未完成；请检查模型、索引和网络后重试。未保存不完整的工具回合。")
    finally:
        if provider:
            provider.close()
