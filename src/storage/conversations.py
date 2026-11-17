"""Persistent complete turns, citations and idempotency, with isolated scope."""

import re
import uuid
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.agent.context_window import validate_turn
from src.llm.provider import ModelConfig
from src.storage.artifacts import StorageError
from src.storage.database import catalog_version, write_session
from src.storage.document_store import now, timestamp
from src.storage.models import (ChunkRecord, CitationRecord, ConversationDocument, ConversationRecord, ConversationTurn,
                                DocumentRecord, IndexVersion, MessageRecord, SchemaVersion)


MAX_CONVERSATION_DOCUMENTS = 5


def source_location_label(source):
    """Display saved locations without assigning page numbers to Word or text."""
    if source.get('page_number'):
        return f"PDF 第 {source['page_number']} 页"
    spans = source.get('source_metadata', {}).get('spans', [])
    starts = [s['line_start'] for s in spans if s.get('line_start') is not None]
    ends = [s.get('line_end') or s['line_start'] for s in spans if s.get('line_start') is not None]
    if starts:
        first, last = min(starts), max(ends)
        return f"原文第 {first} 行" if first == last else f"原文第 {first}–{last} 行"
    blocks = list(dict.fromkeys(s['block_id'] for s in spans if s.get('block_id')))
    section = source.get('section') or ''
    if blocks:
        return (section+' · ' if section else '')+'结构块 '+', '.join(blocks)
    return section or '段落定位'


def _verbatim_markdown(text):
    # A longer fence preserves code fences and literal Markdown inside evidence.
    length = max([len(match[0]) for match in re.finditer(r'`+', text)] + [2]) + 1
    fence = '`' * length
    return f'{fence}text\n{text}\n{fence}'


class ConversationStore:
    def __init__(self, documents):
        self.documents, self.engine = documents, documents.engine
        if catalog_version(self.engine) not in (2, 3):
            raise StorageError("schema_upgrade_required", "文档库需要先备份并升级聊天存储结构")

    @staticmethod
    def _bindings(session, conversation):
        if session.scalar(select(SchemaVersion.version)) == 2:
            return [(conversation.document_id, conversation.index_id)]
        rows = list(session.scalars(select(ConversationDocument).where(
            ConversationDocument.conversation_id == conversation.id).order_by(ConversationDocument.position)))
        if (not 1 <= len(rows) <= MAX_CONVERSATION_DOCUMENTS or rows[0].position != 0
                or (rows[0].document_id, rows[0].index_id) != (conversation.document_id, conversation.index_id)
                or [row.position for row in rows] != list(range(len(rows)))):
            raise StorageError("invalid_conversation_scope", "会话文档集合与索引范围无效")
        return [(row.document_id, row.index_id) for row in rows]

    @staticmethod
    def _guard(session, conversation_id, *, require_active=False):
        conversation = session.get(ConversationRecord, conversation_id)
        if conversation is None or conversation.deleted_at:
            raise StorageError("conversation_not_found", "会话不存在或已删除")
        for document_id, index_id in ConversationStore._bindings(session, conversation):
            document = session.get(DocumentRecord, document_id)
            if document is None or document.deleted_at:
                raise StorageError("document_not_found", "会话所属文档已移出文档库")
            index = session.get(IndexVersion, index_id)
            if index is None or index.document_id != document.id:
                raise StorageError("invalid_conversation_scope", "会话的索引范围无效")
            if require_active and document.active_index_id != index_id:
                raise StorageError("conversation_index_changed", "文档索引已变更，请基于当前版本新建会话；历史来源保持原版本")
        return conversation

    def create(self, document_id, config, title="新会话"):
        if not isinstance(config, ModelConfig):
            raise StorageError("invalid_model_config", "模型配置必须使用不含凭据的标准结构")
        title = self._title(title)
        ids = [document_id] if isinstance(document_id, str) else document_id
        if (not isinstance(ids, (list, tuple)) or not 1 <= len(ids) <= MAX_CONVERSATION_DOCUMENTS
                or any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids)):
            raise StorageError("invalid_document_selection", "请选择 1–5 篇不同的已索引文档")
        with write_session(self.engine) as session:
            version = session.scalar(select(SchemaVersion.version))
            if len(ids) > 1 and version < 3:
                raise StorageError("schema_upgrade_required", "请先备份并启用多文档会话存储")
            selected = []
            for selected_id in ids:
                document = self.documents._require_document(session, selected_id)
                if not document.active_index_id:
                    raise StorageError("not_indexed", "请先为文档建立索引")
                index = session.get(IndexVersion, document.active_index_id)
                if index is None or index.document_id != selected_id:
                    raise StorageError("invalid_conversation_scope", "当前索引与文档不一致")
                selected.append((selected_id, document.active_index_id))
            conversation = ConversationRecord(id=uuid.uuid4().hex, document_id=selected[0][0],
                index_id=selected[0][1], title=title, model_config=ModelConfig.to_dict(config),
                created_at=timestamp(), updated_at=timestamp())
            session.add(conversation)
            session.flush()
            if version >= 3:
                session.add_all(ConversationDocument(conversation_id=conversation.id, document_id=did,
                    index_id=iid, position=i) for i, (did, iid) in enumerate(selected))
            return conversation.id

    @staticmethod
    def _title(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 100:
            raise StorageError("invalid_title", "会话标题应为 1–100 字符")
        return value.strip()

    def list(self):
        with Session(self.engine) as session:
            rows = session.scalars(select(ConversationRecord).where(ConversationRecord.deleted_at.is_(None))
                .order_by(ConversationRecord.updated_at.desc()))
            result = []
            for item in rows:
                try:
                    self._guard(session, item.id)
                except StorageError:
                    continue
                result.append({**self._details(session, item), "updated_at": item.updated_at})
            return result

    @staticmethod
    def _details(session, item):
        documents = []
        for did, iid in ConversationStore._bindings(session, item):
            document = session.get(DocumentRecord, did)
            documents.append({"document_id": did, "index_id": iid, "filename": document.filename})
        return {"id": item.id, "title": item.title, "document_id": item.document_id,
                "index_id": item.index_id, "model_config": item.model_config, "documents": documents}

    def details(self, conversation_id, *, require_active=False):
        with Session(self.engine) as session:
            item = self._guard(session, conversation_id, require_active=require_active)
            return self._details(session, item)

    def rename(self, conversation_id, title):
        title = self._title(title)
        with write_session(self.engine) as session:
            item = self._guard(session, conversation_id)
            item.title, item.updated_at = title, timestamp()

    def delete(self, conversation_id):
        with write_session(self.engine) as session:
            item = self._guard(session, conversation_id)
            item.deleted_at, item.updated_at = timestamp(), timestamp()
            if item.active_turn_id:
                turn = session.get(ConversationTurn, item.active_turn_id)
                if turn and turn.status == "running":
                    turn.status, turn.error_code, turn.updated_at = "cancelled", "conversation_deleted", timestamp()
                item.active_turn_id = None

    def begin(self, conversation_id, question, config, request_id):
        if not isinstance(config, ModelConfig):
            raise StorageError("invalid_model_config", "模型配置必须使用不含凭据的标准结构")
        settings = ModelConfig.to_dict(config)
        if not isinstance(question, str) or not question.strip() or len(question) > 8000:
            raise StorageError("invalid_question", "问题不能为空且不能超过 8000 字符")
        if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id):
            raise StorageError("invalid_request_id", "请求 ID 无效")
        with write_session(self.engine) as session:
            conversation = self._guard(session, conversation_id, require_active=True)
            previous = session.scalar(select(ConversationTurn).where(
                ConversationTurn.conversation_id == conversation_id, ConversationTurn.request_id == request_id))
            if previous:
                if previous.question != question or ModelConfig(**previous.model_config).to_dict() != settings:
                    raise StorageError("request_conflict", "同一请求 ID 不能用于不同问题或模型配置")
                if previous.status == "completed":
                    return previous.id, previous.result
                raise StorageError("request_already_used", "请求已处理或失败，请使用新的请求 ID 重试")
            if conversation.active_turn_id:
                active = session.get(ConversationTurn, conversation.active_turn_id)
                if active and active.status == "running":
                    if active.lease_until > timestamp():
                        raise StorageError("conversation_busy", "此会话正在生成回答，请等待当前请求完成")
                    active.status, active.error_code = "interrupted", "lease_expired"
            ordinal = (session.scalar(select(func.max(ConversationTurn.ordinal)).where(
                ConversationTurn.conversation_id == conversation_id)) or 0) + 1
            turn = ConversationTurn(id=uuid.uuid4().hex, conversation_id=conversation_id, ordinal=ordinal,
                request_id=request_id, question=question, status="running", model_config=settings,
                created_at=timestamp(), updated_at=timestamp(), lease_until=timestamp(now() + timedelta(minutes=15)))
            session.add(turn)
            session.flush()
            session.add(MessageRecord(turn_id=turn.id, position=0, role="user", data={"role": "user", "content": question}))
            conversation.active_turn_id = turn.id
            return turn.id, None

    def history(self, conversation_id):
        with Session(self.engine) as session:
            self._guard(session, conversation_id)
            turns = session.scalars(select(ConversationTurn).where(
                ConversationTurn.conversation_id == conversation_id).order_by(ConversationTurn.ordinal))
            output = []
            for turn in turns:
                messages = list(session.scalars(select(MessageRecord).where(
                    MessageRecord.turn_id == turn.id).order_by(MessageRecord.position)))
                output.append({"id": turn.id, "question": turn.question, "status": turn.status,
                    "error_code": turn.error_code, "model_config": turn.model_config,
                    "result": turn.result, "messages": [message.data for message in messages]})
            return output

    def sources(self, conversation_id):
        with Session(self.engine) as session:
            self._guard(session, conversation_id)
            return [row.source for row in session.scalars(select(CitationRecord).where(
                CitationRecord.conversation_id == conversation_id).order_by(CitationRecord.number))]

    def complete(self, conversation_id, turn_id, result):
        messages = result["messages"]
        validate_turn(messages)
        with write_session(self.engine) as session:
            conversation = self._guard(session, conversation_id, require_active=True)
            turn = session.get(ConversationTurn, turn_id)
            if (turn is None or turn.conversation_id != conversation_id or turn.status != "running"
                    or conversation.active_turn_id != turn_id):
                raise StorageError("turn_superseded", "当前回合已取消或被替代")
            if (messages[0].get("role") != "user" or messages[0].get("content") != turn.question
                    or set(messages[0]) - {"role", "content", "reference_context"}
                    or ("reference_context" in messages[0] and not isinstance(messages[0]["reference_context"], str))):
                raise StorageError("invalid_history", "回合用户消息与原始问题不一致")
            # Validate every registry source against the selected immutable index.
            bindings = dict(self._bindings(session, conversation))
            for source in result["registered_sources"]:
                index_id = bindings.get(source["paper_id"])
                if index_id is None or source.get("persistent_index_id", index_id) != index_id:
                    raise StorageError("citation_scope_violation", "引用不属于会话绑定的文档索引")
                chunk = session.scalar(select(ChunkRecord).where(
                    ChunkRecord.index_id == index_id, ChunkRecord.chunk_id == source["chunk_id"]))
                if chunk is None:
                    raise StorageError("citation_scope_violation", "引用不属于会话绑定的文档索引")
                number = int(source["citation"])
                existing = session.get(CitationRecord, (conversation_id, number))
                if existing and (existing.chunk_id != source["chunk_id"] or existing.index_id != index_id):
                    raise StorageError("citation_conflict", "已分配的引用编号不能指向其他来源")
                if existing is None:
                    snapshot = {**source, "text": chunk.data["text"], "page_number": chunk.data.get("page_number"),
                                "source_metadata": chunk.data.get("source_metadata", {}),
                                "persistent_index_id": index_id}
                    session.add(CitationRecord(conversation_id=conversation_id, number=number,
                        index_id=index_id, chunk_id=source["chunk_id"], source=snapshot))
            session.add_all(MessageRecord(turn_id=turn_id, position=i, role=message["role"], data=message)
                            for i, message in enumerate(messages) if i)
            turn.result, turn.status, turn.updated_at = result, "completed", timestamp()
            conversation.active_turn_id, conversation.model_config, conversation.updated_at = None, turn.model_config, timestamp()

    def fail(self, conversation_id, turn_id, code):
        # Error codes are internal constants; never store SDK exception messages.
        if not re.fullmatch(r"[a-z_]{1,64}", code):
            code = "generation_failed"
        with write_session(self.engine) as session:
            conversation = session.get(ConversationRecord, conversation_id)
            turn = session.get(ConversationTurn, turn_id)
            if turn and turn.conversation_id == conversation_id and turn.status == "running":
                turn.status, turn.error_code, turn.updated_at = "failed", code, timestamp()
            if conversation and conversation.active_turn_id == turn_id:
                conversation.active_turn_id = None

    def cancel(self, conversation_id, turn_id):
        with write_session(self.engine) as session:
            conversation = session.get(ConversationRecord, conversation_id)
            turn = session.get(ConversationTurn, turn_id)
            if turn and turn.conversation_id == conversation_id and turn.status == "running":
                turn.status, turn.error_code, turn.updated_at = "cancelled", "generation_cancelled", timestamp()
            if conversation and conversation.active_turn_id == turn_id:
                conversation.active_turn_id = None

    def recover_expired(self):
        count = 0
        with write_session(self.engine) as session:
            for turn in session.scalars(select(ConversationTurn).where(
                    ConversationTurn.status == "running", ConversationTurn.lease_until <= timestamp())):
                turn.status, turn.error_code, turn.updated_at = "interrupted", "lease_expired", timestamp()
                conversation = session.get(ConversationRecord, turn.conversation_id)
                if conversation and conversation.active_turn_id == turn.id:
                    conversation.active_turn_id = None
                count += 1
        return count

    def export_markdown(self, conversation_id):
        details = self.details(conversation_id)
        sources = {item["citation"]: item for item in self.sources(conversation_id)}
        filenames = {item['document_id']:item['filename'] for item in details['documents']}
        parts = [f"# {details['title']}"]
        parts.append("文档范围：" + "、".join(item["filename"] for item in details["documents"]))
        for turn in self.history(conversation_id):
            parts.append("## 用户\n\n" + turn["question"])
            if turn["status"] != "completed":
                parts.append(f"状态：{turn['status']}")
                continue
            parts.append("## 助手\n\n" + turn["result"]["answer"])
            settings=turn['result'].get('retrieval_settings')
            if settings:
                from src.retrieval.session import EVIDENCE_ORDERS
                mode=settings['multi_document_order']
                parts.append('本轮多文档设置：'+EVIDENCE_ORDERS.get(mode,mode))
            for number in turn["result"]["cited_numbers"]:
                source = sources.get(number)
                if source:
                    location = source_location_label(source)
                    filename = filenames.get(source['paper_id'], source.get('source_metadata', {}).get('filename', source['paper_id']))
                    if list(filenames.values()).count(filename) > 1:
                        filename += ' · '+source['paper_id'].removeprefix('doc_')[:12]
                    parts.append(f"[{number}] {filename} · {source.get('title', '')} · {location}\n\n" + _verbatim_markdown(source['text']))
        return "\n\n".join(parts) + "\n"
