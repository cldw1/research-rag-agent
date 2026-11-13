"""Single-user local document lifecycle, transactional catalog + immutable files.

Each operation owns a short-lived Session. Long work runs outside transactions;
persisted leases prevent concurrent publication, and activation is conditional.
No raw object or historical index is physically deleted by this service.
"""

import hashlib
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.ingestion.document import DocumentParseError, StructuredDocument
from src.ingestion.parsers import PARSER_VERSION, SUPPORTED_TYPES, ParseOptions, UnifiedDocumentParser, parser_dependencies
from src.ingestion.structured_chunker import ChunkConfig, StructuredChunker
from src.storage.artifacts import (StorageError, file_hash, fingerprint, safe_path,
                                   save_index, save_object, verify_index)
from src.storage.database import open_catalog, write_session
from src.storage.models import ChunkRecord, DocumentRecord, IndexVersion, IngestionJob, ParsedVersion


def now():
    return datetime.now(timezone.utc)


def timestamp(value=None):
    return (value or now()).isoformat()


def default_data_root():
    project_root = Path(__file__).resolve().parents[2]
    configured = Path(os.getenv("COPILOT_DATA_DIR") or "data/copilot_v2")
    return configured if configured.is_absolute() else project_root / configured


@dataclass(frozen=True)
class IngestResult:
    document: StructuredDocument
    reused: bool
    job_id: str | None


@dataclass(frozen=True)
class BuildResult:
    index_id: str
    chunk_count: int
    reused: bool
    job_id: str | None


class PersistentRetriever:
    """Check active state before every search, including cached Streamlit objects."""

    def __init__(self, store, index_id, retriever):
        self.store, self.index_id, self.retriever = store, index_id, retriever

    def __getattr__(self, name):
        return getattr(self.retriever, name)

    def get_paper_data(self, paper_id):
        self.store.assert_active(self.retriever.paper_id, self.index_id)
        return self.retriever.get_paper_data(paper_id)

    def search(self, *args, **kwargs):
        self.store.assert_active(self.retriever.paper_id, self.index_id)
        return self.retriever.search(*args, **kwargs)


class LocalDocumentStore:
    lease_seconds = 3600

    def __init__(self, root=None):
        self.root = Path(root or default_data_root()).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "catalog.sqlite"
        self.engine = open_catalog(self.database_path)

    def close(self):
        self.engine.dispose()

    @staticmethod
    def _require_document(session, document_id, *, include_deleted=False):
        record = session.get(DocumentRecord, document_id)
        if record is None or (record.deleted_at and not include_deleted):
            raise StorageError("document_not_found", "文档不存在或已移出文档库")
        return record

    def _claim(self, session, record, kind):
        if record.processing_job_id:
            previous = session.get(IngestionJob, record.processing_job_id)
            if previous and previous.status == "running":
                if previous.lease_until > timestamp():
                    raise StorageError("job_running", "该文档有处理任务正在执行，请稍后重试")
                previous.status, previous.error_code = "interrupted", "lease_expired"
                previous.updated_at = timestamp()
        job = IngestionJob(id=uuid.uuid4().hex, document_id=record.id, kind=kind,
            status="running", phase="starting", progress=0, created_at=timestamp(),
            updated_at=timestamp(), lease_until=timestamp(now() + timedelta(seconds=self.lease_seconds)))
        session.add(job)
        record.processing_job_id = job.id
        return job.id

    def _owned_job(self, session, document_id, job_id):
        record = self._require_document(session, document_id)
        job = session.get(IngestionJob, job_id)
        if not job or record.processing_job_id != job_id or job.status != "running":
            raise StorageError("job_superseded", "任务已取消或被替代，未激活其结果")
        return record, job

    def _progress(self, document_id, job_id, phase, progress):
        with write_session(self.engine) as session:
            _, job = self._owned_job(session, document_id, job_id)
            job.phase, job.progress, job.updated_at = phase, progress, timestamp()
            job.lease_until = timestamp(now() + timedelta(seconds=self.lease_seconds))

    def _fail(self, document_id, job_id, code):
        with write_session(self.engine) as session:
            record = session.get(DocumentRecord, document_id)
            job = session.get(IngestionJob, job_id)
            if job and job.status == "running":
                job.status, job.error_code, job.updated_at = "failed", code, timestamp()
            if record and record.processing_job_id == job_id:
                record.processing_job_id = None
                if not record.current_parse_id and not record.deleted_at:
                    record.status = "failed"

    @staticmethod
    def _complete(record, job):
        job.status, job.phase, job.progress, job.updated_at = "completed", "completed", 100, timestamp()
        record.processing_job_id = None

    def _check_object(self, record):
        path = safe_path(self.root, record.object_path)
        try:
            if path.stat().st_size != record.file_size or file_hash(path) != record.file_hash:
                raise ValueError("object checksum")
        except (OSError, ValueError) as exc:
            raise StorageError("object_corrupt", "原文件缺失或校验失败，未自动覆盖") from exc
        return path

    @staticmethod
    def _structured(record, version):
        if version is None:
            raise StorageError("not_parsed", "文档还没有可恢复的解析结果")
        document = StructuredDocument.from_dict(version.structured_document)
        if (version.document_id != record.id or document.document_id != record.id
                or document.document_version != version.id or document.file_hash != record.file_hash
                or document.file_size != record.file_size):
            raise StorageError("catalog_corrupt", "解析版本与原始文档记录不一致")
        return document

    def ingest(self, content, filename, options=None, *, force=False, progress=None):
        options = options or ParseOptions()
        filename = PurePosixPath(str(filename).replace("\\", "/")).name
        suffix = PurePosixPath(filename).suffix.lower().lstrip(".")
        if suffix not in SUPPORTED_TYPES:
            raise DocumentParseError("unsupported_type", "支持 PDF、DOCX、MD 和 TXT")
        if not isinstance(content, bytes) or not content or len(content) > options.max_bytes:
            raise DocumentParseError("invalid_upload", "文件为空、大小超限或内容无效")
        if ((suffix == "pdf") != content.startswith(b"%PDF") or
                (suffix == "docx") != content.startswith(b"PK")):
            raise DocumentParseError("type_mismatch", "文件内容与扩展名不一致")
        digest = hashlib.sha256(content).hexdigest()
        document_id = "doc_" + digest
        request = fingerprint({"hash": digest, "type": suffix, "parser": PARSER_VERSION,
            "options": asdict(options), "dependencies": parser_dependencies(options, suffix)})
        relative = save_object(self.root, digest, content)
        with write_session(self.engine) as session:
            record = session.get(DocumentRecord, document_id)
            if record is None:
                record = DocumentRecord(id=document_id, file_hash=digest, filename=filename,
                    file_type=suffix, file_size=len(content), object_path=relative,
                    created_at=timestamp(), status="processing")
                session.add(record)
                session.flush()
            if record.deleted_at:
                raise StorageError("document_deleted", "相同文件已移出文档库，请使用恢复操作后重新处理")
            if record.file_type != suffix:
                raise StorageError("file_type_conflict", "相同内容已按另一种文本格式保存，请使用文档库中的原记录")
            filename = record.filename
            version = session.get(ParsedVersion, record.current_parse_id) if record.current_parse_id else None
            if version and version.request_fingerprint == request and not force:
                return IngestResult(self._structured(record, version), True, None)
            job_id = self._claim(session, record, "parse")
        try:
            self._progress(document_id, job_id, "parsing", 10)
            def on_page(done, total):
                self._progress(document_id, job_id, "parsing", 10 + int(60 * done / total))
                if progress:
                    progress(done, total)
            document = UnifiedDocumentParser(options, progress=on_page).parse(content, filename)
            with write_session(self.engine) as session:
                record, job = self._owned_job(session, document_id, job_id)
                version = session.get(ParsedVersion, document.document_version)
                if version is None:
                    session.add(ParsedVersion(id=document.document_version, document_id=document_id,
                        request_fingerprint=request, parser_version=document.parser_version,
                        parse_options=asdict(options), structured_document=document.to_dict(), created_at=timestamp()))
                record.current_parse_id, record.status = document.document_version, document.status
                self._complete(record, job)
            return IngestResult(document, False, job_id)
        except Exception as exc:
            self._fail(document_id, job_id, getattr(exc, "code", "parse_failed"))
            raise

    def list_documents(self, *, include_deleted=False):
        with Session(self.engine) as session:
            query = select(DocumentRecord).order_by(DocumentRecord.created_at.desc(), DocumentRecord.id)
            if not include_deleted:
                query = query.where(DocumentRecord.deleted_at.is_(None))
            result = []
            for record in session.scalars(query):
                index = session.get(IndexVersion, record.active_index_id) if record.active_index_id else None
                result.append({"document_id": record.id, "filename": record.filename,
                    "file_type": record.file_type, "file_size": record.file_size,
                    "status": record.status, "created_at": record.created_at, "deleted_at": record.deleted_at,
                    "parse_id": record.current_parse_id, "index_id": record.active_index_id,
                    "chunk_count": index.chunk_count if index else 0, "processing_job_id": record.processing_job_id})
            return result

    def get_document(self, document_id):
        with Session(self.engine) as session:
            record = self._require_document(session, document_id)
            self._check_object(record)
            version = session.get(ParsedVersion, record.current_parse_id) if record.current_parse_id else None
            return self._structured(record, version)

    def reparse(self, document_id, options=None, progress=None):
        with Session(self.engine) as session:
            record = self._require_document(session, document_id)
            path = self._check_object(record)
            filename = record.filename
        return self.ingest(path.read_bytes(), filename, options, force=True, progress=progress)

    def assert_active(self, document_id, index_id):
        with Session(self.engine) as session:
            record = self._require_document(session, document_id)
            if record.active_index_id != index_id:
                raise StorageError("stale_index", "索引活动版本已改变，请恢复当前版本后重试")

    def active_index_info(self, document_id):
        with Session(self.engine) as session:
            record = self._require_document(session, document_id)
            index = session.get(IndexVersion, record.active_index_id) if record.active_index_id else None
            if index is None:
                return None
            if index.document_id != document_id:
                raise StorageError("catalog_corrupt", "索引指针与当前文档不一致")
            return {"index_id": index.id, "parse_id": index.parse_id, "chunk_count": index.chunk_count,
                    "chunk_config": index.chunk_config, "embedding_profile": index.embedding_profile}

    def build_index(self, document_id, model, profile, counter, config=None, *, force=False):
        from src.retrieval.structured_retriever import StructuredDocumentRetriever

        config = config or ChunkConfig()
        if profile.tokenizer != counter.identity or profile.max_sequence_length != model.max_seq_length:
            raise StorageError("model_mismatch", "Embedding 配置与当前 Tokenizer 或输入上限不一致")
        document = self.get_document(document_id)
        chunks = StructuredChunker(config, counter).chunk(document)
        profile_data = profile.to_dict()
        identity = fingerprint({"chunk_set": chunks[0]["index_version"], "profile": profile_data})
        with write_session(self.engine) as session:
            record = self._require_document(session, document_id)
            candidate = session.scalar(select(IndexVersion).where(
                IndexVersion.document_id == document_id, IndexVersion.fingerprint == identity)
                .order_by(IndexVersion.created_at.desc()))
            if candidate and not force:
                verify_index(self.root, candidate.artifact_path, candidate.manifest)
                if record.current_parse_id != document.document_version:
                    raise StorageError("parse_changed", "解析版本已改变，请重试索引构建")
                if record.processing_job_id:
                    raise StorageError("job_running", "文档有处理任务正在执行")
                record.active_index_id = candidate.id
                return BuildResult(candidate.id, candidate.chunk_count, True, None)
            job_id = self._claim(session, record, "index")
        try:
            self._progress(document_id, job_id, "embedding", 20)
            retriever = StructuredDocumentRetriever(document, chunks, model)
            if retriever.vectors.shape[1] != profile.dimension:
                raise StorageError("dimension_mismatch", "Embedding 实际向量维度与配置不一致")
            import numpy as np
            if not np.allclose(np.linalg.norm(retriever.vectors, axis=1), 1, atol=1e-3):
                raise StorageError("normalization_mismatch", "Embedding 未生成有效的单位向量")
            self._progress(document_id, job_id, "writing_index", 75)
            index_id = uuid.uuid4().hex
            relative, manifest = save_index(self.root, index_id, retriever, profile_data)
            verify_index(self.root, relative, manifest)
            with write_session(self.engine) as session:
                record, job = self._owned_job(session, document_id, job_id)
                if record.current_parse_id != document.document_version:
                    raise StorageError("parse_changed", "解析版本已改变，未激活旧解析的索引")
                session.add(IndexVersion(id=index_id, document_id=document_id,
                    parse_id=document.document_version, fingerprint=identity, chunk_config=asdict(config),
                    embedding_profile=profile_data, artifact_path=relative, manifest=manifest,
                    chunk_count=len(chunks), created_at=timestamp()))
                session.flush()
                session.add_all(ChunkRecord(index_id=index_id, ordinal=i, chunk_id=chunk["chunk_id"], data=chunk)
                                for i, chunk in enumerate(chunks))
                record.active_index_id = index_id
                self._complete(record, job)
            return BuildResult(index_id, len(chunks), False, job_id)
        except Exception as exc:
            self._fail(document_id, job_id, getattr(exc, "code", "index_failed"))
            raise

    def load_retriever(self, document_id, model, profile):
        import numpy as np
        from src.runtime import load_faiss
        from src.retrieval.structured_retriever import StructuredDocumentRetriever

        with Session(self.engine) as session:
            record = self._require_document(session, document_id)
            self._check_object(record)
            version = session.get(IndexVersion, record.active_index_id) if record.active_index_id else None
            if version is None:
                raise StorageError("not_indexed", "文档尚未建立持久化索引")
            if version.embedding_profile != profile.to_dict():
                raise StorageError("model_mismatch", "当前 Embedding 模型或配置已改变，需要重新构建索引")
            parsed = session.get(ParsedVersion, version.parse_id)
            chunks = [c.data for c in session.scalars(select(ChunkRecord).where(
                ChunkRecord.index_id == version.id).order_by(ChunkRecord.ordinal))]
            document = self._structured(record, parsed)
            if (len(chunks) != version.chunk_count or fingerprint(chunks) != version.manifest["chunks_sha256"]
                    or version.document_id != document_id or document.document_id != document_id
                    or version.manifest["document_version"] != document.document_version
                    or version.manifest["embedding_profile"] != profile.to_dict()):
                raise StorageError("catalog_corrupt", "数据库中的索引映射或来源版本不一致")
            directory = verify_index(self.root, version.artifact_path, version.manifest)
            vectors = np.load(directory / "vectors.npy", allow_pickle=False)
            index = load_faiss().deserialize_index(np.frombuffer((directory / "index.faiss").read_bytes(), dtype=np.uint8).copy())
            if vectors.shape != (version.chunk_count, profile.dimension) or vectors.dtype != np.float32:
                raise StorageError("index_corrupt", "向量文件维度或数据类型不一致")
            retriever = StructuredDocumentRetriever.from_saved(document, chunks, model, vectors, index)
            index_id = version.id
        self.assert_active(document_id, index_id)
        return PersistentRetriever(self, index_id, retriever)

    def delete_document(self, document_id):
        with write_session(self.engine) as session:
            record = self._require_document(session, document_id, include_deleted=True)
            if record.deleted_at:
                return
            record.deleted_at, record.status, record.active_index_id = timestamp(), "deleted", None
            if record.processing_job_id:
                job = session.get(IngestionJob, record.processing_job_id)
                if job and job.status == "running":
                    job.status, job.error_code, job.updated_at = "cancelled", "document_deleted", timestamp()
                record.processing_job_id = None

    def restore_document(self, document_id):
        with write_session(self.engine) as session:
            record = self._require_document(session, document_id, include_deleted=True)
            if not record.deleted_at:
                return
            self._check_object(record)
            record.deleted_at = None
            version = session.get(ParsedVersion, record.current_parse_id) if record.current_parse_id else None
            record.status = version.structured_document["status"] if version else "failed"
            # Restoring metadata does not silently reactivate a previously deleted index.
            record.active_index_id = None

    def list_jobs(self, document_id=None):
        with Session(self.engine) as session:
            query = select(IngestionJob).order_by(IngestionJob.created_at.desc()).limit(100)
            if document_id:
                query = query.where(IngestionJob.document_id == document_id)
            return [{key: getattr(job, key) for key in (
                "id", "document_id", "kind", "status", "phase", "progress", "error_code", "updated_at", "lease_until")}
                for job in session.scalars(query)]

    def recover_expired_jobs(self):
        count = 0
        with write_session(self.engine) as session:
            jobs = session.scalars(select(IngestionJob).where(
                IngestionJob.status == "running", IngestionJob.lease_until <= timestamp()))
            for job in jobs:
                job.status, job.error_code, job.updated_at = "interrupted", "lease_expired", timestamp()
                record = session.get(DocumentRecord, job.document_id)
                if record and record.processing_job_id == job.id:
                    record.processing_job_id = None
                    if not record.current_parse_id and not record.deleted_at:
                        record.status = "failed"
                count += 1
        return count
