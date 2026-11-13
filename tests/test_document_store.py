"""Document lifecycle invariants using real SQLite/FAISS and test-only vectors."""

import json
import sqlite3
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.ingestion.document import DocumentParseError
from src.ingestion.parsers import ParseOptions, UnifiedDocumentParser
from src.ingestion.structured_chunker import ByteBudgetCounter, ChunkConfig
from src.storage.artifacts import StorageError, safe_path
from src.storage.backup import create_backup, restore_backup, verify_backup
from src.storage.database import write_session
from src.storage.document_store import LocalDocumentStore
from src.storage.embedding_profile import EmbeddingProfile
from src.storage.models import ChunkRecord, DocumentRecord, IndexVersion, IngestionJob


TEXT = b"# Methods\n\nMomentum uses a position update.\n\n# Results\n\nExperiment measures accuracy."


class TinyTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text) + 2))


class FixtureEmbedding:
    tokenizer = TinyTokenizer()
    max_seq_length = 512

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        vectors = np.array([[1., float("momentum" in t.lower()), float("experiment" in t.lower())]
                            for t in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


@pytest.fixture
def store(tmp_path):
    instance = LocalDocumentStore(tmp_path / "catalog")
    yield instance
    instance.close()


@pytest.fixture
def embedding():
    counter = ByteBudgetCounter()
    profile = EmbeddingProfile("test-only", "deterministic-v1", 3, counter.identity, 512)
    return FixtureEmbedding(), counter, profile


def indexed(store, embedding):
    model, counter, profile = embedding
    document = store.ingest(TEXT, "paper.md").document
    built = store.build_index(document.document_id, model, profile, counter, ChunkConfig(strategy="structure"))
    return document, built, store.load_retriever(document.document_id, model, profile)


def test_restart_recovers_document_and_vectors_without_parsing_or_encoding(store, embedding, monkeypatch):
    document, built, _ = indexed(store, embedding)
    model, _, profile = embedding
    store.close()
    monkeypatch.setattr(UnifiedDocumentParser, "parse", Mock(side_effect=AssertionError("no reparse")))
    model.encode = Mock(side_effect=AssertionError("no document re-embedding"))
    reopened = LocalDocumentStore(store.root)
    try:
        recovered = reopened.get_document(document.document_id)
        assert recovered.to_dict() == document.to_dict()
        retriever = reopened.load_retriever(document.document_id, model, profile)
        results = retriever.search("momentum", document.document_id, method="bm25")
        assert results and "Momentum" in results[0]["text"]
        assert results[0]["source_metadata"]["spans"]
        assert reopened.active_index_info(document.document_id)["index_id"] == built.index_id
        model.encode.assert_not_called()
    finally:
        reopened.close()


def test_duplicate_upload_reuses_parse_after_restart(store, monkeypatch):
    first = store.ingest(TEXT, "first.md")
    monkeypatch.setattr(UnifiedDocumentParser, "parse", Mock(side_effect=AssertionError("duplicate parse")))
    repeated = store.ingest(TEXT, "renamed.md")
    assert repeated.reused and repeated.document.document_id == first.document.document_id
    assert repeated.document.filename == "first.md"
    assert len(store.list_documents()) == len(store.list_jobs()) == 1


def test_identical_index_request_reuses_vectors_and_forced_rebuild_invalidates_old_cache(store, embedding):
    document, old, cached = indexed(store, embedding)
    model, counter, profile = embedding
    count = len(model.calls)
    config = ChunkConfig(strategy="structure")
    same = store.build_index(document.document_id, model, profile, counter, config)
    assert same.reused and same.index_id == old.index_id and len(model.calls) == count
    new = store.build_index(document.document_id, model, profile, counter, config, force=True)
    assert not new.reused and new.index_id != old.index_id
    assert (store.root / "indexes" / old.index_id / "index.faiss").exists()
    with pytest.raises(StorageError, match="活动版本"):
        cached.search("momentum", document.document_id, method="bm25")


def test_failed_embedding_keeps_old_active_index_and_stores_sanitized_error(store, embedding):
    document, old, cached = indexed(store, embedding)
    model, counter, profile = embedding
    model.encode = Mock(side_effect=RuntimeError("private authentication secret"))
    with pytest.raises(RuntimeError):
        store.build_index(document.document_id, model, profile, counter, force=True)
    assert store.active_index_info(document.document_id)["index_id"] == old.index_id
    assert cached.search("momentum", document.document_id, method="bm25")
    assert store.list_jobs()[0]["status"] == "failed"
    assert store.list_jobs()[0]["error_code"] == "index_failed"
    assert "secret" not in json.dumps(store.list_jobs())


def test_failed_publication_rolls_back_new_rows_and_active_pointer(store, embedding, monkeypatch):
    document, old, _ = indexed(store, embedding)
    model, counter, profile = embedding
    monkeypatch.setattr(store, "_complete", Mock(side_effect=RuntimeError("simulate activation failure")))
    with pytest.raises(RuntimeError):
        store.build_index(document.document_id, model, profile, counter, force=True)
    assert store.active_index_info(document.document_id)["index_id"] == old.index_id
    with Session(store.engine) as session:
        assert len(list(session.scalars(select(IndexVersion)))) == 1
        assert {row.index_id for row in session.scalars(select(ChunkRecord))} == {old.index_id}


def test_model_and_chunk_config_changes_cannot_mix_indexes(store, embedding):
    document, old, cached = indexed(store, embedding)
    model, counter, profile = embedding
    changed = replace(profile, revision="deterministic-v2")
    with pytest.raises(StorageError, match="模型或配置"):
        store.load_retriever(document.document_id, model, changed)
    new = store.build_index(document.document_id, model, changed, counter, ChunkConfig(max_tokens=40))
    assert new.index_id != old.index_id
    assert store.active_index_info(document.document_id)["embedding_profile"]["revision"] == "deterministic-v2"
    with pytest.raises(StorageError):
        cached.get_paper_data(document.document_id)


def test_wrong_vector_dimension_never_activates(store, embedding):
    document, old, _ = indexed(store, embedding)
    model, counter, profile = embedding
    with pytest.raises(StorageError, match="实际向量维度"):
        store.build_index(document.document_id, model, replace(profile, dimension=99), counter)
    assert store.active_index_info(document.document_id)["index_id"] == old.index_id


@pytest.mark.parametrize("filename", ["index.faiss", "vectors.npy", "manifest.json"])
def test_corrupt_artifacts_are_rejected_before_native_faiss_deserialization(store, embedding, monkeypatch, filename):
    document, built, _ = indexed(store, embedding)
    from src.runtime import load_faiss
    native = Mock(side_effect=AssertionError("must not parse corrupt data"))
    monkeypatch.setattr(load_faiss(), "deserialize_index", native)
    path = store.root / "indexes" / built.index_id / filename
    path.write_bytes(b"corrupted")
    with pytest.raises(StorageError, match="校验失败"):
        store.load_retriever(document.document_id, embedding[0], embedding[2])
    native.assert_not_called()


def test_tampered_chunk_mapping_is_not_silently_restored(store, embedding):
    document, built, _ = indexed(store, embedding)
    with write_session(store.engine) as session:
        chunk = session.scalar(select(ChunkRecord).where(ChunkRecord.index_id == built.index_id))
        chunk.data = {**chunk.data, "text": "altered evidence"}
    with pytest.raises(StorageError, match="映射或来源版本"):
        store.load_retriever(document.document_id, embedding[0], embedding[2])


def test_deletion_is_idempotent_reversible_and_never_erases_raw_files(store, embedding):
    document, built, cached = indexed(store, embedding)
    original = store.root / "objects" / document.file_hash
    store.delete_document(document.document_id)
    store.delete_document(document.document_id)
    assert not store.list_documents() and original.read_bytes() == TEXT
    assert (store.root / "indexes" / built.index_id).is_dir()
    with pytest.raises(StorageError):
        cached.search("momentum", document.document_id)
    with pytest.raises(StorageError, match="恢复操作"):
        store.ingest(TEXT, "paper.md")
    store.restore_document(document.document_id)
    assert store.active_index_info(document.document_id) is None
    model, counter, profile = embedding
    count = len(model.calls)
    revived = store.build_index(document.document_id, model, profile, counter, ChunkConfig(strategy="structure"))
    assert revived.reused and len(model.calls) == count


def test_document_deleted_during_embedding_cannot_be_published(store, embedding):
    document = store.ingest(TEXT, "paper.md").document
    model, counter, profile = embedding
    encode = model.encode
    def cancel(texts, **kwargs):
        store.delete_document(document.document_id)
        return encode(texts, **kwargs)
    model.encode = cancel
    with pytest.raises(StorageError):
        store.build_index(document.document_id, model, profile, counter)
    assert not store.list_documents()
    assert store.list_jobs()[0]["status"] == "cancelled"
    assert store.list_documents(include_deleted=True)[0]["index_id"] is None


def test_failed_reparse_preserves_previous_parse_and_index(store, embedding, monkeypatch):
    document, built, _ = indexed(store, embedding)
    monkeypatch.setattr(UnifiedDocumentParser, "parse", Mock(side_effect=DocumentParseError("ocr_failed", "OCR failed")))
    with pytest.raises(DocumentParseError):
        store.reparse(document.document_id)
    assert store.get_document(document.document_id).document_version == document.document_version
    assert store.active_index_info(document.document_id)["index_id"] == built.index_id
    assert store.list_jobs()[0]["error_code"] == "ocr_failed"


def test_reparse_does_not_disable_previous_index_until_new_index_is_ready(store, embedding):
    document, built, cached = indexed(store, embedding)
    reparsed = store.reparse(document.document_id, ParseOptions(min_text_chars=20)).document
    assert reparsed.document_version != document.document_version
    assert store.active_index_info(document.document_id)["index_id"] == built.index_id
    assert cached.search("momentum", document.document_id, method="bm25")
    model, counter, profile = embedding
    store.build_index(document.document_id, model, profile, counter)
    restored = store.load_retriever(document.document_id, model, profile)
    assert restored.document.document_version == reparsed.document_version
    assert all(c["document_version"] == reparsed.document_version for c in restored.chunks)


def test_active_lease_blocks_duplicate_work_and_expired_lease_recovers(store):
    document = store.ingest(TEXT, "paper.md").document
    with write_session(store.engine) as session:
        record = session.get(DocumentRecord, document.document_id)
        job_id = store._claim(session, record, "parse")
    with pytest.raises(StorageError, match="正在执行"):
        store.reparse(document.document_id)
    assert store.recover_expired_jobs() == 0
    with write_session(store.engine) as session:
        session.get(IngestionJob, job_id).lease_until = "1970-01-01T00:00:00+00:00"
    assert store.recover_expired_jobs() == 1
    assert store.reparse(document.document_id).document.document_id == document.document_id


def test_original_file_corruption_is_reported_without_overwrite(store):
    document = store.ingest(TEXT, "paper.md").document
    original = store.root / "objects" / document.file_hash
    original.write_bytes(b"damaged")
    with pytest.raises(StorageError, match="未自动覆盖"):
        store.get_document(document.document_id)
    with pytest.raises(StorageError, match="未覆盖原文件"):
        store.ingest(TEXT, "paper.md")
    assert original.read_bytes() == b"damaged"


def test_backup_restore_to_new_directory_recovers_index_without_embedding(store, embedding, tmp_path):
    document, _, _ = indexed(store, embedding)
    source, target = tmp_path / "backup", tmp_path / "restored"
    create_backup(store, source)
    assert verify_backup(source)["files"]["catalog.sqlite"]
    restore_backup(source, target)
    reopened = LocalDocumentStore(target)
    try:
        model, _, profile = embedding
        model.encode = Mock(side_effect=AssertionError("must not re-embed"))
        retrieved = reopened.load_retriever(document.document_id, model, profile)
        assert retrieved.search("momentum", document.document_id, method="bm25")
        model.encode.assert_not_called()
        with pytest.raises(FileExistsError):
            restore_backup(source, target)
    finally:
        reopened.close()


def test_corrupt_backup_never_creates_restored_directory(store, embedding, tmp_path):
    indexed(store, embedding)
    backup = tmp_path / "backup"
    create_backup(store, backup)
    (backup / "catalog.sqlite").write_bytes(b"damaged")
    with pytest.raises(StorageError, match="备份不完整"):
        restore_backup(backup, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_unknown_schema_is_refused_without_destructive_migration(tmp_path):
    root = tmp_path / "unknown"
    root.mkdir()
    with sqlite3.connect(root / "catalog.sqlite") as connection:
        connection.execute("CREATE TABLE valuable_data (value TEXT)")
        connection.execute("INSERT INTO valuable_data VALUES ('keep me')")
    with pytest.raises(ValueError, match="不会自动修改"):
        LocalDocumentStore(root)
    with sqlite3.connect(root / "catalog.sqlite") as connection:
        assert connection.execute("SELECT value FROM valuable_data").fetchone()[0] == "keep me"


@pytest.mark.parametrize("relative", ["../outside", "objects/../../outside", "C:/outside"])
def test_storage_paths_cannot_escape_data_root(tmp_path, relative):
    with pytest.raises(StorageError):
        safe_path(tmp_path, relative)


def test_failed_parse_has_persistent_job_and_original_but_no_active_index(store):
    with pytest.raises(DocumentParseError):
        store.ingest(b"%PDF-broken", "broken.pdf")
    record = store.list_documents()[0]
    assert record["status"] == "failed" and record["index_id"] is None
    assert store.list_jobs()[0]["error_code"] == "corrupt_pdf"


def test_cached_retriever_remains_scoped_to_its_own_document(store, embedding):
    document, _, cached = indexed(store, embedding)
    other = store.ingest(b"Other document with different evidence.", "other.txt").document
    with pytest.raises(ValueError, match="当前文档"):
        cached.search("evidence", other.document_id)


def test_wrong_document_parse_pointer_is_rejected(store):
    first = store.ingest(TEXT, "first.md").document
    other = store.ingest(b"Different original text.", "other.txt").document
    with write_session(store.engine) as session:
        session.get(DocumentRecord, first.document_id).current_parse_id = other.document_version
    with pytest.raises(StorageError, match="原始文档记录"):
        store.get_document(first.document_id)
