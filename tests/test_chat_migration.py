from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, inspect

from src.llm.provider import ModelConfig
from src.storage.artifacts import StorageError
from src.storage.backup import restore_backup, verify_backup
from src.storage.conversations import ConversationStore
from src.storage.database import CHAT_TABLES, V1_TABLES, catalog_version, upgrade_chat_schema
from src.storage.document_store import LocalDocumentStore
from src.storage.models import Base, SchemaVersion


def old_catalog(root):
    root.mkdir()
    engine = create_engine("sqlite:///" + (root / "catalog.sqlite").as_posix())
    with engine.begin() as connection:
        Base.metadata.create_all(connection, tables=[Base.metadata.tables[name] for name in V1_TABLES])
        connection.execute(SchemaVersion.__table__.insert().values(version=1))
    engine.dispose()
    return LocalDocumentStore(root)


def test_additive_upgrade_backs_up_preserves_documents_and_is_idempotent(tmp_path):
    store = old_catalog(tmp_path / "v1")
    try:
        doc = store.ingest(b"Existing original paper.", "original.txt").document
        with pytest.raises(StorageError, match="升级聊天存储"):
            ConversationStore(store)
        backup = upgrade_chat_schema(store)
        assert catalog_version(store.engine) == 2
        assert CHAT_TABLES.issubset(inspect(store.engine).get_table_names())
        assert store.get_document(doc.document_id).to_dict() == doc.to_dict()
        assert verify_backup(backup)["files"]["catalog.sqlite"]
        assert upgrade_chat_schema(store) is None
        restore_backup(backup, tmp_path / "rollback")
        restored = LocalDocumentStore(tmp_path / "rollback")
        try:
            assert catalog_version(restored.engine) == 1
            assert restored.get_document(doc.document_id).to_dict() == doc.to_dict()
        finally:
            restored.close()
    finally:
        store.close()


def test_failed_backup_leaves_old_schema_untouched(tmp_path, monkeypatch):
    from src.storage import backup
    store = old_catalog(tmp_path / "v1")
    try:
        monkeypatch.setattr(backup, "create_backup", Mock(side_effect=OSError("disk full")))
        with pytest.raises(OSError):
            upgrade_chat_schema(store)
        assert catalog_version(store.engine) == 1
        assert not CHAT_TABLES.intersection(inspect(store.engine).get_table_names())
    finally:
        store.close()
