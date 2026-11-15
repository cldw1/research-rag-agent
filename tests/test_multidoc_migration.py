from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, event, inspect

from src.chat_service import ResearchChatService
from src.storage.artifacts import StorageError
from src.storage.backup import restore_backup, verify_backup
from src.storage.conversations import ConversationStore
from src.storage.database import V2_TABLES, catalog_version, upgrade_multidoc_schema
from src.storage.document_store import LocalDocumentStore
from src.storage.models import Base, SchemaVersion
from tests.chat_helpers import make_resources, tool_answer


def v2_library(root):
    root.mkdir()
    engine = create_engine("sqlite:///" + (root / "catalog.sqlite").as_posix())
    with engine.begin() as connection:
        Base.metadata.create_all(connection, tables=[Base.metadata.tables[name] for name in V2_TABLES])
        connection.execute(SchemaVersion.__table__.insert().values(version=2))
    engine.dispose()
    return make_resources(root)


def test_explicit_upgrade_preserves_old_messages_citations_and_restorable_v2_backup(tmp_path):
    r = v2_library(tmp_path / "old")
    try:
        cid = r.conversations.create(r.document.document_id, r.config)
        ResearchChatService(r.conversations).ask(cid, "question", tool_answer(r), r.retriever, "first")
        before, sources = r.conversations.history(cid), r.conversations.sources(cid)
        assert catalog_version(r.store.engine) == 2
        with pytest.raises(StorageError, match="多文档"):
            r.conversations.create([r.document.document_id, "another"], r.config)
        assert "conversation_documents" not in inspect(r.store.engine).get_table_names()
        backup = upgrade_multidoc_schema(r.store)
        assert catalog_version(r.store.engine) == 3 and verify_backup(backup)
        assert upgrade_multidoc_schema(r.store) is None
        upgraded = ConversationStore(r.store)
        assert upgraded.history(cid) == before and upgraded.sources(cid) == sources
        assert upgraded.details(cid)["documents"][0]["index_id"] == r.retriever.index_id
        restore_backup(backup, tmp_path / "restored-v2")
        restored = LocalDocumentStore(tmp_path / "restored-v2")
        try:
            assert catalog_version(restored.engine) == 2
            old = ConversationStore(restored)
            assert old.history(cid) == before and old.sources(cid) == sources
        finally:
            restored.close()
    finally:
        r.store.close()


def test_backup_failure_leaves_v2_schema_and_messages_unchanged(tmp_path, monkeypatch):
    from src.storage import backup
    r = v2_library(tmp_path / "old")
    try:
        monkeypatch.setattr(backup, "create_backup", Mock(side_effect=OSError("disk full")))
        with pytest.raises(OSError):
            upgrade_multidoc_schema(r.store)
        assert catalog_version(r.store.engine) == 2
        assert "conversation_documents" not in inspect(r.store.engine).get_table_names()
    finally:
        r.store.close()


def test_failed_scope_backfill_rolls_back_new_table_and_schema_version(tmp_path):
    r = v2_library(tmp_path / "old")
    try:
        cid = r.conversations.create(r.document.document_id, r.config)
        def fail_backfill(conn, cursor, statement, parameters, context, executemany):
            if statement.startswith("INSERT INTO conversation_documents"):
                raise RuntimeError("simulated backfill failure")
        event.listen(r.store.engine, "before_cursor_execute", fail_backfill)
        try:
            with pytest.raises(RuntimeError, match="backfill"):
                upgrade_multidoc_schema(r.store)
        finally:
            event.remove(r.store.engine, "before_cursor_execute", fail_backfill)
        assert catalog_version(r.store.engine) == 2
        assert "conversation_documents" not in inspect(r.store.engine).get_table_names()
        assert r.conversations.details(cid)["document_id"] == r.document.document_id
    finally:
        r.store.close()
