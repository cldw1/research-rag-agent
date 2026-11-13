"""Additive initial schema only. Refuse unknown schemas instead of guessing."""

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session

from src.storage.models import Base, SchemaVersion


SCHEMA_VERSION = 2
V1_TABLES = {"schema_version", "documents", "document_versions", "index_versions", "chunks", "ingestion_jobs"}
CHAT_TABLES = {"conversations", "conversation_turns", "messages", "citations"}


def open_catalog(path: Path):
    engine = create_engine(URL.create("sqlite", database=str(path)),
                           connect_args={"timeout": 15, "check_same_thread": False})

    @event.listens_for(engine, "connect")
    def configure_sqlite(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")

    # Initialization is also serialized across Streamlit sessions/processes.
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            names = set(inspect(connection).get_table_names())
            if not names:
                Base.metadata.create_all(connection)
                connection.execute(SchemaVersion.__table__.insert().values(version=SCHEMA_VERSION))
            else:
                if "schema_version" not in names:
                    raise ValueError("数据库不是受支持的文档目录；不会自动修改现有表")
                versions = connection.execute(select(SchemaVersion.version)).scalars().all()
                expected = V1_TABLES if versions == [1] else set(Base.metadata.tables)
                if versions not in ([1], [SCHEMA_VERSION]) or not expected.issubset(names):
                    raise ValueError("数据库结构版本不匹配；请先备份并使用显式迁移")
            connection.commit()
    except Exception:
        engine.dispose()
        raise
    return engine


def catalog_version(engine):
    with engine.connect() as connection:
        return connection.execute(select(SchemaVersion.version)).scalar_one()


def upgrade_chat_schema(store):
    """Explicit additive v1 -> v2, with a full backup before touching the catalog."""
    import uuid
    from src.storage.backup import create_backup
    from src.storage.artifacts import StorageError

    version = catalog_version(store.engine)
    if version == SCHEMA_VERSION:
        return None
    if version != 1:
        raise StorageError("schema_upgrade_required", "不支持该数据库版本的自动升级")
    backup = store.root.parent / "catalog_backups" / (store.root.name + "-v1-" + uuid.uuid4().hex)
    create_backup(store, backup)
    with store.engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            if connection.execute(select(SchemaVersion.version)).scalar_one() == SCHEMA_VERSION:
                connection.rollback()
                return backup
            if set(inspect(connection).get_table_names()) != V1_TABLES:
                raise StorageError("schema_mismatch", "现有表与已知 v1 结构不一致，未执行升级")
            Base.metadata.create_all(connection, tables=[Base.metadata.tables[name] for name in CHAT_TABLES])
            connection.execute(SchemaVersion.__table__.update().values(version=SCHEMA_VERSION))
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return backup


@contextmanager
def write_session(engine):
    # Serialize claims and pointer changes, never expensive parsing/embedding.
    with Session(engine, expire_on_commit=False) as session:
        try:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
