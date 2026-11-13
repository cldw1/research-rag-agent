"""SQLAlchemy 2.0 catalog; vectors are external immutable artifacts for now."""

from sqlalchemy import JSON, ForeignKey, ForeignKeyConstraint, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SchemaVersion(Base):
    __tablename__ = "schema_version"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)


class DocumentRecord(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(68), primary_key=True)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True)
    filename: Mapped[str] = mapped_column(Text)
    file_type: Mapped[str] = mapped_column(String(10))
    file_size: Mapped[int] = mapped_column(Integer)
    object_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(24), default="processing")
    # Service transactions enforce pointers. No circular FK creation/migration.
    current_parse_id: Mapped[str | None] = mapped_column(String(64))
    active_index_id: Mapped[str | None] = mapped_column(String(32))
    processing_job_id: Mapped[str | None] = mapped_column(String(32))
    deleted_at: Mapped[str | None] = mapped_column(String(40))


class ParsedVersion(Base):
    __tablename__ = "document_versions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    request_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    parser_version: Mapped[str] = mapped_column(String(40))
    parse_options: Mapped[dict] = mapped_column(JSON)
    structured_document: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(40))


class IndexVersion(Base):
    __tablename__ = "index_versions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    parse_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id"))
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    chunk_config: Mapped[dict] = mapped_column(JSON)
    embedding_profile: Mapped[dict] = mapped_column(JSON)
    artifact_path: Mapped[str] = mapped_column(Text)
    manifest: Mapped[dict] = mapped_column(JSON)
    chunk_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(String(40))


class ChunkRecord(Base):
    __tablename__ = "chunks"
    index_id: Mapped[str] = mapped_column(ForeignKey("index_versions.id"), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(Text)
    data: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("index_id", "chunk_id"),)


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(24))
    phase: Mapped[str] = mapped_column(String(24))
    progress: Mapped[int] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[str] = mapped_column(String(40))
    updated_at: Mapped[str] = mapped_column(String(40))
    lease_until: Mapped[str] = mapped_column(String(40))


class ConversationRecord(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    index_id: Mapped[str] = mapped_column(ForeignKey("index_versions.id"))
    title: Mapped[str] = mapped_column(Text)
    model_config: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(40))
    updated_at: Mapped[str] = mapped_column(String(40))
    deleted_at: Mapped[str | None] = mapped_column(String(40))
    active_turn_id: Mapped[str | None] = mapped_column(String(32))


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    request_id: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24))
    error_code: Mapped[str | None] = mapped_column(String(64))
    model_config: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[str] = mapped_column(String(40))
    updated_at: Mapped[str] = mapped_column(String(40))
    lease_until: Mapped[str] = mapped_column(String(40))
    __table_args__ = (UniqueConstraint("conversation_id", "ordinal"),
                      UniqueConstraint("conversation_id", "request_id"))


class MessageRecord(Base):
    __tablename__ = "messages"
    turn_id: Mapped[str] = mapped_column(ForeignKey("conversation_turns.id"), primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(String(20))
    data: Mapped[dict] = mapped_column(JSON)


class CitationRecord(Base):
    __tablename__ = "citations"
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), primary_key=True)
    number: Mapped[int] = mapped_column(Integer, primary_key=True)
    index_id: Mapped[str] = mapped_column(String(32))
    chunk_id: Mapped[str] = mapped_column(Text)
    source: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (
        ForeignKeyConstraint(["index_id", "chunk_id"], ["chunks.index_id", "chunks.chunk_id"]),
        UniqueConstraint("conversation_id", "index_id", "chunk_id"),
    )
