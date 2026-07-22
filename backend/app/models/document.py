"""Document and ExtractedField models — SQLite/PostgreSQL compatible."""

import uuid
from datetime import datetime
from sqlalchemy import String, Text, ForeignKey, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import UserDefinedType
from sqlalchemy.ext.compiler import compiles

from app.core.database import Base, utc_now_naive


class SafeVector(UserDefinedType):
    """
    A compile-safe pgvector Vector wrapper.
    Compiles to:
      - 'vector(dim)' for PostgreSQL with vector extension enabled.
      - 'TEXT' for SQLite or databases without pgvector.
    """
    def __init__(self, dim=768):
        self.dim = dim

    def get_col_spec(self, **kw):
        return f"vector({self.dim})"

    def bind_processor(self, dialect):
        from app.core.database import HAS_PGVECTOR
        if dialect.name == "postgresql" and HAS_PGVECTOR:
            return lambda value: value
        import json
        return lambda value: json.dumps(value) if value is not None else None

    def result_processor(self, dialect, coltype):
        from app.core.database import HAS_PGVECTOR
        if dialect.name == "postgresql" and HAS_PGVECTOR:
            return lambda value: value
        import json
        return lambda value: json.loads(value) if value is not None else None


@compiles(SafeVector, "postgresql")
def compile_safe_vector_postgresql(type_, compiler, **kw):
    from app.core.database import HAS_PGVECTOR
    if HAS_PGVECTOR:
        return f"vector({type_.dim})"
    return "TEXT"


@compiles(SafeVector, "sqlite")
def compile_safe_vector_sqlite(type_, compiler, **kw):
    return "TEXT"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # File metadata
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)  # pdf | image
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=True, index=True)

    # Processing status
    status: Mapped[str] = mapped_column(String(30), default="uploaded", nullable=False)
    # uploaded | processing | text_extracted | summarized | completed | failed

    # Extracted text
    extracted_text: Mapped[str] = mapped_column(Text, nullable=True)
    extraction_method: Mapped[str] = mapped_column(String(50), nullable=True)

    # Dates and Scoring
    renewal_date: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    premium_due_date: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    safety_score: Mapped[int] = mapped_column(Integer, default=100, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now_naive
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utc_now_naive,
        onupdate=utc_now_naive,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="documents")  # noqa
    summary: Mapped["Summary"] = relationship(  # noqa
        "Summary", back_populates="document", uselist=False, cascade="all, delete-orphan"
    )
    extracted_fields: Mapped[list["ExtractedField"]] = relationship(  # noqa
        "ExtractedField", back_populates="document", cascade="all, delete-orphan"
    )
    risk_analyses: Mapped[list["RiskAnalysis"]] = relationship(  # noqa
        "RiskAnalysis", back_populates="document", cascade="all, delete-orphan"
    )
    reminders: Mapped[list["PolicyReminder"]] = relationship(  # noqa
        "PolicyReminder", back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(  # noqa
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Document {self.original_filename}>"


class ExtractedField(Base):
    __tablename__ = "extracted_fields"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    field_name: Mapped[str] = mapped_column(String(255), nullable=False)
    field_value: Mapped[str] = mapped_column(Text, nullable=True)
    field_category: Mapped[str] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now_naive
    )

    document: Mapped["Document"] = relationship("Document", back_populates="extracted_fields")  # noqa


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text_content: Mapped[str] = mapped_column(Text, nullable=False)
    # Store embedding floats as serialized JSON string (e.g. "[0.021, -0.043, ...]")
    embedding: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Store embedding as native vector for pgvector, falling back to SafeVector representation
    embedding_vector: Mapped[list[float]] = mapped_column(SafeVector(768), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now_naive
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")  # noqa

