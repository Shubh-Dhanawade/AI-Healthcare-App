"""RAGQueryLog database model — compatible with SQLite and PostgreSQL."""

import uuid
from datetime import datetime
from sqlalchemy import String, Text, ForeignKey, DateTime, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, utc_now_naive


class RAGQueryLog(Base):
    __tablename__ = "rag_query_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    
    # Query and answer contents
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)

    # LLM-as-a-judge scores
    faithfulness: Mapped[float] = mapped_column(Float, default=1.0)
    faithfulness_reasoning: Mapped[str] = mapped_column(Text, nullable=True)
    answer_relevance: Mapped[float] = mapped_column(Float, default=1.0)
    answer_relevance_reasoning: Mapped[str] = mapped_column(Text, nullable=True)
    context_relevance: Mapped[float] = mapped_column(Float, default=1.0)
    latency: Mapped[float] = mapped_column(Float, default=0.0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now_naive
    )

    # Relationship
    user: Mapped["User"] = relationship("User") # noqa

    def __repr__(self):
        return f"<RAGQueryLog {self.query[:30]}... -> {self.faithfulness:.2f}>"
