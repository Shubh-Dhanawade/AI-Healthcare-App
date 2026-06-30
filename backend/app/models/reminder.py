"""PolicyReminder database model for scheduling notifications."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class PolicyReminder(Base):
    __tablename__ = "policy_reminders"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True, index=True
    )

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    reminder_type: Mapped[str] = mapped_column(String(50), default="renewal", nullable=False)  # renewal | premium
    reminder_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    premium_amount: Mapped[str] = mapped_column(String(255), nullable=True)
    is_dismissed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="reminders")  # noqa
    document: Mapped["Document"] = relationship("Document", back_populates="reminders")  # noqa

    def __repr__(self):
        return f"<PolicyReminder {self.title} - {self.reminder_date}>"
