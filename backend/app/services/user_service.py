"""User service layer for database operations."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from app.models.user import User
from app.core.security import get_password_hash, verify_password
from app.schemas.schemas import UserRegister


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Fetch a user by email address."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    """Fetch a user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, data: UserRegister) -> User:
    """Create a new user account."""
    hashed_pw = get_password_hash(data.password)
    user = User(
        email=data.email,
        full_name=data.full_name,
        hashed_password=hashed_pw,
        role=data.role or "user",
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]:
    """Verify user credentials."""
    user = await get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
