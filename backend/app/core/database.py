"""
Database Configuration — supports SQLite (dev) and PostgreSQL (prod)
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from loguru import logger

from app.core.config import settings


def _get_engine_kwargs():
    """Return engine kwargs appropriate for the database type."""
    url = settings.DATABASE_URL
    if url.startswith("sqlite"):
        # SQLite requires these settings for async
        return {"connect_args": {"check_same_thread": False, "timeout": 30}}
    else:
        # PostgreSQL supports connection pooling
        return {"pool_size": 5, "max_overflow": 10, "pool_pre_ping": True}


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    **_get_engine_kwargs(),
)

from sqlalchemy import event

@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Optimize SQLite performance using pragmas."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA busy_timeout=5000")
    except Exception as e:
        logger.warning(f"Failed to set SQLite PRAGMAs: {e}")
    finally:
        cursor.close()

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


async def create_tables():
    """Create all database tables on startup."""
    from app.models import user, document, summary, risk_analysis  # noqa
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables created/verified")


async def get_db() -> AsyncSession:
    """Dependency to get a database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
