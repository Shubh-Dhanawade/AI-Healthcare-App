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
    # Only execute pragmas if the database is SQLite
    if "sqlite" not in settings.DATABASE_URL:
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA busy_timeout=30000")
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
    from app.models import user, document, summary, risk_analysis, rag_query_log, chat  # noqa
    
    # 1. Create tables inside a dedicated transaction block
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables structure verified")
        
    # 2. Run column migrations in a separate transaction block (avoids transaction aborts on duplicates)
    async with engine.begin() as conn:
        if "sqlite" in settings.DATABASE_URL:
            try:
                from sqlalchemy import text
                await conn.execute(text("ALTER TABLE documents ADD COLUMN file_hash VARCHAR(64)"))
                logger.info("✅ Dynamically added file_hash column to documents table (SQLite)")
            except Exception:
                pass
        else:
            try:
                from sqlalchemy import text
                # Query schema safely in PostgreSQL to see if file_hash exists
                res = await conn.execute(text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name='documents' AND column_name='file_hash'"
                ))
                if not res.scalar():
                    await conn.execute(text("ALTER TABLE documents ADD COLUMN file_hash VARCHAR(64)"))
                    logger.info("✅ Dynamically added file_hash column to documents table (PostgreSQL)")
            except Exception as e:
                logger.warning(f"Failed to check or alter documents table in PostgreSQL: {e}")

    # 3. Populate file_hash values in a third transaction block
    async with engine.begin() as conn:
        try:
            from sqlalchemy import text
            import hashlib
            import os
            
            result = await conn.execute(text("SELECT id, file_path FROM documents WHERE file_hash IS NULL"))
            rows = result.all()
            for row in rows:
                doc_id, file_path = row
                if file_path and os.path.exists(file_path):
                    try:
                        with open(file_path, "rb") as f:
                            content = f.read()
                        h = hashlib.sha256(content).hexdigest()
                        await conn.execute(
                            text("UPDATE documents SET file_hash = :file_hash WHERE id = :id"),
                            {"file_hash": h, "id": doc_id}
                        )
                        logger.info(f"✅ Populated file_hash for existing document {doc_id}")
                    except Exception as fe:
                        logger.warning(f"Failed to hash existing file {file_path}: {fe}")
        except Exception as e:
            logger.warning(f"Failed to migrate existing document hashes: {e}")
            
    logger.info("✅ Database tables created/verified")


def utc_now_naive():
    """Return a timezone-naive UTC datetime object."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
