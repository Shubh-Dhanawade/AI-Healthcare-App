import asyncio
import sys
import os

# Make sure the backend app package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from loguru import logger

from app.core.config import settings
from app.models.document import Document
from app.api.v1.documents import _run_summary_background, _run_fields_background, _run_risks_background

async def main() -> None:
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Document).where(Document.extracted_text.isnot(None))
        )
        docs = result.scalars().all()

    logger.info(f"Found {len(docs)} documents to re-process with the real LLM.")

    for idx, doc in enumerate(docs, 1):
        logger.info(f"[{idx}/{len(docs)}] Reprocessing: {doc.original_filename} ({doc.id})")
        try:
            # 1. Summary
            await _run_summary_background(doc.id, force_regenerate=True)
            
            # 2. Fields
            await _run_fields_background(doc.id, force_regenerate=True)
            
            # 3. Risks
            await _run_risks_background(doc.id, force_regenerate=True)
            
            logger.info(f"[{idx}/{len(docs)}] Finished reprocessing: {doc.original_filename}")
        except Exception as e:
            logger.error(f"[{idx}/{len(docs)}] Failed: {doc.id} - {e}")

    logger.info("Finished: All documents re-processed successfully with the real LLM.")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
