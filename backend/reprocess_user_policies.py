import asyncio
import os
import sys
from loguru import logger

# Ensure backend directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import select
from app.core.database import AsyncSessionLocal, engine
from app.models.document import Document
from app.api.v1.documents import process_document_background, _analysis_in_progress

async def main():
    logger.info("Starting reprocessing of compared policies...")
    
    target_ids = [
        "b25bd29f-19b1-4f6d-901b-99fb426946e1",
        "96613107-386c-43ec-a8a9-a1304e4d7533"
    ]
    
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Document).where(Document.id.in_(target_ids))
        )
        docs = result.scalars().all()
        
    logger.info(f"Found {len(docs)} document(s) to re-process.")
    
    for doc in docs:
        logger.info(f"Reprocessing document: {doc.original_filename} (ID: {doc.id})")
        # Run the full background processing task (extraction + AI analysis)
        await process_document_background(doc.id, doc.file_path, doc.file_type)
        
        # Wait for the spawned background tasks to complete (fields and risks are detached asyncio tasks)
        logger.info("Waiting for background AI summarization, field extraction, and risk analysis to complete...")
        while any(k.endswith(doc.id) for k in _analysis_in_progress):
            await asyncio.sleep(1.0)
            
        logger.info(f"Finished reprocessing: {doc.original_filename}")
        
    await engine.dispose()
    logger.info("Done!")

if __name__ == "__main__":
    asyncio.run(main())
