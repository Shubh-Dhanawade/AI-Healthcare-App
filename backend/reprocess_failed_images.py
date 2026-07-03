import asyncio
import os
import sys
from loguru import logger

# Ensure backend directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import select
from app.core.database import AsyncSessionLocal, engine
from app.models.document import Document
from app.api.v1.documents import process_document_background

async def main():
    logger.info("Starting reprocessing of failed image uploads...")
    
    async with AsyncSessionLocal() as db:
        # Query all documents where file_type is 'image' and extraction_method was 'unavailable'
        result = await db.execute(
            select(Document).where(
                (Document.file_type == 'image') & 
                (
                    (Document.extraction_method == 'unavailable') | 
                    (Document.extracted_text.contains("requires PaddleOCR")) |
                    (Document.id == '459c256b-c814-48e3-bd8f-00db5b688377')
                )
            )
        )
        docs = result.scalars().all()
        
    logger.info(f"Found {len(docs)} document(s) to re-process.")
    
    from app.api.v1.documents import _analysis_in_progress

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
