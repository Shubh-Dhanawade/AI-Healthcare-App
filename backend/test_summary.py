import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from app.models.document import Document
from app.services.ai_service import generate_summary
from app.core.config import settings
import json

async def main():
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as db:
        # Get the document from the screenshot's URL
        doc_id = "9c61c1a1-e32d-46d0-a431-b38c252da381"
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        
        if not doc:
            print("Document not found in DB.")
            return
            
        print("Document found. Generating summary with updated prompts...")
        summary = await generate_summary(doc.extracted_text, force_regenerate=True)
        print("========== PROSE SUMMARY ==========")
        print(summary['summary_text'])
        print("\n========== BULLET SUMMARIES ==========")
        print("COVERAGE:")
        print(summary['coverage_summary'])
        print("EXCLUSIONS:")
        print(summary['exclusions_summary'])
        print("WAITING PERIODS:")
        print(summary['waiting_period_summary'])
        print("PREMIUM:")
        print(summary['premium_summary'])

if __name__ == "__main__":
    asyncio.run(main())
