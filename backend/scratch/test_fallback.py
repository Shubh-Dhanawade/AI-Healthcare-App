import asyncio
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select
from app.services.ai_service import _build_fallback_fields, _build_fallback_summary

async def main():
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == '7fd1d803-bd6d-4be4-9e9b-3c915022ba16')
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc:
        print("Document not found.")
        return
        
    print(f"Running fallback extraction on document: {doc.original_filename}...")
    
    fields = _build_fallback_fields(doc.extracted_text)
    print("\n--- FALLBACK FIELDS ---")
    for f in fields:
        print(f"{f['field_name']}: {f['field_value']}")
        
    print("\n--- FALLBACK SUMMARY ---")
    summary = _build_fallback_summary(doc.extracted_text)
    print(f"Summary Text:\n{summary['summary_text'][:500]}...")
    print(f"\nWaiting Period Summary:\n{summary['waiting_period_summary']}")

if __name__ == "__main__":
    asyncio.run(main())
