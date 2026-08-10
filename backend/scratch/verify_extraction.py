import asyncio
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select
from app.services.ai_service import extract_policy_fields

async def main():
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == '7fd1d803-bd6d-4be4-9e9b-3c915022ba16')
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc:
        print("Document not found.")
        return
        
    print(f"Running extract_policy_fields on document: {doc.original_filename}...")
    
    # We call it with is_ocr=True since it's a PDF / scanned document
    fields = await extract_policy_fields(doc.extracted_text, force_regenerate=True, is_ocr=True)
    
    print("\n--- EXTRACTED FIELDS ---")
    for f in fields:
        print(f"{f['field_name']}: {f['field_value']} ({f['field_category']})")

if __name__ == "__main__":
    asyncio.run(main())
