import asyncio
import re
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select

async def main():
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == '7fd1d803-bd6d-4be4-9e9b-3c915022ba16')
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc:
        print("Document not found.")
        return
        
    text_clean = re.sub(r'\s+', ' ', doc.extracted_text)
    
    # 1. Match diabetes
    m1 = re.search(r'\b([a-zA-Z\s\-()]{0,50}diabetes(?:\s+mellitus)?)\b', text_clean, re.IGNORECASE)
    d1 = m1.group(1).strip() if m1 else "None"
    
    # 2. Match hypertension
    m2 = re.search(r'\b([a-zA-Z\s\-()]{0,50}hypertension)\b', text_clean, re.IGNORECASE)
    d2 = m2.group(1).strip() if m2 else "None"
    
    print(f"Diabetes Extracted: '{d1}'")
    print(f"Hypertension Extracted: '{d2}'")

if __name__ == "__main__":
    asyncio.run(main())
