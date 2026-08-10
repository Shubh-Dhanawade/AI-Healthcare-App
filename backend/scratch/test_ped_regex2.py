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
    
    print("ALL MATCHES FOR WAITING PERIOD IN NORMALIZED TEXT:")
    matches = re.finditer(r'waiting period', text_clean, re.IGNORECASE)
    for m in matches:
        start = max(0, m.start() - 30)
        end = min(len(text_clean), m.end() + 70)
        snippet = text_clean[start:end].encode('ascii', errors='replace').decode('ascii')
        print(f"Match at {m.start()}: '{snippet}'")

if __name__ == "__main__":
    asyncio.run(main())
