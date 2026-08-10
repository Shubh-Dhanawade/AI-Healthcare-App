import asyncio
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select

async def main():
    target_doc_id = 'b65098e2-fb83-4cdb-8268-c93128a306c5'
    
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == target_doc_id)
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc or not doc.extracted_text:
        print("Document not found or has no text.")
        return
        
    text = doc.extracted_text
    print(f"Document text length: {len(text)}")
    
    import re
    matches = list(re.finditer(r'dental', text, re.IGNORECASE))
    print(f"Found {len(matches)} occurrences of 'dental':")
    for idx, m in enumerate(matches):
        start = max(0, m.start() - 150)
        end = min(len(text), m.end() + 150)
        snippet = text[start:end].replace('\n', ' ')
        print(f"\nMatch {idx+1} (char {m.start()}-{m.end()}):")
        print(f"  ... {snippet} ...")

if __name__ == "__main__":
    asyncio.run(main())
