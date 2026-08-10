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
        
    text = doc.extracted_text
    print(f"Loaded document: {doc.original_filename}")
    
    # Find all occurrences of "sum" or "insured" case-insensitively, printing the surrounding 50 characters
    print("\n--- OCCURRENCES OF SUM/INSURED/LAKH ---")
    matches = re.finditer(r'(?:sum|insured|lakh|10,00,000|10,000,000|1,000,000|1,00,00,000)', text, re.IGNORECASE)
    seen_snippets = set()
    for m in matches:
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        snippet = text[start:end].replace('\n', ' ')
        if snippet not in seen_snippets:
            print(f"Pos {m.start()}: ... {snippet} ...")
            seen_snippets.add(snippet)

if __name__ == "__main__":
    asyncio.run(main())
