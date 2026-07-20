import asyncio
import os
import sys

# Add backend app directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import AsyncSessionLocal
from app.models.document import Document, DocumentChunk
from sqlalchemy import select

async def main():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Document.id, Document.original_filename))
        docs = res.all()
        print("Documents in DB:")
        for d in docs:
            print(f"- ID: {d[0]}, Filename: {d[1]}")
            
        res_chunks = await db.execute(select(DocumentChunk.document_id))
        chunks = res_chunks.scalars().all()
        print(f"Total chunks in DB: {len(chunks)}")
        
        for d in docs:
            doc_chunks = [c for c in chunks if c == d[0]]
            print(f"  * Document {d[1]} has {len(doc_chunks)} chunks.")

if __name__ == "__main__":
    asyncio.run(main())
