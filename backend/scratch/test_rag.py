import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import AsyncSessionLocal
from app.services.vector_store import search_vector_store
from app.models.document import Document

async def main():
    query = "Is this policy covers maternity benefits tell me"
    print(f"Query: '{query}'")
    
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Document.id, Document.original_filename))
        docs = res.all()
        if not docs:
            print("No documents found in DB!")
            return
            
        doc_id = docs[0][0]
        filename = docs[0][1]
        print(f"Testing against Document: {filename} (id={doc_id})")
        
        policies = [{"id": doc_id, "filename": filename}]
        
        # Search vector store
        chunks = await search_vector_store(db, query, policies, top_k=6)
        print(f"Found {len(chunks)} chunks:")
        for idx, c in enumerate(chunks):
            # Clean non-ascii characters for clean terminal logging on Windows
            clean_text = c['text'].encode('ascii', 'ignore').decode('ascii')
            print(f"\n[{idx+1}] Score: {c['score']:.4f} (Raw: {c.get('raw_score', 0.0):.4f}) | Source: {c['source']} page {c['page']}")
            print(f"Snippet: {clean_text[:250]}...")

if __name__ == "__main__":
    from sqlalchemy import select
    asyncio.run(main())
