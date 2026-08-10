import asyncio
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select
from app.services.vector_store import search_vector_store

async def main():
    target_doc_id = 'b65098e2-fb83-4cdb-8268-c93128a306c5'
    
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == target_doc_id)
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc:
        print("Document not found.")
        return
        
    print(f"\nTesting Qdrant search for document: {doc.original_filename} ({doc.id})")
    
    policies = [{"id": doc.id, "filename": doc.original_filename}]
    
    queries = [
        "is this policy covers dental treatment",
        "dental treatment",
        "dental",
    ]
    
    async with AsyncSessionLocal() as db:
        for q in queries:
            print(f"\n--- QUERY: '{q}' ---")
            hits = await search_vector_store(db, q, policies, top_k=6)
            print(f"Retrieved {len(hits)} hits:")
            for idx, h in enumerate(hits):
                safe_text = h['text'].replace('\n', ' ').encode('ascii', errors='replace').decode('ascii')
                print(f"  Hit {idx+1} (Page {h['page']}, Score: {h['score']:.4f}):")
                print(f"    Text: {safe_text[:300]}...")

if __name__ == "__main__":
    asyncio.run(main())
