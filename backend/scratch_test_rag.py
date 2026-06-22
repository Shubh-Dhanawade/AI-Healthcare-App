import asyncio
import os
import sys
from sqlalchemy import select

# Add backend directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import AsyncSessionLocal
from app.models.document import Document
from app.services.vector_store import search_vector_store

async def main():
    async with AsyncSessionLocal() as db:
        # Fetch completed documents
        result = await db.execute(select(Document).where(Document.status == "completed"))
        docs = result.scalars().all()
        if not docs:
            print("No completed documents found in database.")
            return
            
        policies = [
            {
                "id": d.id,
                "filename": d.original_filename
            }
            for d in docs
        ]
        
        print(f"Loaded {len(policies)} completed policies.")
        for p in policies:
            print(f" - {p['filename']} (ID: {p['id']})")
            
        # Test queries
        queries = [
            "What is the room rent limit?",
            "Are pre-existing conditions covered?",
            "What is a deductible?",
            "who are you?",
            "explain what you just said"
        ]
        
        for q in queries:
            print(f"\nQuery: '{q}'")
            hits = await search_vector_store(db, q, policies, top_k=4)
            print(f"Retrieved {len(hits)} hits:")
            for i, hit in enumerate(hits):
                print(f"  {i+1}. Score: {hit['score']:.4f} | Source: {hit['source']} | Page: {hit['page']}")
                print(f"     Snippet: {hit['text'][:120]}...")

if __name__ == "__main__":
    asyncio.run(main())
