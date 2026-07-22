import asyncio
import os
import sys
import uuid
import json

# Add backend app directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import AsyncSessionLocal, create_tables, HAS_PGVECTOR
from app.models.document import Document, DocumentChunk
from app.services.vector_store import (
    index_chunks_in_vector_stores,
    search_vector_store,
    get_qdrant_client,
    init_qdrant_collection
)
from app.services.chat_service import classify_question, run_chat_query
from sqlalchemy import select

async def main():
    print("[START] Running Hybrid Retrieval Verification Script...")
    
    # 1. Initialize Tables & Migrations
    print("Database migration check...")
    await create_tables()
    print(f"pgvector extension status (HAS_PGVECTOR): {HAS_PGVECTOR}")
    
    # 2. Check Qdrant connection
    print("Checking Qdrant client connection...")
    client = get_qdrant_client()
    if client:
        print("CONNECTED: Connected to Qdrant!")
        init_qdrant_collection()
    else:
        print("WARNING: Qdrant not connected/running. Verification will run on pgvector/FAISS fallback mode.")
        
    # 3. Fetch or Create test document
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(Document))
        doc = res.scalars().first()
        if not doc:
            print("No policies found in DB. Creating dummy policy for validation...")
            doc = Document(
                original_filename="Test_Health_Policy.pdf",
                stored_filename="test_policy.pdf",
                file_path="./uploads/test_policy.pdf",
                file_type="pdf",
                file_size_bytes=1024,
                status="completed",
                extracted_text="This is a test health policy. Maternity benefits are excluded under Excl18."
            )
            db.add(doc)
            await db.commit()
            await db.refresh(doc)
            
        print(f"Validating against Document: {doc.original_filename} (id={doc.id})")
        
        # 4. Create sample chunks and index
        print("Building test chunks...")
        test_vector = [0.01] * 768  # 768 dimensions for nomic-embed-text
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            text_content="Policy Exclusions: Cataract surgery is limited to $500. Maternity childbirth is excluded under Excl18.",
            embedding=json.dumps(test_vector),
            embedding_vector=test_vector
        )
        db.add(chunk)
        await db.commit()
        await db.refresh(chunk)
        
        print("Indexing chunks to FAISS, pgvector, and Qdrant...")
        await index_chunks_in_vector_stores(db, doc.user_id, doc.id, [chunk])
        
        # 5. Verify Classifier Routing
        print("\n--- Verifying Question Classifier ---")
        q1 = "What is my policy number?"
        q2 = "Does this cover cataract surgery?"
        q3 = "Summarize my policy coverage exclusions."
        
        print(f"Query: '{q1}' -> Classified Category: {classify_question(q1)}")
        print(f"Query: '{q2}' -> Classified Category: {classify_question(q2)}")
        print(f"Query: '{q3}' -> Classified Category: {classify_question(q3)}")
        
        # 6. Verify Search Retrieval Gateway
        print("\n--- Testing Retrieval Gateway ---")
        search_query = "Does this cover cataract surgery?"
        policies = [{"id": doc.id, "filename": doc.original_filename}]
        hits = await search_vector_store(db, search_query, policies, top_k=2)
        print(f"Search Query: '{search_query}' | Matches found: {len(hits)}")
        for i, h in enumerate(hits):
            print(f"  [{i+1}] Score: {h['score']:.4f} (Source: {h['source']} Page {h['page']})")
            print(f"      Snippet: {h['text'][:120]}...")
            
        # 7. Test complete chat query engine
        print("\n--- Testing Chat Router Execution ---")
        chat_history = []
        user_name = "VerificationUser"
        
        response = await run_chat_query(
            policies=policies,
            query="Is maternity covered under my policy?",
            db=db,
            history=chat_history,
            user_name=user_name,
            user_id=doc.user_id
        )
        print("\nChat Assistant Response:")
        print(response)
        
    print("\n[SUCCESS] Verification script complete!")

if __name__ == "__main__":
    asyncio.run(main())
