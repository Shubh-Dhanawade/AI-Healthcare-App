import asyncio
import re
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.services.chat_service import generate_smart_fallback_answer

async def main():
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == '7fd1d803-bd6d-4be4-9e9b-3c915022ba16').options(
            selectinload(Document.summary),
            selectinload(Document.extracted_fields)
        )
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc:
        print("Document not found.")
        return
        
    policies_data = [{
        "id": doc.id,
        "filename": doc.original_filename,
        "text": doc.extracted_text or "",
        "summary": {
            "summary_text": doc.summary.summary_text if doc.summary else "",
            "premium_summary": doc.summary.premium_summary if doc.summary else "",
            "coverage_summary": doc.summary.coverage_summary if doc.summary else "",
            "exclusions_summary": doc.summary.exclusions_summary if doc.summary else "",
            "waiting_period_summary": doc.summary.waiting_period_summary if doc.summary else "",
        },
        "extracted_fields": [
            {"field_name": f.field_name, "field_value": f.field_value}
            for f in doc.extracted_fields
        ]
    }]
    
    print("\n--- DATABASE FIELDS ---")
    for f in doc.extracted_fields:
        safe_val = str(f.field_value).encode('ascii', errors='replace').decode('ascii')
        print(f"  {f.field_name}: {safe_val}")
        
    test_queries = [
        "What is the sum insured?",
        "How much is the premium?",
        "Who is covered under this policy?",
        "What are the waiting periods?",
        "Are there any exclusions or room rent limits?",
        "When does the policy renew?",
    ]
    
    print("\n--- FALLBACK QA ENGINE TESTING ---")
    for q in test_queries:
        ans = generate_smart_fallback_answer(q, policies_data)
        safe_ans = ans.encode('ascii', errors='replace').decode('ascii')
        print(f"\nQuery: '{q}'")
        print(f"Answer:\n{safe_ans}")

if __name__ == "__main__":
    asyncio.run(main())
