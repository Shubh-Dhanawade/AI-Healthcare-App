import asyncio
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.services.chat_service import prepare_chat_rag_prompt

async def main():
    target_doc_id = 'b65098e2-fb83-4cdb-8268-c93128a306c5'
    
    async with AsyncSessionLocal() as db:
        # Load document along with its summary and extracted_fields
        stmt = select(Document).where(Document.id == target_doc_id).options(
            selectinload(Document.summary),
            selectinload(Document.extracted_fields)
        )
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
    if not doc:
        print("Document not found.")
        return
        
    # Package policy data
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
    
    async with AsyncSessionLocal() as db:
        prompt, filtered_chunks, is_short = await prepare_chat_rag_prompt(
            policies=policies_data,
            query="is this policy covers dental treatment",
            db=db,
            history=[],
            user_name="Monu"
        )
        
    print("\n--- GENERATED PROMPT ---")
    safe_prompt = prompt.encode('ascii', errors='replace').decode('ascii')
    print(safe_prompt)

if __name__ == "__main__":
    asyncio.run(main())
