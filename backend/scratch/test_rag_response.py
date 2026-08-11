import asyncio
from app.core.database import AsyncSessionLocal
from app.models.document import Document
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.services.chat_service import prepare_chat_rag_prompt
from app.services.ollama_client import call_ollama

async def main():
    target_doc_id = 'b916e230-688a-4fab-8b9c-82f04c580063'
    
    async with AsyncSessionLocal() as db:
        stmt = select(Document).where(Document.id == target_doc_id).options(
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
    
    print("Preparing RAG prompt...")
    async with AsyncSessionLocal() as db:
        prompt, filtered_chunks, is_short = await prepare_chat_rag_prompt(
            policies=policies_data,
            query="is this policy covers dental treatment",
            db=db,
            history=[],
            user_name="Monu"
        )
        
    print("Calling Ollama...")
    try:
        from app.core.config import settings
        response = await call_ollama(prompt, num_predict=200, num_ctx=settings.OLLAMA_NUM_CTX)
        safe_response = response.encode('ascii', errors='replace').decode('ascii')
        print("\n--- LLM RESPONSE ---")
        print(safe_response)
    except Exception as e:
        print(f"Ollama call failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
