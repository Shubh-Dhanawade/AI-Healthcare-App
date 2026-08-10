import asyncio
from app.core.database import AsyncSessionLocal
from app.models.document import Document, ExtractedField
from app.models.risk_analysis import Summary
from sqlalchemy import select, delete
from app.services.ai_service import _build_fallback_fields, _build_fallback_summary

async def main():
    async with AsyncSessionLocal() as db:
        # 1. Fetch document
        stmt = select(Document).where(Document.id == '7fd1d803-bd6d-4be4-9e9b-3c915022ba16')
        res = await db.execute(stmt)
        doc = res.scalar_one_or_none()
        
        if not doc:
            print("Document not found.")
            return
            
        print(f"Reprocessing document: {doc.original_filename}...")
        
        # 2. Delete old extracted fields and summary
        await db.execute(delete(ExtractedField).where(ExtractedField.document_id == doc.id))
        await db.execute(delete(Summary).where(Summary.document_id == doc.id))
        
        # 3. Extract new fields using our fallback logic
        fields = _build_fallback_fields(doc.extracted_text)
        for f in fields:
            db_field = ExtractedField(
                document_id=doc.id,
                field_name=f['field_name'],
                field_value=f['field_value'],
                field_category=f['field_category']
            )
            db.add(db_field)
            
        # 4. Extract new summary using our fallback logic
        sum_data = _build_fallback_summary(doc.extracted_text)
        db_sum = Summary(
            document_id=doc.id,
            summary_text=sum_data['summary_text'],
            coverage_summary=sum_data['coverage_summary'],
            exclusions_summary=sum_data['exclusions_summary'],
            waiting_period_summary=sum_data['waiting_period_summary'],
            premium_summary=sum_data['premium_summary'],
            model_used="fallback_rules"
        )
        db.add(db_sum)
        
        # Update document status
        doc.status = "completed"
        
        await db.commit()
        print("Successfully updated document fields and summary in database!")

if __name__ == "__main__":
    asyncio.run(main())
