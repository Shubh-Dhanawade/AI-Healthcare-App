"""
Summary Service
Generates plain-language policy summaries using Ollama and stores them to the database.
All in-memory caching has been removed to prevent state sync issues.
"""

from typing import Dict, Any, Optional
from loguru import logger
from sqlalchemy import select, delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.risk_analysis import Summary
from app.services.ollama_client import settings


async def get_document_summary(db: AsyncSession, doc_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve summary from the database."""
    result = await db.execute(select(Summary).where(Summary.document_id == doc_id))
    summary = result.scalar_one_or_none()
    if summary:
        return {
            "summary_text": summary.summary_text,
            "coverage_summary": summary.coverage_summary,
            "exclusions_summary": summary.exclusions_summary,
            "waiting_period_summary": summary.waiting_period_summary,
            "premium_summary": summary.premium_summary,
        }
    return None


async def generate_and_store_summary(
    db: AsyncSession,
    doc_id: str,
    text: str,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """Generate summary using Ollama and save directly to DB."""
    if force_regenerate:
        # Delete any existing DB summary row for this document
        await db.execute(sql_delete(Summary).where(Summary.document_id == doc_id))
        await db.flush()
        logger.info(f"[SUMMARY] force_regenerate=True — existing DB summary cleared for {doc_id}")
    else:
        # Return existing if already generated in database
        existing = await get_document_summary(db, doc_id)
        if existing:
            logger.info(f"[SUMMARY] Existing DB summary found for {doc_id}, skipping generation.")
            return existing

    from app.models.document import Document, ExtractedField
    from sqlalchemy import select
    res = await db.execute(select(Document).where(Document.id == doc_id))
    doc = res.scalar_one_or_none()
    is_ocr = (doc.file_type == "image" or doc.extraction_method in ("easyocr", "paddleocr")) if doc else False

    # Fetch extracted fields from DB if they exist
    fields_res = await db.execute(select(ExtractedField).where(ExtractedField.document_id == doc_id))
    fields = fields_res.scalars().all()
    fields_summary = ""
    if fields:
        def _ensure_rupee(name: str, value: str) -> str:
            """Ensure monetary fields always show ₹ symbol."""
            monetary_fields = {"sum insured", "premium amount", "co payment", "room rent limit"}
            if name.lower() in monetary_fields and value and not any(c in value for c in ("₹", "Rs", "INR")):
                import re
                if re.search(r'\d', value):
                    return f"₹{value}"
            return value
        fields_summary = "\n".join(
            f"- {f.field_name}: {_ensure_rupee(f.field_name, f.field_value)}"
            for f in fields
        )

    # Commit to release any held DB locks before the slow Ollama call
    await db.commit()

    from app.services.ai_service import generate_summary
    summary_data = await generate_summary(text, force_regenerate=force_regenerate, is_ocr=is_ocr, fields_summary=fields_summary)

    # Save freshly generated summary to DB
    summary = Summary(
        document_id=doc_id,
        summary_text=summary_data["summary_text"],
        coverage_summary=summary_data.get("coverage_summary"),
        exclusions_summary=summary_data.get("exclusions_summary"),
        waiting_period_summary=summary_data.get("waiting_period_summary"),
        premium_summary=summary_data.get("premium_summary"),
        model_used=settings.OLLAMA_MODEL,
    )
    db.add(summary)
    await db.flush()

    logger.info(f"[SUMMARY] ✅ Fresh summary generated and stored in DB for document {doc_id}")
    return summary_data
