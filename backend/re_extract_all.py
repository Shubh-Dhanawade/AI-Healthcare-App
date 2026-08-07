"""
re_extract_all.py
─────────────────
Run this script ONCE to re-extract summary, fields and risks for EVERY
completed document in the database using the improved fallback extractors.

Usage (from the backend/ directory):
    .\\venv\\Scripts\\python.exe re_extract_all.py
"""

import asyncio
import sys
import os

# Make sure the backend app package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from loguru import logger

from app.core.config import settings
from app.models.document import Document, ExtractedField
from app.models.risk_analysis import Summary, RiskAnalysis
from app.services.ai_service import (
    generate_summary,
    extract_policy_fields,
    analyze_risks,
)


async def re_extract_document(db: AsyncSession, doc: Document) -> None:
    """Re-run all three extractions for a single document."""
    text = doc.extracted_text
    if not text or len(text.strip()) < 20:
        logger.warning(f"  ⚠️  [{doc.id[:8]}] No text — skipping.")
        return

    logger.info(f"  ↻  Processing: {doc.original_filename} ({len(text):,} chars)")

    # ── 1. Summary ────────────────────────────────────────────────────────────
    try:
        summary_data = await generate_summary(text, force_regenerate=True)

        # Delete old summary row if exists
        existing = await db.execute(select(Summary).where(Summary.document_id == doc.id))
        old = existing.scalar_one_or_none()
        if old:
            await db.delete(old)
            await db.flush()

        db.add(Summary(
            document_id=doc.id,
            summary_text=summary_data["summary_text"],
            coverage_summary=summary_data.get("coverage_summary"),
            exclusions_summary=summary_data.get("exclusions_summary"),
            waiting_period_summary=summary_data.get("waiting_period_summary"),
            premium_summary=summary_data.get("premium_summary"),
            model_used="hybrid-ollama-v2",
        ))
        logger.info(f"     ✅ Summary regenerated")
    except Exception as e:
        logger.error(f"     ❌ Summary failed: {e}")

    # ── 2. Extracted Fields ───────────────────────────────────────────────────
    try:
        fields_data = await extract_policy_fields(text, force_regenerate=True)

        # Delete old fields
        old_fields = await db.execute(
            select(ExtractedField).where(ExtractedField.document_id == doc.id)
        )
        for f in old_fields.scalars().all():
            await db.delete(f)
        await db.flush()

        for field in fields_data:
            db.add(ExtractedField(
                document_id=doc.id,
                field_name=field["field_name"],
                field_value=field["field_value"],
                field_category=field.get("field_category"),
            ))
        logger.info(f"     ✅ {len(fields_data)} fields extracted")
    except Exception as e:
        logger.error(f"     ❌ Fields failed: {e}")

    # ── 3. Risk Analysis ──────────────────────────────────────────────────────
    try:
        risk_data = await analyze_risks(text, force_regenerate=True)
        risks = risk_data.get("risks", [])

        # Delete old risks
        old_risks = await db.execute(
            select(RiskAnalysis).where(RiskAnalysis.document_id == doc.id)
        )
        for r in old_risks.scalars().all():
            await db.delete(r)
        await db.flush()

        for risk in risks:
            db.add(RiskAnalysis(
                document_id=doc.id,
                clause_text=risk["clause_text"],
                risk_type=risk["risk_type"],
                severity=risk.get("severity", "medium"),
                explanation=risk.get("explanation"),
                recommendation=risk.get("recommendation"),
            ))
        logger.info(f"     ✅ {len(risks)} risks identified (overall: {risk_data.get('overall_risk_level')})")
    except Exception as e:
        logger.error(f"     ❌ Risks failed: {e}")

    await db.commit()


async def main() -> None:
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with AsyncSessionLocal() as db:
        # Fetch all documents that have extracted text
        result = await db.execute(
            select(Document).where(Document.extracted_text.isnot(None))
        )
        docs = result.scalars().all()

    logger.info(f"Found {len(docs)} document(s) to re-process.\n")

    for doc in docs:
        async with AsyncSessionLocal() as db:
            # Re-fetch inside its own session for clean state
            res = await db.execute(select(Document).where(Document.id == doc.id))
            fresh_doc = res.scalar_one_or_none()
            if fresh_doc:
                await re_extract_document(db, fresh_doc)

    logger.info("\n✅ All documents re-processed. Refresh the browser to see updated results.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
