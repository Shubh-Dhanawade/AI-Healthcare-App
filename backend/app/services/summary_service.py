"""
Summary Service
Generates plain-language policy summaries using Ollama and caches them to SQLite
to avoid redundant model execution on subsequent loads.
"""

from typing import Dict, Any, Optional
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.risk_analysis import Summary
from app.services.cache_manager import CacheManager
from app.services.ollama_client import call_ollama, settings
from app.services.ai_service import SUMMARIZATION_PROMPT, _build_fallback_summary, extract_json_from_response, _clean_field

async def get_document_summary(db: AsyncSession, doc_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve summary from memory cache or database."""
    # Check memory cache
    cached = CacheManager.get_summary(doc_id)
    if cached:
        return cached
        
    # Check DB
    result = await db.execute(select(Summary).where(Summary.document_id == doc_id))
    summary = result.scalar_one_or_none()
    if summary:
        summary_data = {
            "summary_text": summary.summary_text,
            "coverage_summary": summary.coverage_summary,
            "exclusions_summary": summary.exclusions_summary,
            "waiting_period_summary": summary.waiting_period_summary,
            "premium_summary": summary.premium_summary,
        }
        CacheManager.set_summary(doc_id, summary_data)
        return summary_data
        
    return None

async def generate_and_store_summary(
    db: AsyncSession,
    doc_id: str,
    text: str,
    force_regenerate: bool = False,
) -> Dict[str, Any]:
    """Generate summary using Ollama, save to SQLite and cache.
    
    Pass force_regenerate=True to delete the old summary and generate a fresh one from the actual document.
    """
    # Check if already exists in DB
    existing = await get_document_summary(db, doc_id)
    if existing and not force_regenerate:
        logger.info(f"Summary already exists for document {doc_id}, skipping generation.")
        return existing
    
    # If force_regenerate, delete existing DB row and evict cache
    if force_regenerate and existing:
        del_result = await db.execute(select(Summary).where(Summary.document_id == doc_id))
        old_summary = del_result.scalar_one_or_none()
        if old_summary:
            await db.delete(old_summary)
            await db.flush()
        CacheManager.invalidate_summary(doc_id)
        logger.info(f"Deleted existing summary for {doc_id} to force regeneration.")
        
    # Commit transaction to release SQLite database locks during the slow LLM call
    await db.commit()
        
    from app.services.ai_service import generate_summary
    summary_data = await generate_summary(text, force_regenerate=force_regenerate)
        
    # Save to DB
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
    
    # Cache in memory
    CacheManager.set_summary(doc_id, summary_data)
    logger.info(f"✅ Summary generated and cached for document {doc_id}")
    return summary_data
