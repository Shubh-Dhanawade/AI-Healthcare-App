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
from app.services.ai_service import SUMMARIZATION_PROMPT, MOCK_SUMMARY, extract_json_from_response, _clean_field

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
    text: str
) -> Dict[str, Any]:
    """Generate BRIEF summary using Ollama, save to SQLite and cache. Optimized for speed."""
    # Check if already exists in DB (safety check)
    existing = await get_document_summary(db, doc_id)
    if existing:
        logger.info(f"Summary already exists for document {doc_id}, skipping generation.")
        return existing
        
    # Generate new summary from document excerpt
    truncated = text[:2000] if len(text) > 2000 else text
    summary_data = {}
    
    try:
        response = await call_ollama(
            SUMMARIZATION_PROMPT.format(document_text=truncated),
            num_predict=400,  # Reduced from 600 for faster generation of brief summaries
            num_ctx=2048,
        )
        result = extract_json_from_response(response)
        if result.get("summary_text"):
            logger.info("Ollama summary generation successful")
            summary_data = {
                "summary_text": _clean_field(result.get("summary_text", MOCK_SUMMARY["summary_text"])),
                "coverage_summary": _clean_field(result.get("coverage_summary")),
                "exclusions_summary": _clean_field(result.get("exclusions_summary")),
                "waiting_period_summary": _clean_field(result.get("waiting_period_summary")),
                "premium_summary": _clean_field(result.get("premium_summary")),
            }
    except Exception as e:
        logger.warning(f"Ollama summarization error ({e}), using mock fallback.")
        summary_data = dict(MOCK_SUMMARY)
        
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
