"""AI Service API endpoints for summarization, field extraction, and risk analysis."""


from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
import asyncio
from collections import defaultdict

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.document import Document, ExtractedField
from app.models.risk_analysis import Summary, RiskAnalysis
from app.schemas.schemas import (
    SummarizeRequest, SummarizeResponse, SummarySchema,
    ExtractFieldsRequest, ExtractedFieldsResponse, ExtractedFieldSchema,
    RiskAnalysisRequest, RiskAnalysisResponse, RiskAnalysisSchema,
    ChatQueryRequest, ChatQueryResponse, TranslateRequest, TranslateResponse,
    ClaimsChecklistRequest, ClaimsChecklistResponse,
    QueryRequest, QueryResponse,
)
from app.services.rag_service import query_rag_pipeline
from app.services.ai_service import (
    generate_summary, extract_policy_fields, analyze_risks,
    query_policy_rag, translate_text, generate_claims_checklist,
)

router = APIRouter()

# Global dictionary to map document ID to an asyncio.Lock to prevent concurrent AI processing
_document_locks = defaultdict(asyncio.Lock)


async def _get_document(
    document_id: str,
    current_user: User,
    db: AsyncSession,
) -> Document:
    """Fetch document and verify ownership."""
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.user_id == current_user.id,
        )
    )
    doc = result.scalar_one_or_none()
    
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    if not doc.extracted_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text not yet extracted. Please wait for processing to complete.",
        )
    
    return doc


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(
    request: SummarizeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate an AI summary of the insurance document."""
    doc = await _get_document(request.document_id, current_user, db)
    
    # Acquire lock for this document to prevent concurrent AI processing
    async with _document_locks[doc.id]:
        # Delete existing summary for re-generation
        existing = await db.execute(
            select(Summary).where(Summary.document_id == doc.id)
        )
        existing_summary = existing.scalar_one_or_none()
        
        if existing_summary:
            logger.info(f"Deleting existing summary for document {doc.id} to regenerate")
            await db.delete(existing_summary)
            await db.flush()
        
        # Generate new summary via AI
        logger.info(f"Generating AI summary for document {doc.id}")
        summary_data = await generate_summary(doc.extracted_text)
        
        # Save to database
        summary = Summary(
            document_id=doc.id,
            summary_text=summary_data["summary_text"],
            coverage_summary=summary_data.get("coverage_summary"),
            exclusions_summary=summary_data.get("exclusions_summary"),
            waiting_period_summary=summary_data.get("waiting_period_summary"),
            premium_summary=summary_data.get("premium_summary"),
            model_used="phi3",
        )
        db.add(summary)
        
        # Update document status
        doc.status = "summarized"
        
        from sqlalchemy.exc import IntegrityError
        try:
            await db.flush()
        except IntegrityError:
            await db.rollback()
            # Try to fetch existing summary that was inserted concurrently
            existing = await db.execute(
                select(Summary).where(Summary.document_id == doc.id)
            )
            existing_summary = existing.scalar_one_or_none()
            if existing_summary:
                summary = existing_summary
            else:
                raise
        else:
            await db.refresh(summary)
        
        return SummarizeResponse(
            document_id=doc.id,
            summary=SummarySchema.model_validate(summary),
        )


@router.post("/extract-fields", response_model=ExtractedFieldsResponse)
async def extract_fields(
    request: ExtractFieldsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Extract key policy fields from the insurance document."""
    doc = await _get_document(request.document_id, current_user, db)
    
    # Acquire lock for this document to prevent concurrent AI processing
    async with _document_locks[doc.id]:
        # Delete existing fields for re-extraction
        existing = await db.execute(
            select(ExtractedField).where(ExtractedField.document_id == doc.id)
        )
        existing_fields = existing.scalars().all()
        
        if existing_fields:
            for f in existing_fields:
                await db.delete(f)
            await db.flush()
        
        # Extract via AI
        logger.info(f"Extracting fields for document {doc.id}")
        fields_data = await extract_policy_fields(doc.extracted_text)
        
        saved_fields = []
        for field in fields_data:
            ef = ExtractedField(
                document_id=doc.id,
                field_name=field["field_name"],
                field_value=field["field_value"],
                field_category=field.get("field_category"),
            )
            db.add(ef)
            saved_fields.append(ef)
        
        await db.flush()
        for f in saved_fields:
            await db.refresh(f)
        
        # Update document status
        if doc.status not in ("completed",):
            doc.status = "completed"
        
        return ExtractedFieldsResponse(
            document_id=doc.id,
            fields=[ExtractedFieldSchema.model_validate(f) for f in saved_fields],
        )


@router.post("/risk-analysis", response_model=RiskAnalysisResponse)
async def risk_analysis(
    request: RiskAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Detect and analyze risky clauses in the insurance document."""
    doc = await _get_document(request.document_id, current_user, db)
    
    # Acquire lock for this document to prevent concurrent AI processing
    async with _document_locks[doc.id]:
        # Delete existing risks for re-generation
        existing = await db.execute(
            select(RiskAnalysis).where(RiskAnalysis.document_id == doc.id)
        )
        existing_risks = existing.scalars().all()
        
        if existing_risks:
            logger.info(f"Deleting existing risk analysis for document {doc.id} to regenerate")
            for r in existing_risks:
                await db.delete(r)
            await db.flush()
        
        # Analyze via AI
        logger.info(f"Running risk analysis for document {doc.id}")
        risk_data = await analyze_risks(doc.extracted_text)
        
        saved_risks = []
        for risk in risk_data.get("risks", []):
            ra = RiskAnalysis(
                document_id=doc.id,
                clause_text=risk["clause_text"],
                risk_type=risk["risk_type"],
                severity=risk.get("severity", "medium"),
                explanation=risk.get("explanation"),
                recommendation=risk.get("recommendation"),
            )
            db.add(ra)
            saved_risks.append(ra)
        
        await db.flush()
        for r in saved_risks:
            await db.refresh(r)
        
        return RiskAnalysisResponse(
            document_id=doc.id,
            risks=[RiskAnalysisSchema.model_validate(r) for r in saved_risks],
            overall_risk_level=risk_data.get("overall_risk_level", "medium"),
        )


@router.post("/chat", response_model=ChatQueryResponse)
async def query_chatbot(
    request: ChatQueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Conversational AI chatbot query over policies using RAG."""
    # 1. Fetch user documents (filtering by IDs if provided)
    query_stmt = select(Document).where(Document.user_id == current_user.id)
    if request.document_ids:
        query_stmt = query_stmt.where(Document.id.in_(request.document_ids))
    else:
        # Defaults to completed/summarized policies
        query_stmt = query_stmt.where(Document.status.in_(["completed", "summarized", "text_extracted"]))

    res = await db.execute(query_stmt)
    docs = res.scalars().all()
    
    if not docs:
        return ChatQueryResponse(response="No policies found in your library. Please upload policy documents first.")

    # 2. Package policy data for RAG
    policies_data = [
        {
            "id": d.id,
            "filename": d.original_filename,
            "text": d.extracted_text or ""
        }
        for d in docs
    ]

    # 3. Format history for service
    history_data = []
    if request.history:
        history_data = [
            {"role": h.role, "content": h.content}
            for h in request.history
        ]

    # 4. Generate RAG answer
    response_text = await query_policy_rag(policies_data, request.query, history_data)
    return ChatQueryResponse(response=response_text)


@router.post("/translate", response_model=TranslateResponse)
async def translate_summary(
    request: TranslateRequest,
    current_user: User = Depends(get_current_user),
):
    """Translate summary texts dynamically using Ollama."""
    translated = await translate_text(request.text, request.target_language)
    return TranslateResponse(translated_text=translated)


@router.post("/claims-checklist", response_model=ClaimsChecklistResponse)
async def generate_checklist(
    request: ClaimsChecklistRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate dynamic claim checklist for a document and treatment."""
    doc = await _get_document(request.document_id, current_user, db)
    
    # Format fields context
    res_fields = await db.execute(
        select(ExtractedField).where(ExtractedField.document_id == doc.id)
    )
    fields = res_fields.scalars().all()
    fields_summary = "\n".join([f"{f.field_name}: {f.field_value}" for f in fields])
    
    checklist_data = await generate_claims_checklist(
        policy_name=doc.original_filename,
        fields_summary=fields_summary,
        treatment_type=request.treatment_type
    )
    return ClaimsChecklistResponse(**checklist_data)


@router.post("/documents/{document_id}/query", response_model=QueryResponse)
async def query_document(
    document_id: str,
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Query a document using local RAG and calculate evaluation metrics."""
    doc = await _get_document(document_id, current_user, db)
    
    logger.info(f"Querying document {doc.id} with prompt: {request.query} (evaluate={request.evaluate})")
    try:
        result = await query_rag_pipeline(doc.extracted_text, request.query, evaluate=request.evaluate)
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline failure: {e}",
        )
        
    return QueryResponse(
        document_id=doc.id,
        answer=result["answer"],
        context=result["context"],
        evaluation=result["evaluation"],
    )

