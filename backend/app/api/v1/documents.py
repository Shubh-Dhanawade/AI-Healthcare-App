"""Document upload and management API endpoints."""

import os
import uuid
from pathlib import Path as FilePath
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from loguru import logger

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User
from app.models.document import Document, ExtractedField
from app.models.risk_analysis import Summary, RiskAnalysis
from app.models.reminder import PolicyReminder
from app.schemas.schemas import DocumentResponse, DocumentDetailResponse, CompareRequest, CompareResponse, ComparisonSynthesisSchema
from app.services.ocr_service import extract_document_text
from app.services.ai_service import generate_comparison_synthesis
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# In-progress tracker — prevents duplicate background analysis jobs
# ─────────────────────────────────────────────────────────────────────────────
_analysis_in_progress: set = set()


async def _run_fields_background(doc_id: str, force_regenerate: bool = False) -> None:
    """Server-side asyncio task: Extract Fields only. Launched via create_task()."""
    from app.core.database import AsyncSessionLocal
    from app.services.ai_service import extract_policy_fields

    tracker_key = f"fields:{doc_id}"
    logger.info(f"[BG-FIELDS] Starting field extraction for {doc_id} (force={force_regenerate})")

    # 1. Fetch extracted text in a quick read-only block to release locks immediately
    extracted_text = None
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc and doc.extracted_text:
            extracted_text = doc.extracted_text

    if not extracted_text:
        logger.warning(f"[BG-FIELDS] Document {doc_id} not found or has no text. Aborting.")
        _analysis_in_progress.discard(tracker_key)
        return

    try:
        # 2. Call Ollama FIRST (takes 10-30s, NO database session/lock is held!)
        fields_data = await extract_policy_fields(extracted_text, force_regenerate=force_regenerate)

        # 3. Save to database in a new quick write transaction
        async with AsyncSessionLocal() as db:
            # Clear old fields first
            existing = await db.execute(select(ExtractedField).where(ExtractedField.document_id == doc_id))
            for f in existing.scalars().all():
                await db.delete(f)
            await db.flush()

            for field in fields_data:
                db.add(ExtractedField(
                    document_id=doc_id,
                    field_name=field["field_name"],
                    field_value=field["field_value"],
                    field_category=field.get("field_category"),
                ))

            # Update document status
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc and doc.status not in ("completed",):
                doc.status = "completed"

            await db.commit()
            logger.info(f"[BG-FIELDS] Done for {doc_id} — {len(fields_data)} fields saved")
    except Exception as e:
        logger.error(f"[BG-FIELDS] Failed for {doc_id}: {e}")

    _analysis_in_progress.discard(tracker_key)


async def _run_risks_background(doc_id: str, force_regenerate: bool = False) -> None:
    """Server-side asyncio task: Risk Analysis only. Launched via create_task()."""
    from app.core.database import AsyncSessionLocal
    from app.services.ai_service import analyze_risks

    tracker_key = f"risks:{doc_id}"
    logger.info(f"[BG-RISKS] Starting risk analysis for {doc_id} (force={force_regenerate})")

    # 1. Fetch extracted text in a quick read-only block to release locks immediately
    extracted_text = None
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc and doc.extracted_text:
            extracted_text = doc.extracted_text

    if not extracted_text:
        logger.warning(f"[BG-RISKS] Document {doc_id} not found or has no text. Aborting.")
        _analysis_in_progress.discard(tracker_key)
        return

    try:
        # 2. Call Ollama FIRST (takes 10-30s, NO database session/lock is held!)
        risk_data = await analyze_risks(extracted_text, force_regenerate=force_regenerate)

        # 3. Save to database in a new quick write transaction
        async with AsyncSessionLocal() as db:
            # Clear old risks first
            existing = await db.execute(select(RiskAnalysis).where(RiskAnalysis.document_id == doc_id))
            for r in existing.scalars().all():
                await db.delete(r)
            await db.flush()

            for risk in risk_data.get("risks", []):
                db.add(RiskAnalysis(
                    document_id=doc_id,
                    clause_text=risk["clause_text"],
                    risk_type=risk["risk_type"],
                    severity=risk.get("severity", "medium"),
                    explanation=risk.get("explanation"),
                    recommendation=risk.get("recommendation"),
                ))

            await db.commit()
            logger.info(f"[BG-RISKS] Done for {doc_id} — {len(risk_data.get('risks', []))} risks saved")
    except Exception as e:
        logger.error(f"[BG-RISKS] Failed for {doc_id}: {e}")

    _analysis_in_progress.discard(tracker_key)


async def _run_summary_background(doc_id: str, force_regenerate: bool = False) -> None:
    """Server-side asyncio task: Summarization only. Launched via create_task()."""
    from app.core.database import AsyncSessionLocal
    from app.services.summary_service import generate_and_store_summary

    tracker_key = f"summary:{doc_id}"
    logger.info(f"[BG-SUMMARY] Starting summarization for {doc_id} (force={force_regenerate})")

    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if not doc or not doc.extracted_text:
                logger.warning(f"[BG-SUMMARY] Document {doc_id} not found or has no text. Aborting.")
                return

            extracted_text = doc.extracted_text

            # Call summary service — this calls Ollama then saves to DB
            await generate_and_store_summary(db, doc_id, extracted_text, force_regenerate=force_regenerate)
            await db.commit()
            logger.info(f"[BG-SUMMARY] Done for {doc_id}")
        except Exception as e:
            logger.error(f"[BG-SUMMARY] Failed for {doc_id}: {e}")

    _analysis_in_progress.discard(tracker_key)


async def run_full_analysis_background(doc_id: str) -> None:
    """Combined wrapper: runs summary, fields, and risks sequentially (used by upload pipeline)."""
    await _run_summary_background(doc_id, force_regenerate=False)
    await _run_fields_background(doc_id)
    await _run_risks_background(doc_id)



# Allowed MIME types
ALLOWED_MIME_TYPES = {
    "application/pdf": "pdf",
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/png": "image",
    "image/tiff": "image",
    "image/webp": "image",
}

MAX_FILE_SIZE = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes


async def process_document_background(doc_id: str, file_path: str, file_type: str):
    """Background task to extract text from uploaded document and perform auto-analysis.
    
    Phase 1 (FAST): Extract text → commit to DB → mark as 'text_extracted' so UI unblocks immediately.
    Phase 2 (BACKGROUND): Run chunking+embedding and AI summary CONCURRENTLY, then mark 'completed'.
    """
    from app.core.database import AsyncSessionLocal
    import asyncio
    
    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if not doc:
                return
            
            # ── PHASE 1: Fast text extraction (unblocks UI quickly) ──
            doc.status = "processing"
            await db.commit()
            
            text, method, page_count = await extract_document_text(file_path, file_type)
            
            doc.extracted_text = text
            doc.extraction_method = method
            doc.page_count = page_count
            doc.status = "text_extracted"  # ← UI can now display the document
            await db.commit()
            logger.info(f"Phase 1 done for {doc_id}: text extracted ({len(text)} chars), status=text_extracted")

        except Exception as e:
            logger.error(f"Phase 1 text extraction failed for {doc_id}: {e}")
            async with AsyncSessionLocal() as err_db:
                result = await err_db.execute(select(Document).where(Document.id == doc_id))
                doc = result.scalar_one_or_none()
                if doc:
                    doc.status = "failed"
                    await err_db.commit()
            return

    # ── PHASE 2: Concurrent embedding + summary (background, non-blocking for user) ──
    async def _run_embeddings():
        async with AsyncSessionLocal() as db2:
            try:
                from app.services.rag_service import generate_document_chunks
                await generate_document_chunks(doc_id, text, db2)
                await db2.commit()
                logger.info(f"Chunking & FAISS indexing complete for {doc_id}")
            except Exception as e:
                logger.error(f"Chunking/embedding failed for {doc_id}: {e}")

    async def _run_summary():
        async with AsyncSessionLocal() as db3:
            try:
                from app.services.summary_service import generate_and_store_summary
                await generate_and_store_summary(db3, doc_id, text)
                await db3.commit()
                logger.info(f"AI summary complete for {doc_id}")
            except Exception as e:
                logger.error(f"Auto-summarization failed for {doc_id}: {e}")

    # Run embeddings + summary concurrently
    await asyncio.gather(_run_embeddings(), _run_summary())

    # ── PHASE 3: Auto-launch field extraction + risk analysis ──────────────────
    # Register tracker keys so manual re-triggers don't create duplicates
    _analysis_in_progress.add(f"fields:{doc_id}")
    _analysis_in_progress.add(f"risks:{doc_id}")
    # Run both concurrently as detached tasks (won't block Phase 3 status update)
    asyncio.create_task(_run_fields_background(doc_id))
    asyncio.create_task(_run_risks_background(doc_id))
    logger.info(f"[AUTO] Field extraction + risk analysis tasks launched for {doc_id}")

    # Final status update — mark as completed (fields/risks will update DB when done)
    async with AsyncSessionLocal() as db_final:
        try:
            result = await db_final.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = "completed"
                await db_final.commit()
            logger.info(f"Document {doc_id} fully processed (status=completed)")
        except Exception as e:
            logger.error(f"Failed to update final status for {doc_id}: {e}")


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a healthcare insurance document (PDF or image)."""
    # Validate file type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Supported: PDF, JPG, PNG, TIFF, WEBP",
        )
    
    file_type = ALLOWED_MIME_TYPES[content_type]
    
    # Read file and validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB",
        )
    
    # Generate unique filename
    extension = FilePath(file.filename).suffix.lower() or (".pdf" if file_type == "pdf" else ".jpg")
    stored_filename = f"{uuid.uuid4()}{extension}"
    
    # Create user-specific upload directory
    user_upload_dir = FilePath(settings.UPLOAD_DIR) / str(current_user.id)
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = str(user_upload_dir / stored_filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(content)
    
    logger.info(f"File saved: {file_path}")
    
    # Create database record
    doc = Document(
        user_id=current_user.id,
        original_filename=file.filename,
        stored_filename=stored_filename,
        file_path=file_path,
        file_type=file_type,
        file_size_bytes=len(content),
        mime_type=content_type,
        status="uploaded",
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)
    
    # Start background text extraction
    background_tasks.add_task(
        process_document_background, doc.id, file_path, file_type
    )
    
    return DocumentResponse.model_validate(doc)


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all documents for the current user."""
    result = await db.execute(
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(desc(Document.created_at))
    )
    docs = result.scalars().all()
    return [DocumentResponse.model_validate(d) for d in docs]


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: All static sub-path routes (e.g. /reminders, /compare) MUST be
# registered BEFORE the /{document_id} wildcard to avoid being caught by it.
# ─────────────────────────────────────────────────────────────────────────────

class ScheduleReminderRequest(BaseModel):
    document_id: str
    renewal_date: Optional[datetime] = None
    premium_due_date: Optional[datetime] = None
    premium_amount: Optional[str] = None


class EmailReportRequest(BaseModel):
    email: EmailStr


@router.get("/reminders")
async def get_reminders(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List scheduled premium/renewal reminders for the user."""
    result = await db.execute(
        select(PolicyReminder)
        .where(PolicyReminder.user_id == current_user.id, PolicyReminder.is_dismissed == False)
        .order_by(PolicyReminder.reminder_date)
    )
    reminders = result.scalars().all()
    
    # Format responses dynamically
    return [
        {
            "id": r.id,
            "document_id": r.document_id,
            "title": r.title,
            "reminder_type": r.reminder_type,
            "reminder_date": r.reminder_date,
            "premium_amount": r.premium_amount,
            "is_dismissed": r.is_dismissed
        }
        for r in reminders
    ]


@router.post("/reminders")
async def schedule_reminder(
    request: ScheduleReminderRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Schedule policy premium and renewal notifications."""
    result = await db.execute(
        select(Document).where(
            Document.id == request.document_id,
            Document.user_id == current_user.id
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Update document dates
    if request.renewal_date:
        doc.renewal_date = request.renewal_date
        # Create reminder alert
        # Trigger 7 days prior
        trigger_date = request.renewal_date - timedelta(days=7)
        r1 = PolicyReminder(
            user_id=current_user.id,
            document_id=doc.id,
            title=f"Policy Renewal Approaching: {doc.original_filename}",
            reminder_type="renewal",
            reminder_date=trigger_date
        )
        db.add(r1)
        
    if request.premium_due_date:
        doc.premium_due_date = request.premium_due_date
        trigger_date = request.premium_due_date - timedelta(days=5)
        r2 = PolicyReminder(
            user_id=current_user.id,
            document_id=doc.id,
            title=f"Premium Payment Approaching: {doc.original_filename}",
            reminder_type="premium",
            reminder_date=trigger_date,
            premium_amount=request.premium_amount
        )
        db.add(r2)
        
    await db.commit()
    return {"message": "Policy dates and reminders successfully scheduled"}


@router.patch("/reminders/{reminder_id}/dismiss")
async def dismiss_reminder(
    reminder_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Dismiss/acknowledge an active notification alert."""
    result = await db.execute(
        select(PolicyReminder).where(
            PolicyReminder.id == reminder_id,
            PolicyReminder.user_id == current_user.id
        )
    )
    reminder = result.scalar_one_or_none()
    if not reminder:
        raise HTTPException(status_code=404, detail="Notification alert not found")
        
    reminder.is_dismissed = True
    await db.commit()
    return {"message": "Notification successfully dismissed"}


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get detailed document info including AI results."""
    from sqlalchemy.orm import selectinload
    
    result = await db.execute(
        select(Document)
        .where(Document.id == document_id, Document.user_id == current_user.id)
        .options(
            selectinload(Document.summary),
            selectinload(Document.extracted_fields),
            selectinload(Document.risk_analyses),
        )
    )
    doc = result.scalar_one_or_none()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    
    return DocumentDetailResponse.model_validate(doc)


@router.post("/{document_id}/run-summary", status_code=status.HTTP_202_ACCEPTED)
async def trigger_background_summary(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Force-regenerate the AI Summary from the actual document text (deletes stale/mock data first).
    Returns 202 immediately. Poll GET /documents/{id} for results.
    """
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
            detail="Document text not yet extracted. Please wait for processing.",
        )

    tracker_key = f"summary:{document_id}"
    if tracker_key in _analysis_in_progress:
        return {"status": "already_running", "message": "Summarization is already running for this document."}

    _analysis_in_progress.add(tracker_key)
    import asyncio
    asyncio.create_task(_run_summary_background(document_id, force_regenerate=True))

    logger.info(f"[API] Background re-summarization launched for {document_id}")
    return {
        "status": "started",
        "message": "Summary regeneration queued on server. Results will appear automatically.",
        "document_id": document_id,
        "job": "summary",
    }


@router.post("/{document_id}/run-fields", status_code=status.HTTP_202_ACCEPTED)
async def trigger_background_fields(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Kick off Extract Fields as a server-side asyncio background task.
    Returns 202 immediately. Poll GET /documents/{id} for results.
    """
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
            detail="Document text not yet extracted. Please wait for processing.",
        )

    tracker_key = f"fields:{document_id}"
    if tracker_key in _analysis_in_progress:
        return {"status": "already_running", "message": "Field extraction is already running for this document."}

    _analysis_in_progress.add(tracker_key)
    import asyncio
    asyncio.create_task(_run_fields_background(document_id, force_regenerate=True))

    logger.info(f"[API] Background field extraction launched for {document_id}")
    return {
        "status": "started",
        "message": "Field extraction queued on server. Results will appear automatically — safe to navigate away.",
        "document_id": document_id,
        "job": "fields",
    }


@router.post("/{document_id}/run-risks", status_code=status.HTTP_202_ACCEPTED)
async def trigger_background_risks(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Kick off Risk Analysis as a server-side asyncio background task.
    Returns 202 immediately. Poll GET /documents/{id} for results.
    """
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
            detail="Document text not yet extracted. Please wait for processing.",
        )

    tracker_key = f"risks:{document_id}"
    if tracker_key in _analysis_in_progress:
        return {"status": "already_running", "message": "Risk analysis is already running for this document."}

    _analysis_in_progress.add(tracker_key)
    import asyncio
    asyncio.create_task(_run_risks_background(document_id, force_regenerate=True))

    logger.info(f"[API] Background risk analysis launched for {document_id}")
    return {
        "status": "started",
        "message": "Risk analysis queued on server. Results will appear automatically — safe to navigate away.",
        "document_id": document_id,
        "job": "risks",
    }


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a document and its associated data."""
    result = await db.execute(
        select(Document).where(
            Document.id == document_id, Document.user_id == current_user.id
        )
    )
    doc = result.scalar_one_or_none()
    
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    # Delete physical file
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    
    await db.delete(doc)
    logger.info(f"Document {document_id} deleted")


@router.post("/compare", response_model=CompareResponse)
async def compare_documents(
    request: CompareRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Compare 2 or 3 documents side-by-side."""
    from sqlalchemy.orm import selectinload
    
    # Check length
    if len(request.document_ids) < 2 or len(request.document_ids) > 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must select between 2 and 3 documents to compare.",
        )
    
    # Fetch documents with their summary, extracted fields, and risks
    result = await db.execute(
        select(Document)
        .where(
            Document.id.in_(request.document_ids),
            Document.user_id == current_user.id
        )
        .options(
            selectinload(Document.summary),
            selectinload(Document.extracted_fields),
            selectinload(Document.risk_analyses),
        )
    )
    docs = result.scalars().all()
    
    # Verify all exist and belong to user
    if len(docs) != len(request.document_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or more documents not found or access denied.",
        )
        
    # Check status of documents - they must have been processed (completed/summarized/text_extracted)
    for doc in docs:
        if doc.status in ("uploaded", "processing"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document '{doc.original_filename}' is still processing. Please wait.",
            )
        if doc.status == "failed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document '{doc.original_filename}' failed to process and cannot be compared.",
            )

    # Format documents data for the AI synthesis service call
    policies_data = []
    for doc in docs:
        doc_dict = {
            "id": doc.id,
            "original_filename": doc.original_filename,
            "status": doc.status,
            "summary": {
                "summary_text": doc.summary.summary_text if doc.summary else "",
                "coverage_summary": doc.summary.coverage_summary if doc.summary else "",
                "exclusions_summary": doc.summary.exclusions_summary if doc.summary else "",
                "waiting_period_summary": doc.summary.waiting_period_summary if doc.summary else "",
                "premium_summary": doc.summary.premium_summary if doc.summary else "",
            } if doc.summary else None,
            "extracted_fields": [
                {
                    "field_name": f.field_name,
                    "field_value": f.field_value,
                    "field_category": f.field_category,
                }
                for f in doc.extracted_fields
            ],
            "risk_analyses": [
                {
                    "clause_text": r.clause_text,
                    "risk_type": r.risk_type,
                    "severity": r.severity,
                    "explanation": r.explanation,
                    "recommendation": r.recommendation,
                }
                for r in doc.risk_analyses
            ]
        }
        
        # Calculate dynamic overall risk level
        high_count = sum(1 for r in doc.risk_analyses if r.severity == "high")
        med_count = sum(1 for r in doc.risk_analyses if r.severity == "medium")
        if high_count > 0:
            doc_dict["overall_risk_level"] = "high"
        elif med_count > 0:
            doc_dict["overall_risk_level"] = "medium"
        else:
            doc_dict["overall_risk_level"] = "low"
            
        policies_data.append(doc_dict)

    # Generate comparison synthesis using Ollama or fallback mock
    synthesis_report = await generate_comparison_synthesis(policies_data)
    
    # Return documents details + comparative synthesis
    return CompareResponse(
        documents=[DocumentDetailResponse.model_validate(d) for d in docs],
        comparison_synthesis=ComparisonSynthesisSchema(
            synthesis=synthesis_report.get("synthesis", ""),
            best_for=synthesis_report.get("best_for", ""),
            verdict=synthesis_report.get("verdict", ""),
            feature_winners=synthesis_report.get("feature_winners", [])
        )
    )


# ─────────────────────────────────────────
# Reminders and Exporter Endpoints
# ─────────────────────────────────────────

class ScheduleReminderRequest(BaseModel):
    document_id: str
    renewal_date: Optional[datetime] = None
    premium_due_date: Optional[datetime] = None
    premium_amount: Optional[str] = None


class EmailReportRequest(BaseModel):
    email: EmailStr


@router.get("/reminders")
async def get_reminders(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List scheduled premium/renewal reminders for the user."""
    result = await db.execute(
        select(PolicyReminder)
        .where(PolicyReminder.user_id == current_user.id, PolicyReminder.is_dismissed == False)
        .order_by(PolicyReminder.reminder_date)
    )
    reminders = result.scalars().all()
    
    # Format responses dynamically
    return [
        {
            "id": r.id,
            "document_id": r.document_id,
            "title": r.title,
            "reminder_type": r.reminder_type,
            "reminder_date": r.reminder_date,
            "premium_amount": r.premium_amount,
            "is_dismissed": r.is_dismissed
        }
        for r in reminders
    ]


@router.post("/reminders")
async def schedule_reminder(
    request: ScheduleReminderRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Schedule policy premium and renewal notifications."""
    result = await db.execute(
        select(Document).where(
            Document.id == request.document_id,
            Document.user_id == current_user.id
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Update document dates
    if request.renewal_date:
        doc.renewal_date = request.renewal_date
        # Create reminder alert
        # Trigger 7 days prior
        trigger_date = request.renewal_date - timedelta(days=7)
        r1 = PolicyReminder(
            user_id=current_user.id,
            document_id=doc.id,
            title=f"Policy Renewal Approaching: {doc.original_filename}",
            reminder_type="renewal",
            reminder_date=trigger_date
        )
        db.add(r1)
        
    if request.premium_due_date:
        doc.premium_due_date = request.premium_due_date
        trigger_date = request.premium_due_date - timedelta(days=5)
        r2 = PolicyReminder(
            user_id=current_user.id,
            document_id=doc.id,
            title=f"Premium Payment Approaching: {doc.original_filename}",
            reminder_type="premium",
            reminder_date=trigger_date,
            premium_amount=request.premium_amount
        )
        db.add(r2)
        
    await db.commit()
    return {"message": "Policy dates and reminders successfully scheduled"}


@router.patch("/reminders/{reminder_id}/dismiss")
async def dismiss_reminder(
    reminder_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Dismiss/acknowledge an active notification alert."""
    result = await db.execute(
        select(PolicyReminder).where(
            PolicyReminder.id == reminder_id,
            PolicyReminder.user_id == current_user.id
        )
    )
    reminder = result.scalar_one_or_none()
    if not reminder:
        raise HTTPException(status_code=404, detail="Notification alert not found")
        
    reminder.is_dismissed = True
    await db.commit()
    return {"message": "Notification successfully dismissed"}


def generate_html_report(doc: Document) -> str:
    """Helper to generate a clean, responsive HTML print template for policy reports."""
    fields_list = ""
    for f in doc.extracted_fields:
        fields_list += f"""
        <div class="field-item">
            <span class="field-label">{f.field_name}</span>
            <span class="field-value">{f.field_value or "—"}</span>
        </div>
        """
        
    risks_list = ""
    if not doc.risk_analyses:
        risks_list = "<p style='color: #10b981; font-weight: 500;'>No critical risk clauses detected.</p>"
    else:
        for r in doc.risk_analyses:
            color = "#f87171" if r.severity == "high" else ("#fbbf24" if r.severity == "medium" else "#34d399")
            risks_list += f"""
            <div class="risk-card" style="border-left: 4px solid {color}">
                <div class="risk-header">
                    <span class="risk-type">{r.risk_type.replace('_', ' ').upper()}</span>
                    <span class="risk-severity" style="color: {color}; font-weight: bold;">{r.severity.upper()}</span>
                </div>
                <p class="risk-clause"><strong>Clause:</strong> <em>"{r.clause_text}"</em></p>
                <p class="risk-explanation"><strong>Explanation:</strong> {r.explanation or "—"}</p>
                <p class="risk-rec"><strong>Recommendation:</strong> {r.recommendation or "—"}</p>
            </div>
            """
            
    summary_text = doc.summary.summary_text if doc.summary else "No summary available."
    coverage = doc.summary.coverage_summary if doc.summary else "—"
    exclusions = doc.summary.exclusions_summary if doc.summary else "—"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>HealthAI Document Report - {doc.original_filename}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                color: #0f172a;
                line-height: 1.5;
                margin: 0;
                padding: 40px;
                background: #f8fafc;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: #ffffff;
                padding: 40px;
                border-radius: 16px;
                box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
                border: 1px solid #e2e8f0;
            }}
            .header {{
                border-bottom: 2px solid #3b82f6;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 24px;
                color: #1e3a8a;
            }}
            .metadata {{
                font-size: 12px;
                color: #64748b;
                margin-top: 5px;
            }}
            .section {{
                margin-bottom: 35px;
            }}
            .section h2 {{
                font-size: 16px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #475569;
                border-bottom: 1px solid #e2e8f0;
                padding-bottom: 8px;
                margin-bottom: 15px;
            }}
            .field-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
            }}
            .field-item {{
                background: #f8fafc;
                padding: 12px 15px;
                border-radius: 8px;
                border: 1px solid #f1f5f9;
            }}
            .field-label {{
                display: block;
                font-size: 10px;
                text-transform: uppercase;
                color: #64748b;
                font-weight: bold;
            }}
            .field-value {{
                font-size: 14px;
                font-weight: 500;
                color: #1e293b;
                margin-top: 2px;
            }}
            .risk-card {{
                background: #fff8f8;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            }}
            .risk-header {{
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                font-weight: bold;
            }}
            .risk-clause {{
                font-size: 13px;
                color: #334155;
            }}
            .risk-explanation {{
                font-size: 13px;
                color: #475569;
            }}
            .risk-rec {{
                font-size: 12px;
                color: #2563eb;
                font-weight: 500;
            }}
            @media print {{
                body {{ background: none; padding: 0; }}
                .container {{ box-shadow: none; border: none; padding: 0; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Healthcare Policy Analysis Report</h1>
                <div class="metadata">
                    <strong>Document:</strong> {doc.original_filename} &nbsp;|&nbsp;
                    <strong>Processed:</strong> {doc.created_at.strftime('%Y-%m-%d')} &nbsp;|&nbsp;
                    <strong>Pages:</strong> {doc.page_count}
                </div>
            </div>
            
            <div class="section">
                <h2>AI Executive Summary</h2>
                <p style="font-size: 14px; line-height: 1.6; color: #334155;">{summary_text}</p>
                <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <strong style="font-size: 13px; color: #1e293b;">Top Coverages:</strong>
                        <pre style="font-family: inherit; font-size: 12px; color: #475569; white-space: pre-wrap; margin-top: 5px;">{coverage}</pre>
                    </div>
                    <div>
                        <strong style="font-size: 13px; color: #1e293b;">Top Exclusions:</strong>
                        <pre style="font-family: inherit; font-size: 12px; color: #475569; white-space: pre-wrap; margin-top: 5px;">{exclusions}</pre>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Extracted Policy Parameters</h2>
                <div class="field-grid">
                    {fields_list}
                </div>
            </div>
            
            <div class="section">
                <h2>Critical Risk Audit</h2>
                {risks_list}
            </div>
        </div>
    </body>
    </html>
    """
    return html


@router.get("/{id}/export", response_class=HTMLResponse)
async def export_report(
    id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Export policy analysis report as a formatted printable HTML/PDF attachment."""
    result = await db.execute(
        select(Document)
        .where(Document.id == id, Document.user_id == current_user.id)
        .options(
            selectinload(Document.summary),
            selectinload(Document.extracted_fields),
            selectinload(Document.risk_analyses),
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Explicitly load relationships to avoid lazy loading MissingGreenlet errors in sync generator
    await db.refresh(doc, ["summary", "extracted_fields", "risk_analyses"])
    
    html_content = generate_html_report(doc)
    headers = {"Content-Disposition": f"attachment; filename=HealthAI_Report_{doc.id}.html"}
    return HTMLResponse(content=html_content, headers=headers)


@router.post("/{id}/email")
async def email_report(
    id: str,
    request: EmailReportRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Email the formatted HTML policy audit report directly to the user."""
    result = await db.execute(
        select(Document)
        .where(Document.id == id, Document.user_id == current_user.id)
        .options(
            selectinload(Document.summary),
            selectinload(Document.extracted_fields),
            selectinload(Document.risk_analyses),
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Explicitly load relationships to avoid lazy loading MissingGreenlet errors in sync generator
    await db.refresh(doc, ["summary", "extracted_fields", "risk_analyses"])
    
    html_content = generate_html_report(doc)
    
    # 1. Setup email structure
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[HealthAI] Policy Analysis Audit Report: {doc.original_filename}"
    msg["From"] = "noreply@healthai.local"
    msg["To"] = request.email
    
    # Plaintext fallback
    text_fallback = f"Dear User,\n\nPlease find attached the HealthAI policy analysis report for {doc.original_filename}."
    part1 = MIMEText(text_fallback, "plain")
    part2 = MIMEText(html_content, "html")
    msg.attach(part1)
    msg.attach(part2)
    
    # 2. Try sending SMTP or write to local debug folder
    sent_successfully = False
    error_msg = ""
    try:
        # Check if email configs are set up in environment, otherwise log
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        if smtp_server and smtp_user and smtp_password:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(msg["From"], msg["To"], msg.as_string())
            sent_successfully = True
            logger.info(f"📧 Email report successfully sent via SMTP to {request.email}")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to send email via SMTP: {e}")
        
    # Write to local debug logs folder
    debug_dir = "./logs/sent_emails"
    os.makedirs(debug_dir, exist_ok=True)
    debug_filepath = f"{debug_dir}/email_{doc.id}_{uuid.uuid4().hex[:6]}.html"
    try:
        with open(debug_filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"💾 Logged outgoing email report locally to: {debug_filepath}")
    except Exception as io_err:
        logger.error(f"Failed to write email debug log: {io_err}")
        
    if sent_successfully:
        return {"status": "sent", "message": f"Report successfully emailed to {request.email}."}
    else:
        return {
            "status": "logged",
            "message": f"SMTP is not configured in local development environment. Outgoing report logged locally to: {debug_filepath}.",
            "details": error_msg
        }


