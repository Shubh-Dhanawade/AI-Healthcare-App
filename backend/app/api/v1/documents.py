"""Document upload and management API endpoints."""

import os
import uuid
from pathlib import Path as FilePath
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from loguru import logger

from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User
from app.models.document import Document, ExtractedField
from app.models.risk_analysis import Summary, RiskAnalysis
from app.schemas.schemas import DocumentResponse, DocumentDetailResponse, CompareRequest, CompareResponse, ComparisonSynthesisSchema
from app.services.ocr_service import extract_document_text
from app.services.ai_service import generate_comparison_synthesis

router = APIRouter()

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
    """Background task to extract text from uploaded document."""
    from app.core.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as db:
        try:
            # Get document
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if not doc:
                return
            
            # Update status
            doc.status = "processing"
            await db.commit()
            
            # Extract text
            text, method, page_count = await extract_document_text(file_path, file_type)
            
            # Save results
            doc.extracted_text = text
            doc.extraction_method = method
            doc.page_count = page_count
            doc.status = "text_extracted"
            await db.commit()
            
            logger.info(f"✅ Document {doc_id} text extracted via {method}")
            
        except Exception as e:
            logger.error(f"Background text extraction failed for {doc_id}: {e}")
            async with AsyncSessionLocal() as err_db:
                result = await err_db.execute(select(Document).where(Document.id == doc_id))
                doc = result.scalar_one_or_none()
                if doc:
                    doc.status = "failed"
                    await err_db.commit()


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

