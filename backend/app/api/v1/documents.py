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
from app.schemas.schemas import DocumentResponse, DocumentDetailResponse
from app.services.ocr_service import extract_document_text

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
