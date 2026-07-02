"""
OCR & Text Extraction Service
Handles text extraction from PDFs (PyMuPDF) and optionally images (PaddleOCR)
PaddleOCR is optional — falls back gracefully if not installed.
"""

import os
from typing import Tuple
from loguru import logger 


def extract_text_from_pdf(file_path: str) -> Tuple[str, str, int]:
    """
    Extract text from a digital PDF using PyMuPDF.
    Returns: (extracted_text, method, page_count)
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        page_count = len(doc)
        text_parts = []

        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")

        doc.close()
        full_text = "\n\n".join(text_parts)

        # If very little text was extracted, try OCR
        if len(full_text.strip()) < 100:
            logger.info("PDF appears scanned, attempting OCR...")
            try:
                return extract_text_with_ocr(file_path, page_count)
            except Exception as ocr_err:
                logger.warning(f"OCR fallback failed: {ocr_err}. Using empty text.")
                return full_text or "Could not extract text from this document.", "pymupdf", page_count

        logger.info(f"✅ PyMuPDF extracted {len(full_text)} chars from {page_count} pages")
        return full_text, "pymupdf", page_count

    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        raise


def extract_text_with_ocr(file_path: str, page_count: int = 1) -> Tuple[str, str, int]:
    """
    Extract text using PaddleOCR (optional).
    Raises ImportError if PaddleOCR not installed.
    """
    try:
        from paddleocr import PaddleOCR
        import fitz

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        text_parts = []

        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            page_count = len(doc)
            for page_num in range(page_count):
                page = doc[page_num]
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                result = ocr.ocr(img_bytes, cls=True)
                if result and result[0]:
                    page_text = "\n".join(
                        [line[1][0] for line in result[0] if line[1][0].strip()]
                    )
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            doc.close()
        else:
            result = ocr.ocr(file_path, cls=True)
            if result and result[0]:
                page_text = "\n".join(
                    [line[1][0] for line in result[0] if line[1][0].strip()]
                )
                text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)
        logger.info(f"✅ PaddleOCR extracted {len(full_text)} chars")
        return full_text, "paddleocr", page_count

    except ImportError:
        raise ImportError("PaddleOCR not installed. Install paddleocr for scanned document support.")


def extract_text_from_image(file_path: str) -> Tuple[str, str, int]:
    """Extract text from an image file."""
    try:
        return extract_text_with_ocr(file_path, 1)
    except ImportError:
        # Graceful fallback
        logger.warning("PaddleOCR not available — returning placeholder for image")
        return (
            "Image text extraction requires PaddleOCR. Install it with: pip install paddleocr paddlepaddle",
            "unavailable",
            1,
        )


def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""

    import re

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r" {3,}", "  ", text)
    return text.strip()


async def extract_document_text(file_path: str, file_type: str) -> Tuple[str, str, int]:
    """
    Main entry point for text extraction.
    Runs synchronous extractors in a thread pool so they don't block the async event loop.
    Returns: (cleaned_text, method, page_count)
    """
    import asyncio
    from functools import partial

    logger.info(f"Extracting text from: {file_path} (type: {file_type})")

    loop = asyncio.get_event_loop()

    if file_type == "pdf":
        from app.services.pdf_processor import extract_text_from_pdf as fast_extract
        # Run blocking I/O in thread pool — avoids stalling the event loop
        cleaned, method, page_count = await loop.run_in_executor(
            None, fast_extract, file_path
        )
    else:
        raw_text, method, page_count = await loop.run_in_executor(
            None, extract_text_from_image, file_path
        )
        cleaned = clean_extracted_text(raw_text)

    logger.info(f"✅ Extraction complete: {len(cleaned)} chars via {method}")
    return cleaned, method, page_count
