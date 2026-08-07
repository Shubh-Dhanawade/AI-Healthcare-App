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


_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        # Initialize reader (gpu=False for safe CPU fallback on Windows/Mac)
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _easyocr_reader


_paddleocr_instance = None

def get_paddleocr_reader():
    global _paddleocr_instance
    if _paddleocr_instance is None:
        from paddleocr import PaddleOCR
        _paddleocr_instance = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _paddleocr_instance



def extract_text_with_easyocr(file_path: str, page_count: int = 1) -> Tuple[str, str, int]:
    """Extract text using EasyOCR."""
    try:
        import fitz
        reader = get_easyocr_reader()
        text_parts = []

        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            page_count = len(doc)
            for page_num in range(page_count):
                page = doc[page_num]
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                
                result = reader.readtext(img_bytes, detail=0)
                if result:
                    page_text = "\n".join(result)
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            doc.close()
        else:
            result = reader.readtext(file_path, detail=0)
            if result:
                page_text = "\n".join(result)
                text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)
        logger.info(f"✅ EasyOCR extracted {len(full_text)} chars")
        return full_text, "easyocr", page_count
    except Exception as e:
        logger.error(f"EasyOCR extraction failed: {e}")
        raise


def extract_text_with_pytesseract(file_path: str, page_count: int = 1) -> Tuple[str, str, int]:
    """Extract text using PyTesseract (requires Tesseract binary installed)."""
    try:
        import pytesseract
        import fitz
        from PIL import Image
        import io

        # Fast verification that tesseract is installed & configured on system PATH
        pytesseract.get_tesseract_version()

        text_parts = []

        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            page_count = len(doc)
            for page_num in range(page_count):
                page = doc[page_num]
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                
                img = Image.open(io.BytesIO(img_bytes))
                page_text = pytesseract.image_to_string(img)
                if page_text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text.strip()}")
            doc.close()
        else:
            img = Image.open(file_path)
            page_text = pytesseract.image_to_string(img)
            if page_text.strip():
                text_parts.append(page_text.strip())

        full_text = "\n\n".join(text_parts)
        logger.info(f"✅ PyTesseract extracted {len(full_text)} chars")
        return full_text, "pytesseract", page_count
    except Exception as e:
        logger.error(f"PyTesseract extraction failed: {e}")
        raise


def extract_text_with_ocr(file_path: str, page_count: int = 1) -> Tuple[str, str, int]:
    """
    Unified entry point for OCR extraction.
    Tries multiple OCR engines sequentially:
    1. PaddleOCR (Optional)
    2. EasyOCR (Optional)
    3. PyTesseract (Optional, requires Tesseract binary)
    """
    errors = []

    # 1. Try PaddleOCR
    try:
        import fitz

        logger.info("Attempting OCR with PaddleOCR...")
        ocr = get_paddleocr_reader()
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

    except (ImportError, Exception) as e:
        errors.append(f"PaddleOCR failed: {e}")
        logger.debug(f"PaddleOCR is not available or failed: {e}")

    # 2. Try EasyOCR
    try:
        logger.info("Attempting OCR fallback with EasyOCR...")
        return extract_text_with_easyocr(file_path, page_count)
    except (ImportError, Exception) as e:
        errors.append(f"EasyOCR failed: {e}")
        logger.debug(f"EasyOCR is not available or failed: {e}")

    # 3. Try PyTesseract
    try:
        logger.info("Attempting OCR fallback with PyTesseract...")
        return extract_text_with_pytesseract(file_path, page_count)
    except (ImportError, Exception) as e:
        errors.append(f"PyTesseract failed: {e}")
        logger.debug(f"PyTesseract is not available or failed: {e}")

    # If all engines fail, raise a combined exception
    err_msg = "; ".join(errors)
    raise RuntimeError(f"All OCR engines failed. Details: {err_msg}")


def extract_text_from_image(file_path: str) -> Tuple[str, str, int]:
    """Extract text from an image file."""
    try:
        return extract_text_with_ocr(file_path, 1)
    except Exception as e:
        # Graceful fallback
        logger.warning(f"All OCR engines failed: {e}")
        return (
            "Image text extraction requires PaddleOCR, EasyOCR, or PyTesseract. "
            "Please ensure at least one OCR library is installed and configured.",
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
