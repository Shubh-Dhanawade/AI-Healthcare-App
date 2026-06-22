"""
PDF Processing Service
Optimized text extraction from digital PDFs using PyMuPDF, with blank page detection,
whitespace cleaning, header/footer removal heuristics, and PaddleOCR fallback.
"""

import os
import re
from typing import Tuple, List
from loguru import logger
from app.services.cache_manager import CacheManager

def clean_duplicate_whitespace(text: str) -> str:
    """Normalize whitespace by collapsing multiple spaces and blank lines."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse multiple consecutive spaces (3 or more) into 2 spaces
    text = re.sub(r" {3,}", "  ", text)
    return text.strip()

def remove_headers_footers_heuristics(page_text: str, page_num: int) -> str:
    """Remove common header/footer layouts like page numbers and generic running headers."""
    lines = page_text.split("\n")
    cleaned_lines = []
    
    # Heuristics to detect running footers/headers or simple page indicators
    page_num_patterns = [
        re.compile(r"^\s*page\s*\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE),
        re.compile(r"^\s*\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE),
        re.compile(r"^\s*-\s*\d+\s*-\s*$", re.IGNORECASE)
    ]
    
    for idx, line in enumerate(lines):
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            cleaned_lines.append("")
            continue
            
        # 1. Skip page number indicators if they appear at start (header) or end (footer) of page text
        is_page_num = any(pat.match(stripped) for pat in page_num_patterns)
        if is_page_num and (idx < 2 or idx > len(lines) - 3):
            continue
            
        # 2. Skip running headers/footers containing illustrative or document titles
        is_generic_title = "basic health insurance" in stripped.lower() or "illustrative only" in stripped.lower()
        if is_generic_title and (idx < 2 or idx > len(lines) - 3):
            continue
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def extract_text_from_pdf(file_path: str) -> Tuple[str, str, int]:
    """
    Extract text using PyMuPDF. Skips blank pages, cleans layout, and removes headers/footers.
    Falls back to OCR if the document appears to be scanned (contains very little text).
    """
    import fitz  # PyMuPDF
    
    logger.info(f"Opening PDF file: {file_path}")
    doc = fitz.open(file_path)
    page_count = len(doc)
    text_parts: List[str] = []
    
    for page_num in range(page_count):
        page = doc[page_num]
        # Use blocks to preserve layout/tables where possible
        text = page.get_text("blocks")
        
        # Sort blocks top-to-bottom, left-to-right
        text.sort(key=lambda b: (b[1], b[0]))
        
        page_lines = []
        for b in text:
            block_text = b[4].strip()
            if block_text:
                page_lines.append(block_text)
                
        raw_page_text = "\n".join(page_lines)
        
        # Skip completely blank pages
        if not raw_page_text.strip():
            logger.info(f"Skipping blank page {page_num + 1}")
            continue
            
        # Clean page numbers, headers, and footers
        cleaned_page_text = remove_headers_footers_heuristics(raw_page_text, page_num + 1)
        
        if cleaned_page_text.strip():
            text_parts.append(f"[Page {page_num + 1}]\n{cleaned_page_text}")
            
    doc.close()
    full_text = "\n\n".join(text_parts)
    cleaned_full_text = clean_duplicate_whitespace(full_text)
    
    # Scanned document detection: if character count is extremely low (<100 characters per page average)
    if len(cleaned_full_text.strip()) < 100 * page_count:
        logger.info("PDF appears to be scanned or contains image-only pages. Falling back to PaddleOCR...")
        try:
            from app.services.ocr_service import extract_text_with_ocr
            return extract_text_with_ocr(file_path, page_count)
        except Exception as ocr_err:
            logger.warning(f"OCR fallback failed: {ocr_err}. Returning best-effort PyMuPDF text.")
            return cleaned_full_text or "Could not extract text from this document.", "pymupdf", page_count
            
    logger.info(f"✅ PyMuPDF extracted {len(cleaned_full_text)} chars from {page_count} pages")
    return cleaned_full_text, "pymupdf", page_count
