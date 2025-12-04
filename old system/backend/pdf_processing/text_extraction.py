import io
import logging
from typing import List, Dict
import pypdf
import warnings

logger = logging.getLogger(__name__)

# Suppress PDF warnings about malformed objects
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def extract_text_with_pages(pdf_bytes: bytes) -> List[Dict]:
    """
    Extract text from PDF preserving page numbers.

    Args:
        pdf_bytes: Raw PDF file bytes

    Returns:
        List of dicts with page_number and text
    """
    try:
        # Suppress pypdf warnings about malformed PDFs
        import sys
        from contextlib import redirect_stderr
        from io import StringIO
        
        stderr_capture = StringIO()
        
        with redirect_stderr(stderr_capture):
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            pages_text = []

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text().strip()
                    print(f"TEXT_EXTRACTION_DEBUG: Page {page_num} text length: {len(text)}")
                    if text:
                        print(f"TEXT_EXTRACTION_DEBUG: Page {page_num} first 100 chars: {text[:100]}")
                    else:
                        print(f"TEXT_EXTRACTION_DEBUG: Page {page_num} has no extractable text (likely scanned)")
                    pages_text.append({
                        "page_number": page_num,
                        "text": text
                    })
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    pages_text.append({
                        "page_number": page_num,
                        "text": f"[Page {page_num} - Text extraction failed]"
                    })

        # Log any suppressed stderr warnings only if they're serious
        captured_errors = stderr_capture.getvalue()
        if captured_errors and "error" in captured_errors.lower():
            logger.debug(f"PDF processing warnings (non-critical): {captured_errors[:200]}")

        logger.info(f"Extracted text from {len(pages_text)} pages")
        return pages_text

    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise

def extract_text_with_line_references(pdf_bytes: bytes) -> List[Dict]:
    """
    Extract text from PDF with detailed line and position references.
    
    Args:
        pdf_bytes: Raw PDF file bytes
    
    Returns:
        List of dicts with page_number, text, and line_data for precise referencing
    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages_data = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text().strip()
            lines = text.split('\n')
            
            # Create line references
            line_data = []
            for line_idx, line_text in enumerate(lines, 1):
                if line_text.strip():  # Only include non-empty lines
                    line_data.append({
                        "line_number": line_idx,
                        "text": line_text.strip(),
                        "character_start": sum(len(l) + 1 for l in lines[:line_idx-1]),
                        "character_end": sum(len(l) + 1 for l in lines[:line_idx]) - 1
                    })
            
            pages_data.append({
                "page_number": page_num,
                "text": text,
                "line_count": len([l for l in lines if l.strip()]),
                "line_data": line_data
            })
        
        logger.info(f"Extracted text with line references from {len(pages_data)} pages")
        return pages_data
        
    except Exception as e:
        logger.error(f"Failed to extract text with line references: {e}")
        raise

def validate_pdf_content(pdf_bytes: bytes) -> Dict:
    """
    Validate PDF and return basic info.

    Args:
        pdf_bytes: Raw PDF file bytes

    Returns:
        Dict with validation results and basic info
    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))

        info = {
            "valid": True,
            "page_count": len(reader.pages),
            "encrypted": reader.is_encrypted,
            "size_kb": len(pdf_bytes) / 1024,
            "error": None
        }

        if len(reader.pages) > 0:
            first_page_text = reader.pages[0].extract_text().strip()
            info["has_text"] = len(first_page_text) > 0
            info["sample_text"] = first_page_text[:200] + "..." if len(first_page_text) > 200 else first_page_text
        else:
            info["has_text"] = False
            info["sample_text"] = ""

        return info

    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "page_count": 0,
            "encrypted": None,
            "size_kb": len(pdf_bytes) / 1024,
            "has_text": False,
            "sample_text": ""
        }