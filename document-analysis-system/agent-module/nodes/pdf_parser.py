"""
PDF Parser Node
Extracts text content from PDF files.
"""

import base64
from typing import Dict, Any
import PyPDF2
from io import BytesIO

from settings import settings
from utils.logger import get_logger
from utils.metrics import track_node_execution

logger = get_logger(__name__)


async def parse_pdf_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse PDF and extract text content.

    Args:
        state: Graph state containing pdf_bytes (base64 encoded PDF)

    Returns:
        Updated state with extracted_text and page_count
    """
    with track_node_execution("pdf_parser"):
        try:
            logger.info("Starting PDF parsing")

            # Get PDF bytes from state
            pdf_base64 = state.get("pdf_bytes")
            if not pdf_base64:
                raise ValueError("No PDF data found in state")

            # Decode base64
            pdf_bytes = base64.b64decode(pdf_base64)
            pdf_file = BytesIO(pdf_bytes)

            # Parse PDF
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            page_count = len(pdf_reader.pages)

            logger.info(f"PDF has {page_count} pages")

            # Extract text from all pages
            extracted_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                    extracted_text += f"\n--- Page {page_num + 1} (extraction failed) ---\n"

            # Get metadata if available
            metadata = {}
            if pdf_reader.metadata:
                metadata = {
                    "title": pdf_reader.metadata.get("/Title", ""),
                    "author": pdf_reader.metadata.get("/Author", ""),
                    "subject": pdf_reader.metadata.get("/Subject", ""),
                    "creator": pdf_reader.metadata.get("/Creator", ""),
                }

            text_length = len(extracted_text)
            word_count = len(extracted_text.split())

            logger.info(
                "PDF parsing completed",
                extra={
                    "page_count": page_count,
                    "text_length": text_length,
                    "word_count": word_count,
                    "has_metadata": bool(metadata)
                }
            )

            return {
                **state,
                "extracted_text": extracted_text,
                "page_count": page_count,
                "word_count": word_count,
                "pdf_metadata": metadata,
                "parsing_success": True,
                "parsing_error": None
            }

        except Exception as e:
            logger.error(
                f"PDF parsing failed: {str(e)}",
                extra={"error_type": type(e).__name__}
            )

            return {
                **state,
                "extracted_text": "",
                "page_count": 0,
                "word_count": 0,
                "pdf_metadata": {},
                "parsing_success": False,
                "parsing_error": str(e)
            }


async def validate_pdf_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate PDF data before parsing.

    Args:
        state: Graph state

    Returns:
        Updated state with validation results
    """
    with track_node_execution("pdf_validator"):
        try:
            pdf_base64 = state.get("pdf_bytes")

            if not pdf_base64:
                return {
                    **state,
                    "validation_success": False,
                    "validation_error": "No PDF data provided"
                }

            # Decode to check size
            pdf_bytes = base64.b64decode(pdf_base64)
            size_mb = len(pdf_bytes) / (1024 * 1024)

            max_size_mb = settings.max_request_size_mb

            if size_mb > max_size_mb:
                logger.warning(
                    f"PDF size ({size_mb:.2f} MB) exceeds limit ({max_size_mb} MB)"
                )
                return {
                    **state,
                    "validation_success": False,
                    "validation_error": f"PDF too large: {size_mb:.2f} MB (max: {max_size_mb} MB)"
                }

            # Check if it's a valid PDF (starts with %PDF)
            if not pdf_bytes.startswith(b'%PDF'):
                logger.warning("Invalid PDF: does not start with %PDF header")
                return {
                    **state,
                    "validation_success": False,
                    "validation_error": "Invalid PDF file format"
                }

            logger.info(
                "PDF validation successful",
                extra={"size_mb": f"{size_mb:.2f}"}
            )

            return {
                **state,
                "validation_success": True,
                "validation_error": None,
                "pdf_size_mb": size_mb
            }

        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return {
                **state,
                "validation_success": False,
                "validation_error": str(e)
            }
