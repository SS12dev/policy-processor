"""
PDF processing module for text and image extraction.
"""
import base64
import io
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import PyPDF2
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from app.utils.logger import get_logger
from app.models.schemas import SourceReference

logger = get_logger(__name__)


class PDFPage:
    """Represents a single page from a PDF document."""

    def __init__(
        self,
        page_number: int,
        text: str,
        images: List[Image.Image],
        tables: List[List[List[str]]],
        layout_info: Dict[str, Any],
    ):
        self.page_number = page_number
        self.text = text
        self.images = images
        self.tables = tables
        self.layout_info = layout_info


class PDFProcessor:
    """Processes PDF documents with advanced text and image extraction."""

    def __init__(self):
        """Initialize PDF processor."""
        self.current_document = None
        self.document_hash = None

    def process_document(self, base64_pdf: str) -> Tuple[List[PDFPage], Dict[str, Any]]:
        """
        Process a base64-encoded PDF document.

        Args:
            base64_pdf: Base64-encoded PDF string

        Returns:
            Tuple of (list of PDFPage objects, document metadata)
        """
        try:
            # Decode base64
            pdf_bytes = base64.b64decode(base64_pdf)
            self.document_hash = hashlib.sha256(pdf_bytes).hexdigest()

            logger.info(f"Processing PDF document (hash: {self.document_hash[:8]}...)")

            # Extract metadata
            metadata = self._extract_metadata(pdf_bytes)

            # Process pages
            pages = self._process_pages(pdf_bytes)

            logger.info(f"Successfully processed {len(pages)} pages")

            return pages, metadata

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    def _extract_metadata(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Extract metadata from PDF.

        Args:
            pdf_bytes: PDF file bytes

        Returns:
            Dictionary containing metadata
        """
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            reader = PyPDF2.PdfReader(pdf_file)

            metadata = {
                "total_pages": len(reader.pages),
                "has_images": False,
                "has_tables": False,
                "is_scanned": False,
                "pdf_info": {},
            }

            # Extract PDF info
            if reader.metadata:
                metadata["pdf_info"] = {
                    "title": reader.metadata.get("/Title", ""),
                    "author": reader.metadata.get("/Author", ""),
                    "subject": reader.metadata.get("/Subject", ""),
                    "creator": reader.metadata.get("/Creator", ""),
                }

            return metadata

        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return {"total_pages": 0, "error": str(e)}

    def _process_pages(self, pdf_bytes: bytes) -> List[PDFPage]:
        """
        Process all pages in the PDF.

        Args:
            pdf_bytes: PDF file bytes

        Returns:
            List of PDFPage objects
        """
        pages = []
        pdf_file = io.BytesIO(pdf_bytes)

        try:
            # Use pdfplumber for text and table extraction
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    logger.debug(f"Processing page {page_num}/{len(pdf.pages)}")

                    # Extract text
                    text = page.extract_text() or ""

                    # Extract tables
                    tables = page.extract_tables() or []

                    # Get layout information
                    layout_info = self._extract_layout_info(page)

                    # Extract images (placeholder - would need additional logic)
                    images = []

                    # Check if page is likely scanned (very little text)
                    if len(text.strip()) < 50 and page_num <= len(pdf.pages):
                        logger.info(f"Page {page_num} appears to be scanned, attempting OCR")
                        ocr_text = self._perform_ocr(pdf_bytes, page_num)
                        if ocr_text:
                            text = ocr_text

                    pdf_page = PDFPage(
                        page_number=page_num,
                        text=text,
                        images=images,
                        tables=tables,
                        layout_info=layout_info,
                    )

                    pages.append(pdf_page)

            return pages

        except Exception as e:
            logger.error(f"Error processing pages: {e}")
            raise

    def _extract_layout_info(self, page) -> Dict[str, Any]:
        """
        Extract layout information from a page.

        Args:
            page: pdfplumber page object

        Returns:
            Dictionary with layout information
        """
        try:
            layout = {
                "width": page.width,
                "height": page.height,
                "has_headers": False,
                "has_footers": False,
                "column_count": 1,
                "orientation": "portrait" if page.height > page.width else "landscape",
            }

            # Detect potential headers/footers by analyzing text position
            words = page.extract_words()
            if words:
                # Check for consistent text at top (header)
                top_words = [w for w in words if w["top"] < page.height * 0.1]
                if top_words:
                    layout["has_headers"] = True

                # Check for consistent text at bottom (footer)
                bottom_words = [w for w in words if w["top"] > page.height * 0.9]
                if bottom_words:
                    layout["has_footers"] = True

            return layout

        except Exception as e:
            logger.warning(f"Error extracting layout info: {e}")
            return {}

    def _perform_ocr(self, pdf_bytes: bytes, page_number: int) -> Optional[str]:
        """
        Perform OCR on a scanned page.

        Args:
            pdf_bytes: PDF file bytes
            page_number: Page number to OCR (1-indexed)

        Returns:
            Extracted text or None
        """
        try:
            # Convert PDF page to image
            images = convert_from_bytes(
                pdf_bytes,
                first_page=page_number,
                last_page=page_number,
                dpi=300,
            )

            if images:
                # Perform OCR
                text = pytesseract.image_to_string(images[0], lang="eng")
                logger.info(f"OCR extracted {len(text)} characters from page {page_number}")
                return text

            return None

        except Exception as e:
            logger.warning(f"OCR failed for page {page_number}: {e}")
            return None

    def extract_structure(self, pages: List[PDFPage]) -> Dict[str, Any]:
        """
        Analyze document structure from extracted pages.

        Args:
            pages: List of PDFPage objects

        Returns:
            Dictionary describing document structure
        """
        structure = {
            "has_numbered_sections": False,
            "has_hierarchy": False,
            "section_pattern": None,
            "has_toc": False,
            "has_index": False,
            "has_appendices": False,
        }

        # Analyze first few pages for structure
        full_text = "\n".join([p.text for p in pages[:5]])

        # Check for numbered sections (e.g., "1.", "1.1", "Article 1")
        import re

        numbered_patterns = [
            r"^\s*\d+\.\s+[A-Z]",  # "1. Section"
            r"^\s*\d+\.\d+\s+[A-Z]",  # "1.1 Subsection"
            r"^\s*Article\s+\d+",  # "Article 1"
            r"^\s*Section\s+\d+",  # "Section 1"
            r"^\s*Chapter\s+\d+",  # "Chapter 1"
        ]

        for pattern in numbered_patterns:
            if re.search(pattern, full_text, re.MULTILINE):
                structure["has_numbered_sections"] = True
                structure["section_pattern"] = pattern
                break

        # Check for table of contents
        if re.search(r"table\s+of\s+contents", full_text, re.IGNORECASE):
            structure["has_toc"] = True

        # Check for appendices
        if re.search(r"appendix", full_text, re.IGNORECASE):
            structure["has_appendices"] = True

        # Determine if document has clear hierarchy
        if structure["has_numbered_sections"] or structure["has_toc"]:
            structure["has_hierarchy"] = True

        return structure

    def create_source_reference(
        self, page_number: int, section: str, quoted_text: str
    ) -> SourceReference:
        """
        Create a source reference object.

        Args:
            page_number: Page number
            section: Section identifier
            quoted_text: Quoted text from source

        Returns:
            SourceReference object
        """
        return SourceReference(
            page_number=page_number,
            section=section,
            quoted_text=quoted_text[:500],  # Limit to 500 chars
        )

    def get_page_text_with_context(
        self, pages: List[PDFPage], page_number: int, context_pages: int = 1
    ) -> str:
        """
        Get text from a page with surrounding context.

        Args:
            pages: List of all pages
            page_number: Target page number (1-indexed)
            context_pages: Number of pages before/after to include

        Returns:
            Combined text with context
        """
        start_page = max(0, page_number - context_pages - 1)
        end_page = min(len(pages), page_number + context_pages)

        context_text = []
        for i in range(start_page, end_page):
            if i < len(pages):
                context_text.append(f"--- Page {pages[i].page_number} ---\n{pages[i].text}")

        return "\n\n".join(context_text)
