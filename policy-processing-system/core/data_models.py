"""
Core data models used across the processing pipeline.
These simple data classes represent documents, pages, and chunks.
"""
from typing import List, Dict, Any
from PIL import Image


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


class DocumentChunk:
    """Represents a chunk of a document."""

    def __init__(
        self,
        chunk_id: int,
        text: str,
        start_page: int,
        end_page: int,
        section_context: str,
        token_count: int,
        metadata: Dict[str, Any],
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.start_page = start_page
        self.end_page = end_page
        self.section_context = section_context
        self.token_count = token_count
        self.metadata = metadata
