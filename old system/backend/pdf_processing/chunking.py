"""
PDF Text Chunking module for processing extracted text into manageable segments.
Unified chunking system with configurable parameters for memory optimization.
"""

import gc
import logging
from typing import List, Dict, Optional, Union, Any
from backend.settings import BackendSettings

logger = logging.getLogger(__name__)
settings = BackendSettings()

class ChunkingStrategy:
    """Configuration class for chunking strategies."""
    
    def __init__(
        self,
        pages_per_chunk: int = None,
        overlap_pages: int = None,
        max_chars_per_chunk: int = None,
        char_overlap: int = None
    ):
        self.pages_per_chunk = pages_per_chunk if pages_per_chunk is not None else settings.POLICY_PAGES_PER_CHUNK
        self.overlap_pages = overlap_pages if overlap_pages is not None else settings.POLICY_CHUNK_OVERLAP_PAGES
        self.max_chars_per_chunk = max_chars_per_chunk if max_chars_per_chunk is not None else settings.POLICY_MAX_CHARS_PER_CHUNK
        self.char_overlap = char_overlap if char_overlap is not None else settings.POLICY_CHAR_OVERLAP

def create_unified_chunks(
    content: Union[str, List[Dict[str, str]]], 
    strategy: Optional[ChunkingStrategy] = None,
    doc_type: str = "policy",
    force_method: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Unified chunking function that handles both text and page-based chunking.
    
    Args:
        content: Either text string or list of page dictionaries
        strategy: ChunkingStrategy configuration
        doc_type: Document type ("policy" or "application")
        force_method: Force specific method ("semantic" or "page")
    
    Returns:
        List of chunk dictionaries with 'content' key
    """
    logger.debug(f"create_unified_chunks called with content type: {type(content)}, length: {len(content) if content else 'None'}")
    logger.debug(f"doc_type: {doc_type}, force_method: {force_method}")
    
    # Add comprehensive debugging
    print(f"CHUNKING_DEBUG: create_unified_chunks called")
    print(f"CHUNKING_DEBUG: Content type: {type(content)}")
    print(f"CHUNKING_DEBUG: Content length: {len(content) if content else 'None'}")
    print(f"CHUNKING_DEBUG: Doc type: {doc_type}")
    
    if isinstance(content, list) and content:
        print(f"CHUNKING_DEBUG: First page keys: {list(content[0].keys()) if content[0] else 'No first page'}")
        for i, page in enumerate(content[:3]):  # Show first 3 pages
            text_content = page.get("text", "NO_TEXT")
            print(f"CHUNKING_DEBUG: Page {i+1} text preview: {text_content[:100] if text_content else 'EMPTY'}...")

    if not content:
        logger.warning("No content provided for chunking, returning empty list")
        return []
    
    # Initialize strategy based on document type
    if strategy is None:
        if doc_type == "application":
            # Check if this document has images (which increase request size significantly)
            if isinstance(content, list) and content:
                total_text = sum(len(page.get("text", "").strip()) for page in content)
                total_images = sum(1 for page in content if page.get('image_base64') or page.get('image'))
                
                # Be very conservative with image-heavy documents to avoid 413 errors
                if total_images > 0:
                    print(f"CHUNKING_DEBUG: Detected {total_images} images, using conservative chunking to avoid 413 errors")
                    
                    # Use configurable max pages for images to avoid request size limits
                    pages_per_chunk = settings.APP_MAX_PAGES_FOR_IMAGES
                    overlap_pages = 0  # No overlap for image chunks to minimize size
                    
                    print(f"CHUNKING_DEBUG: Using {pages_per_chunk} pages per chunk for image document")
                    
                    strategy = ChunkingStrategy(
                        pages_per_chunk=pages_per_chunk,
                        overlap_pages=overlap_pages
                    )
                else:
                    # Text-only documents can use larger chunks
                    print(f"CHUNKING_DEBUG: Text-only document, using standard chunking")
                    strategy = ChunkingStrategy(
                        pages_per_chunk=settings.APP_PAGES_PER_CHUNK,
                        overlap_pages=settings.APP_CHUNK_OVERLAP_PAGES
                    )
            else:
                strategy = ChunkingStrategy(
                    pages_per_chunk=settings.APP_PAGES_PER_CHUNK,
                    overlap_pages=settings.APP_CHUNK_OVERLAP_PAGES
                )
        else:  # policy
            strategy = ChunkingStrategy()

    logger.debug(f"Chunking strategy: pages_per_chunk={strategy.pages_per_chunk}, overlap_pages={strategy.overlap_pages}")

    # Determine chunking method
    if force_method:
        method = force_method
    elif isinstance(content, str):
        method = "semantic"
    else:
        # Page-based for list of pages
        method = "page"
        
        # For applications, always use page-based chunking to avoid issues
        if doc_type == "application":
            method = "page"
            print(f"CHUNKING_DEBUG: Forcing page-based chunking for application document")
        else:
            # But switch to semantic if document is small (for policies)
            if len(content) <= settings.SMALL_DOC_PAGE_THRESHOLD:
                combined_text = "\n\n".join(page.get("text", "") for page in content)
                print(f"CHUNKING_DEBUG: Combined text length: {len(combined_text)}")
                print(f"CHUNKING_DEBUG: Combined text preview: {combined_text[:200] if combined_text else 'EMPTY'}...")
                if len(combined_text) <= settings.SMALL_DOC_CHAR_THRESHOLD:
                    content = combined_text
                    method = "semantic"
                    print(f"CHUNKING_DEBUG: Switched to semantic chunking")

    logger.info(f"Using {method} chunking method for {doc_type} document with {len(content) if isinstance(content, list) else 'text'} {'pages' if isinstance(content, list) else 'content'}")
    print(f"CHUNKING_DEBUG: Selected method: {method}")
    print(f"CHUNKING_DEBUG: Strategy - pages_per_chunk: {strategy.pages_per_chunk}, overlap: {strategy.overlap_pages}")
        
    # Execute appropriate chunking method
    try:
        if method == "semantic":
            # Handle semantic chunking for both string and page data
            if isinstance(content, str):
                text_content = content
                source_metadata = {
                    "chunk_id": "semantic_chunk_1", 
                    "start_page": 1,
                    "end_page": 1,
                    "pages_covered": [1],
                    "line_references": []
                }
            else:
                # Extract text from page data and build metadata
                text_content = "\n\n".join(page.get("text", "") for page in content)
                page_numbers = [page.get("page_number", i+1) for i, page in enumerate(content)]
                source_metadata = {
                    "chunk_id": f"semantic_chunk_pages_{min(page_numbers)}_{max(page_numbers)}",
                    "start_page": min(page_numbers),
                    "end_page": max(page_numbers), 
                    "pages_covered": page_numbers,
                    "line_references": []
                }
            
            chunks = _create_semantic_chunks(
                text=text_content,
                max_chars=strategy.max_chars_per_chunk,
                overlap=strategy.char_overlap,
                source_metadata=source_metadata
            )
        else:  # page
            chunks = _create_page_based_chunks(
                pages=content,
                pages_per_chunk=strategy.pages_per_chunk,
                overlap_pages=strategy.overlap_pages
            )
        # Validate chunk sizes for memory optimization
        print(f"CHUNKING_DEBUG: Created {len(chunks)} chunks before validation")
        validated_chunks = _validate_chunk_sizes(chunks)
        print(f"CHUNKING_DEBUG: {len(validated_chunks)} chunks after validation")
        
        # Memory cleanup
        if settings.ENABLE_MEMORY_CLEANUP:
            del content
            gc.collect()
        
        logger.info(f"Created {len(validated_chunks)} chunks using {method} method for {doc_type} document")
        return validated_chunks
        
    except Exception as e:
        logger.error(f"Error in chunking: {e}")
        return []

def _create_semantic_chunks(text: str, max_chars: int, overlap: int, source_metadata: Dict = None) -> List[Dict[str, str]]:
    """
    Create semantic chunks based on character count with intelligent sentence boundaries.
    Ensures all chunks have required metadata for orchestrator compatibility.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Character overlap between chunks
        source_metadata: Original metadata to inherit (page info, etc.)
    """
    if not text or not text.strip():
        print(f"SEMANTIC_CHUNKING_DEBUG: Text is empty or whitespace only")
        print(f"SEMANTIC_CHUNKING_DEBUG: Text repr: {repr(text)}")
        return []
    
    text = text.strip()
    
    # Default metadata if none provided
    default_metadata = {
        "chunk_id": "semantic_chunk_1",
        "start_page": 1,
        "end_page": 1,
        "pages_covered": [1],
        "line_references": []
    }
    
    base_metadata = source_metadata or default_metadata
    
    # Single chunk for small documents
    if len(text) <= max_chars:
        return [{
            "content": text,
            "text": text,  # Orchestrator compatibility
            "chunk_id": base_metadata.get("chunk_id", "semantic_chunk_1"),
            "start_page": base_metadata.get("start_page", 1),
            "end_page": base_metadata.get("end_page", 1),
            "pages_covered": base_metadata.get("pages_covered", [1]),
            "line_references": base_metadata.get("line_references", [])
        }]

    chunks = []
    start = 0
    chunk_counter = 1

    while start < len(text):
        end = min(start + max_chars, len(text))

        # Find intelligent break point for non-final chunks
        if end < len(text):
            end = _find_sentence_boundary(text, start, end)

        chunk_text = text[start:end].strip()
        if chunk_text:
            # Create chunk with all required metadata
            chunk_id = f"semantic_chunk_{chunk_counter}"
            if "chunk_id" in base_metadata:
                chunk_id = f"{base_metadata['chunk_id']}_split_{chunk_counter}"
            
            chunks.append({
                "content": chunk_text,
                "text": chunk_text,  # Orchestrator compatibility
                "chunk_id": chunk_id,
                "start_page": base_metadata.get("start_page", 1),
                "end_page": base_metadata.get("end_page", 1),
                "pages_covered": base_metadata.get("pages_covered", [1]),
                "line_references": base_metadata.get("line_references", [])
            })
            chunk_counter += 1

        if end >= len(text):
            break

        # Calculate next start with overlap
        start = max(start + 1, end - overlap)
    
    return chunks

def _create_page_based_chunks(pages: List[Dict[str, str]], pages_per_chunk: int, overlap_pages: int) -> List[Dict[str, str]]:
    """
    Create chunks by combining multiple pages with smart overlap management.
    Ensures clean page boundaries and prevents excessive overlapping.
    """
    logger.info(f"CHUNKING_DEBUG: _create_page_based_chunks called with {len(pages) if pages else 0} pages")
    logger.info(f"CHUNKING_DEBUG: pages_per_chunk: {pages_per_chunk}, overlap_pages: {overlap_pages}")
    
    print(f"PAGE_CHUNKING_DEBUG: _create_page_based_chunks called with {len(pages) if pages else 0} pages")
    print(f"PAGE_CHUNKING_DEBUG: pages_per_chunk: {pages_per_chunk}, overlap_pages: {overlap_pages}")
    
    if not pages:
        logger.warning(f"CHUNKING_DEBUG: No pages provided to _create_page_based_chunks")
        print(f"PAGE_CHUNKING_DEBUG: No pages provided")
        return []
    
    # Single chunk for small documents
    if len(pages) <= pages_per_chunk:
        print(f"PAGE_CHUNKING_DEBUG: Single chunk case - {len(pages)} pages <= {pages_per_chunk}")
        # Handle both "content" and "text" field names for compatibility
        combined_content = "\n\n".join(page.get("text", page.get("content", "")) for page in pages).strip()
        
        logger.info(f"CHUNKING_DEBUG: Single chunk case - {len(pages)} pages <= {pages_per_chunk}")
        logger.info(f"CHUNKING_DEBUG: Combined content length: {len(combined_content) if combined_content else 0}")
        logger.info(f"CHUNKING_DEBUG: First 100 chars: {combined_content[:100] if combined_content else 'EMPTY'}")
        
        print(f"PAGE_CHUNKING_DEBUG: Combined content length: {len(combined_content) if combined_content else 0}")
        print(f"PAGE_CHUNKING_DEBUG: First 100 chars: {combined_content[:100] if combined_content else 'EMPTY'}")
        
        if not combined_content:
            logger.warning(f"CHUNKING_DEBUG: Combined content is empty, checking if pages have images...")
            print(f"PAGE_CHUNKING_DEBUG: Combined content is empty!")
            
            # Check if pages have images even without text (scanned PDFs)
            pages_with_images = 0
            for i, page in enumerate(pages):
                has_image = bool(page.get('image_base64') or page.get('image'))
                print(f"PAGE_CHUNKING_DEBUG: Page {i+1} - has_image: {has_image}")
                if has_image:
                    pages_with_images += 1
                    
            print(f"PAGE_CHUNKING_DEBUG: {pages_with_images}/{len(pages)} pages have images")
            
            # If pages have images but no text, create chunks anyway for image processing
            if pages_with_images > 0:
                print(f"PAGE_CHUNKING_DEBUG: Creating image-only chunks for scanned document")
                page_numbers = [page.get("page_number", i+1) for i, page in enumerate(pages)]
                return [{
                    "content": "[Scanned document - processing images with AI vision]",
                    "text": "[Scanned document - processing images with AI vision]",
                    "chunk_id": f"image_chunk_pages_{min(page_numbers)}_{max(page_numbers)}",
                    "start_page": min(page_numbers),
                    "end_page": max(page_numbers),
                    "pages_covered": page_numbers,
                    "line_references": [],
                    "is_image_only": True  # Flag to indicate this chunk needs image processing
                }]
            else:
                print(f"PAGE_CHUNKING_DEBUG: No images or text found, returning empty")
                return []
        
        # Build metadata for single chunk
        page_numbers = [page.get("page_number", i+1) for i, page in enumerate(pages)]
        return [{
            "content": combined_content,
            "text": combined_content,  # Alias for orchestrator compatibility
            "chunk_id": f"single_chunk_pages_{min(page_numbers)}_{max(page_numbers)}",
            "start_page": min(page_numbers),
            "end_page": max(page_numbers),
            "pages_covered": page_numbers,
            "line_references": []
        }]
    
    chunks = []
    i = 0
    chunk_counter = 1
    
    while i < len(pages):
        end_idx = min(i + pages_per_chunk, len(pages))
        
        # Combine pages for this chunk
        chunk_pages = pages[i:end_idx]
        # Handle both "content" and "text" field names for compatibility
        combined_content = "\n\n".join(page.get("text", page.get("content", "")) for page in chunk_pages).strip()
        
        if combined_content:
            # Build page references for this chunk
            page_numbers = [page.get("page_number", i+j+1) for j, page in enumerate(chunk_pages)]
            
            chunk_data = {
                "content": combined_content,
                "text": combined_content,  # Alias for orchestrator compatibility
                "chunk_id": f"chunk_{chunk_counter}_pages_{min(page_numbers)}_{max(page_numbers)}",
                "start_page": min(page_numbers),
                "end_page": max(page_numbers),
                "pages_covered": page_numbers,
                "line_references": []
            }
            chunks.append(chunk_data)
            chunk_counter += 1
        else:
            # Check if chunk pages have images even without text
            pages_with_images = sum(1 for page in chunk_pages if page.get('image_base64') or page.get('image'))
            if pages_with_images > 0:
                page_numbers = [page.get("page_number", i+j+1) for j, page in enumerate(chunk_pages)]
                chunk_data = {
                    "content": "[Scanned document - processing images with AI vision]",
                    "text": "[Scanned document - processing images with AI vision]",
                    "chunk_id": f"image_chunk_{chunk_counter}_pages_{min(page_numbers)}_{max(page_numbers)}",
                    "start_page": min(page_numbers),
                    "end_page": max(page_numbers),
                    "pages_covered": page_numbers,
                    "line_references": [],
                    "is_image_only": True
                }
                chunks.append(chunk_data)
                chunk_counter += 1
                print(f"PAGE_CHUNKING_DEBUG: Created image-only chunk {chunk_counter-1} with {pages_with_images} images")
        
        # Smart advancement: reduce overlap if we're near the end
        remaining_pages = len(pages) - end_idx
        if remaining_pages <= overlap_pages:
            # Jump to end to avoid tiny chunks
            i = end_idx
        else:
            # Normal advancement with overlap
            i = end_idx - overlap_pages
    
    return chunks


def _find_sentence_boundary(text: str, start: int, preferred_end: int) -> int:
    """
    Find an intelligent sentence boundary near the preferred end position.
    """
    search_start = max(preferred_end - 200, start)
    
    # Look for sentence endings
    for i in range(preferred_end - 1, search_start - 1, -1):
        if text[i] in '.!?\n':
            # Validate it's a real sentence boundary
            if i + 1 < len(text) and (text[i + 1].isspace() or text[i + 1].isupper()):
                return i + 1
    
    # No good boundary found, use preferred end
    return preferred_end

def _ensure_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure chunk has all required metadata fields for orchestrator compatibility.
    
    Args:
        chunk: Chunk dictionary to validate
        
    Returns:
        Chunk with guaranteed metadata fields
    """
    required_fields = {
        "content": chunk.get("content", ""),
        "text": chunk.get("text", chunk.get("content", "")),  # Orchestrator compatibility
        "chunk_id": chunk.get("chunk_id", "unknown_chunk"),
        "start_page": chunk.get("start_page", 1),
        "end_page": chunk.get("end_page", 1),
        "pages_covered": chunk.get("pages_covered", [1]),
        "line_references": chunk.get("line_references", [])
    }
    
    # Update chunk with all required fields
    chunk.update(required_fields)
    return chunk


def _validate_chunk_sizes(chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Validate chunk sizes and only split if absolutely necessary.
    Uses configurable thresholds and can be disabled entirely.
    """
    # If splitting is disabled, return chunks as-is
    if getattr(settings, 'DISABLE_CHUNK_SPLITTING', False):
        logger.info("Chunk splitting is disabled, returning chunks as-is")
        return [_ensure_chunk_metadata(chunk) for chunk in chunks]
    
    validated_chunks = []
    
    for chunk in chunks:
        # Ensure chunk has all required metadata first
        chunk = _ensure_chunk_metadata(chunk)
        
        content = chunk.get("content", "")
        content_size = len(content.encode('utf-8'))
        
        # Use configurable threshold multiplier
        threshold_multiplier = getattr(settings, 'CHUNK_SPLITTING_THRESHOLD_MULTIPLIER', 1.5)
        size_threshold = int(settings.MAX_CHUNK_SIZE_BYTES * threshold_multiplier)
        
        if content_size <= size_threshold:
            validated_chunks.append(chunk)
        else:
            # Only split extremely oversized chunks
            logger.warning(f"Chunk size {content_size} bytes exceeds threshold ({size_threshold}), splitting...")
            
            # Extract metadata to preserve
            source_metadata = {
                "chunk_id": chunk.get("chunk_id", "unknown_chunk"),
                "start_page": chunk.get("start_page", 1),
                "end_page": chunk.get("end_page", 1), 
                "pages_covered": chunk.get("pages_covered", [1]),
                "line_references": chunk.get("line_references", [])
            }
            
            # Split with larger target size to minimize sub-chunks
            target_size = max(settings.MAX_CHUNK_SIZE_BYTES, len(content) // 2)
            
            sub_chunks = _create_semantic_chunks(
                content, 
                max_chars=target_size,
                overlap=500,  # Reduced overlap
                source_metadata=source_metadata
            )
            validated_chunks.extend(sub_chunks)
    
    return validated_chunks