"""
Intelligent Chunking Strategy for Policy Documents.

This module provides sophisticated chunking that leverages rich metadata from:
1. PDFProcessor - headings, TOC, section boundaries, page layout
2. DocumentAnalyzer - page content types, policy boundaries, content zones, 
   semantic continuity, policy flow graph

Key Features:
- Page filtering (removes TOC, bibliography, references, administrative pages)
- Policy-boundary-aware chunking (no policy mixing across chunks)
- Content zone utilization (respects semantic sections)
- Semantic continuity preservation (handles multi-page policies)
- Smart context preservation (includes definitions, prerequisites)
- Duplicate policy detection and merging
- Context completeness validation

The strategy ensures that:
1. Each chunk contains complete policy information (no mid-policy splits)
2. Non-policy content is filtered out before extraction
3. Related policies maintain their context and dependencies
4. Duplicate policies across documents are detected and merged
5. All necessary definitions and prerequisites are included
"""

import re
import tiktoken
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from langchain_openai import ChatOpenAI

from app.utils.logger import get_logger
from app.models.schemas import (
    PDFPage,
    PDFMetadata,
    PageContentType,
    PageAnalysis,
    PolicyBoundary,
    ContentZone,
    PolicyFlowNode,
    DocumentMetadata,
)
from config.settings import settings

logger = get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class PolicyChunk:
    """Represents a semantically coherent chunk focused on policy content."""
    
    def __init__(
        self,
        chunk_id: int,
        text: str,
        start_page: int,
        end_page: int,
        policy_ids: List[str],
        content_zones: List[str],
        token_count: int,
        has_definitions: bool,
        has_complete_context: bool,
        continuity_preserved: bool,
        metadata: Dict[str, Any],
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.start_page = start_page
        self.end_page = end_page
        self.policy_ids = policy_ids  # IDs of policies contained in this chunk
        self.content_zones = content_zones  # Zone identifiers
        self.token_count = token_count
        self.has_definitions = has_definitions
        self.has_complete_context = has_complete_context
        self.continuity_preserved = continuity_preserved
        self.metadata = metadata


class DuplicatePolicyCandidate:
    """Represents a potential duplicate policy detected across chunks."""
    
    def __init__(
        self,
        policy_id_1: str,
        policy_id_2: str,
        similarity_score: float,
        chunk_id_1: int,
        chunk_id_2: int,
        merge_recommendation: str,
    ):
        self.policy_id_1 = policy_id_1
        self.policy_id_2 = policy_id_2
        self.similarity_score = similarity_score
        self.chunk_id_1 = chunk_id_1
        self.chunk_id_2 = chunk_id_2
        self.merge_recommendation = merge_recommendation


class ChunkingResult:
    """Complete result of the chunking process."""
    
    def __init__(
        self,
        chunks: List[PolicyChunk],
        filtered_pages: List[int],
        duplicate_candidates: List[DuplicatePolicyCandidate],
        context_validation: Dict[str, Any],
        statistics: Dict[str, Any],
    ):
        self.chunks = chunks
        self.filtered_pages = filtered_pages
        self.duplicate_candidates = duplicate_candidates
        self.context_validation = context_validation
        self.statistics = statistics


# =============================================================================
# Semantic Chunking Strategy
# =============================================================================

class ChunkingStrategy:
    """
    Intelligent chunking strategy that uses rich document metadata to create
    semantically coherent chunks while preventing policy mixing and context loss.
    """
    
    def __init__(
        self,
        target_chunk_size: int = None,
        max_chunk_size: int = None,
        overlap: int = None,
        llm: Optional[ChatOpenAI] = None,
    ):
        """
        Initialize semantic chunking strategy.
        
        Args:
            target_chunk_size: Target chunk size in tokens
            max_chunk_size: Maximum chunk size in tokens
            overlap: Overlap size in tokens
            llm: Optional pre-configured LLM client for ambiguous cases
        """
        self.target_chunk_size = target_chunk_size or settings.target_chunk_tokens
        self.max_chunk_size = max_chunk_size or settings.max_chunk_tokens
        self.overlap = overlap or settings.chunk_overlap
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.llm = llm
        
        logger.info(
            f"Semantic chunking init: target={self.target_chunk_size}, "
            f"max={self.max_chunk_size}, overlap={self.overlap}"
        )
    
    # =========================================================================
    # Main Chunking Method
    # =========================================================================
    
    async def chunk_document(
        self,
        pages: List[Dict[str, Any]],  # PDFPage as dicts
        pdf_metadata: Dict[str, Any],  # PDFMetadata as dict
        doc_metadata: Dict[str, Any],  # DocumentMetadata as dict
    ) -> ChunkingResult:
        """
        Perform intelligent chunking using all available metadata.
        
        This is the main entry point that orchestrates the entire chunking process:
        1. Filter out non-policy pages
        2. Create policy-boundary-aware chunks
        3. Validate context completeness
        4. Detect duplicate policies
        5. Generate aggregation recommendations
        
        Args:
            pages: List of PDFPage objects (as dicts)
            pdf_metadata: PDFMetadata (as dict)
            doc_metadata: DocumentMetadata (as dict)
        
        Returns:
            ChunkingResult with chunks, filtered pages, and metadata
        """
        logger.info("=" * 80)
        logger.info("STARTING INTELLIGENT CHUNKING")
        logger.info("=" * 80)
        
        # Step 1: Filter pages
        logger.info("\n[STEP 1] Filtering non-policy pages...")
        filtered_pages, policy_pages = self._filter_non_policy_pages(
            pages, doc_metadata
        )
        logger.info(
            f"Filtered {len(filtered_pages)} pages, "
            f"processing {len(policy_pages)} policy pages"
        )
        
        # Step 2: Extract document structure and content zones
        logger.info("\n[STEP 2] Extracting document structure and content zones...")
        document_structure = doc_metadata.get("document_structure", {})
        major_sections = document_structure.get("major_sections", [])
        
        # Fallback to old policy boundaries if document structure not available
        if not major_sections:
            logger.warning("Document structure not available, falling back to policy boundaries")
            policy_boundaries = self._extract_policy_boundaries(doc_metadata)
        else:
            logger.info(f"Using document structure with {len(major_sections)} major sections")
            policy_boundaries = None  # Will use major_sections instead
        
        content_zones = self._extract_content_zones(doc_metadata)
        logger.info(
            f"Found {len(major_sections) if major_sections else len(policy_boundaries or [])} sections, "
            f"{len(content_zones)} content zones"
        )
        
        # Step 3: Create structure-aware chunks
        logger.info("\n[STEP 3] Creating structure-aware chunks...")
        chunks = await self._create_structure_aware_chunks(
            policy_pages,
            major_sections,
            policy_boundaries,
            content_zones,
            pdf_metadata,
            doc_metadata,
        )
        logger.info(f"Created {len(chunks)} structure-aware chunks")
        
        # Step 4: Validate context completeness
        logger.info("\n[STEP 4] Validating context completeness...")
        context_validation = self._validate_context_completeness(
            chunks, doc_metadata
        )
        logger.info(
            f"Context validation: {context_validation['complete_chunks']}/{len(chunks)} "
            f"chunks have complete context"
        )
        
        # Step 5: Detect duplicate policies
        logger.info("\n[STEP 5] Detecting duplicate policies...")
        duplicate_candidates = await self._detect_duplicate_policies(chunks)
        logger.info(f"Found {len(duplicate_candidates)} potential duplicate policies")
        
        # Step 6: Generate statistics
        statistics = self._generate_statistics(
            chunks, filtered_pages, policy_pages, duplicate_candidates
        )
        
        logger.info("=" * 80)
        logger.info("CHUNKING COMPLETE")
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(f"Filtered pages: {len(filtered_pages)}")
        logger.info(f"Policy pages processed: {len(policy_pages)}")
        logger.info(f"Avg tokens/chunk: {statistics['avg_tokens_per_chunk']:.0f}")
        logger.info("=" * 80)
        
        return ChunkingResult(
            chunks=chunks,
            filtered_pages=filtered_pages,
            duplicate_candidates=duplicate_candidates,
            context_validation=context_validation,
            statistics=statistics,
        )
    
    # =========================================================================
    # Step 1: Page Filtering
    # =========================================================================
    
    def _filter_non_policy_pages(
        self,
        pages: List[Dict[str, Any]],
        doc_metadata: Dict[str, Any],
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        Filter out pages that don't contain policy content.
        
        Uses:
        - PageContentType classifications
        - pages_to_filter from metadata
        - Content quality scores
        
        Filters out:
        - TOC pages
        - Bibliography/references
        - Administrative content
        - Index pages
        - Pages with low policy content ratio
        
        Args:
            pages: List of PDFPage dicts
            doc_metadata: DocumentMetadata dict
        
        Returns:
            Tuple of (filtered_page_numbers, policy_pages)
        """
        logger.info("Filtering non-policy pages using metadata...")
        
        # Get explicit filter guidance
        pages_to_filter_set = set(
            doc_metadata.get("pages_to_filter", [])
        )
        
        # Get page analyses
        page_analyses = {
            pa["page_number"]: pa
            for pa in doc_metadata.get("page_analyses", [])
        }
        
        # Filter criteria
        non_policy_types = {
            PageContentType.TABLE_OF_CONTENTS.value,
            PageContentType.BIBLIOGRAPHY.value,
            PageContentType.REFERENCES.value,
            PageContentType.INDEX.value,
            PageContentType.ADMINISTRATIVE.value,
        }
        
        filtered_pages = []
        policy_pages = []
        
        for page_dict in pages:
            page_num = page_dict["page_number"]
            
            # Check if explicitly marked for filtering
            if page_num in pages_to_filter_set:
                filtered_pages.append(page_num)
                logger.debug(f"Page {page_num}: Filtered (explicit guidance)")
                continue
            
            # Check page analysis
            page_analysis = page_analyses.get(page_num)
            if page_analysis:
                primary_type = page_analysis.get("primary_content_type")
                
                # Filter non-policy content types
                if primary_type in non_policy_types:
                    filtered_pages.append(page_num)
                    logger.debug(
                        f"Page {page_num}: Filtered (type={primary_type})"
                    )
                    continue
                
                # Check quality - filter low-quality pages
                quality_score = page_analysis.get("quality_score", 1.0)
                if quality_score < 0.3:
                    filtered_pages.append(page_num)
                    logger.debug(
                        f"Page {page_num}: Filtered (low quality={quality_score:.2f})"
                    )
                    continue
            
            # Keep as policy page
            policy_pages.append(page_dict)
            logger.debug(f"Page {page_num}: Kept for processing")
        
        logger.info(
            f"Page filtering complete: {len(filtered_pages)} filtered, "
            f"{len(policy_pages)} kept"
        )
        
        return filtered_pages, policy_pages
    
    # =========================================================================
    # Step 2: Policy Boundary and Zone Extraction
    # =========================================================================
    
    def _extract_policy_boundaries(
        self, doc_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract policy boundaries from metadata and convert to ranges.
        
        Converts per-page PolicyBoundary objects into ranged boundaries with:
        - policy_id
        - start_page, end_page
        - heading_text
        - is_multi_page
        
        Args:
            doc_metadata: DocumentMetadata dict

        Returns:
            List of ranged policy boundary dicts
        """
        boundaries = doc_metadata.get("policy_boundaries", [])        # Convert Pydantic objects to dicts if needed
        boundary_dicts = []
        for boundary in boundaries:
            if hasattr(boundary, 'dict'):
                boundary_dicts.append(boundary.dict())
            elif isinstance(boundary, dict):
                boundary_dicts.append(boundary)
            else:
                continue
        
        # Group per-page boundaries into ranges
        ranged_boundaries = []
        current_range = None
        
        for boundary in sorted(boundary_dicts, key=lambda b: b.get('page_number', 0)):
            page_num = boundary.get('page_number')
            policy_ids = boundary.get('policy_ids', [])
            
            # Skip unassigned boundaries
            if not policy_ids or policy_ids == ['unassigned']:
                continue
            
            policy_id = policy_ids[0]  # Use first policy ID
            
            # Start new range if policy starts here
            if boundary.get('policy_starts_here', False):
                if current_range:
                    ranged_boundaries.append(current_range)
                
                current_range = {
                    'policy_id': policy_id,
                    'start_page': page_num,
                    'end_page': page_num,
                    'heading_text': boundary.get('heading_at_start', ''),
                    'is_multi_page': False
                }
            
            # Extend current range if policy continues
            elif current_range and boundary.get('policy_continues_from_previous', False):
                current_range['end_page'] = page_num
                current_range['is_multi_page'] = True
            
            # Close range if policy ends here
            if boundary.get('policy_ends_here', False) and current_range:
                current_range['end_page'] = page_num
                ranged_boundaries.append(current_range)
                current_range = None
        
        # Add final range if open
        if current_range:
            ranged_boundaries.append(current_range)
        
        logger.info(f"Extracted {len(ranged_boundaries)} policy boundaries")
        for boundary in ranged_boundaries:
            logger.debug(
                f"  Policy '{boundary.get('policy_id', 'unknown')}': "
                f"pages {boundary.get('start_page')}-{boundary.get('end_page')}, "
                f"multi-page={boundary.get('is_multi_page', False)}"
            )
        
        return ranged_boundaries
    
    def _extract_content_zones(
        self, doc_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract content zones from metadata.
        
        Content zones represent semantically coherent sections that should
        stay together during chunking.
        
        Args:
            doc_metadata: DocumentMetadata dict

        Returns:
            List of content zone dicts
        """
        zones = doc_metadata.get("content_zones", [])
        logger.info(f"Extracted {len(zones)} content zones")
        for zone in zones:
            logger.debug(
                f"  Zone '{zone.get('zone_id', 'unknown')}': "
                f"pages {zone.get('start_page')}-{zone.get('end_page')}, "
                f"type={zone.get('zone_type')}"
            )
        
        return zones
    
    # =========================================================================
    # Step 3: Structure-Aware Chunk Creation (Phase 3 Improvement)
    # =========================================================================
    
    async def _create_structure_aware_chunks(
        self,
        policy_pages: List[Dict],
        major_sections: List[Dict],
        policy_boundaries: Optional[List[Dict]],
        content_zones: List[Dict],
        pdf_metadata: PDFMetadata,
        doc_metadata: Dict,
    ) -> List[PolicyChunk]:
        """
        Create chunks using document structure analysis (Phase 3 improvement).
        
        This method groups major sections with their subsections to maintain
        proper hierarchy and avoid fragmenting policies across chunks.
        
        Algorithm:
        1. Use major_sections from document structure analysis
        2. Group each major section with ALL its subsections in one chunk
        3. Target chunk size: 2000-4000 tokens (larger than old 1000)
        4. Only split at true major section boundaries
        5. Maintain parent-child relationships within chunks
        
        Args:
            policy_pages: Filtered pages containing policy content
            major_sections: Major policy sections from document structure
            policy_boundaries: Old policy boundaries (fallback)
            content_zones: Semantic content zones
            pdf_metadata: PDF metadata
            doc_metadata: Document analysis metadata
        
        Returns:
            List of PolicyChunk objects
        """
        chunks = []
        chunk_id = 0
        
        # Use document structure if available
        if major_sections:
            logger.info(
                f"Using structure-aware chunking with {len(major_sections)} major sections"
            )
            logger.info(f"Target chunk size: 2000-4000 tokens (increased from 1000)")
            
            # Sort sections by start page
            sorted_sections = sorted(major_sections, key=lambda s: s['start_page'])
            
            for section in sorted_sections:
                # Get pages for this major section AND all its subsections
                section_start = section['start_page']
                section_end = section['end_page']
                
                section_pages = [
                    p for p in policy_pages
                    if section_start <= p['page_number'] <= section_end
                ]
                
                if not section_pages:
                    logger.warning(
                        f"No pages found for section '{section['title']}' "
                        f"(pages {section_start}-{section_end})"
                    )
                    continue
                
                # Build chunk text including all subsections
                chunk_text = self._build_section_chunk_text(
                    section_pages, section, pdf_metadata
                )
                
                token_count = self._count_tokens(chunk_text)
                
                # Check if chunk is within acceptable range (2000-8000 tokens)
                if token_count > self.max_chunk_size:
                    logger.warning(
                        f"Section '{section['title']}' exceeds max chunk size "
                        f"({token_count} > {self.max_chunk_size}), splitting by subsections..."
                    )
                    # Split by subsections if section too large
                    sub_chunks = self._split_by_subsections(
                        section_pages,
                        section,
                        content_zones,
                        pdf_metadata,
                        chunk_id,
                    )
                    chunks.extend(sub_chunks)
                    chunk_id += len(sub_chunks)
                else:
                    # Create single chunk for entire major section
                    chunk = self._create_chunk_from_section(
                        chunk_id,
                        section_pages,
                        section,
                        content_zones,
                        pdf_metadata,
                        doc_metadata,
                    )
                    chunks.append(chunk)
                    logger.info(
                        f"Created chunk {chunk_id} for section '{section['title']}': "
                        f"pages {chunk.start_page}-{chunk.end_page}, {token_count} tokens, "
                        f"{len(section.get('subsections', []))} subsections"
                    )
                    chunk_id += 1
            
            logger.info(
                f"Created {len(chunks)} structure-aware chunks from {len(major_sections)} major sections"
            )
        
        else:
            # Fallback to old policy-boundary method
            logger.warning("Document structure not available, falling back to old chunking method")
            chunks = await self._create_policy_aware_chunks(
                policy_pages,
                policy_boundaries,
                content_zones,
                pdf_metadata,
                doc_metadata,
            )
        
        return chunks
    
    def _build_section_chunk_text(
        self,
        section_pages: List[Dict],
        section: Dict,
        pdf_metadata: PDFMetadata,
    ) -> str:
        """
        Build chunk text for a major section including all subsections.
        
        Args:
            section_pages: Pages for this section
            section: Section metadata
            pdf_metadata: PDF metadata
        
        Returns:
            Formatted chunk text
        """
        lines = []
        
        # Add section header
        lines.append(f"=== {section['title']} ===")
        lines.append(f"Section ID: {section['section_id']}")
        lines.append(f"Pages: {section['start_page']}-{section['end_page']}")
        
        # Add subsection info if available
        subsections = section.get('subsections', [])
        if subsections:
            lines.append(f"Subsections: {len(subsections)}")
            for subsec in subsections:
                lines.append(f"  - {subsec['title']} (pages {subsec['start_page']}-{subsec['end_page']})")
        
        lines.append("")
        
        # Add page content
        for page in section_pages:
            lines.append(f"[Page {page['page_number']}]")
            lines.append(page['text'])
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_chunk_from_section(
        self,
        chunk_id: int,
        section_pages: List[Dict],
        section: Dict,
        content_zones: List[Dict],
        pdf_metadata: PDFMetadata,
        doc_metadata: Dict,
    ) -> PolicyChunk:
        """
        Create a PolicyChunk from a major section.
        
        Args:
            chunk_id: Chunk identifier
            section_pages: Pages for this section
            section: Section metadata
            content_zones: Content zones
            pdf_metadata: PDF metadata
            doc_metadata: Document metadata
        
        Returns:
            PolicyChunk object
        """
        chunk_text = self._build_section_chunk_text(section_pages, section, pdf_metadata)
        token_count = self._count_tokens(chunk_text)
        
        # Determine zone types present in this section
        start_page = section['start_page']
        end_page = section['end_page']
        zone_types = list(set(
            zone.get('zone_type', 'policy')
            for zone in content_zones
            if zone.get('start_page', 0) <= end_page and zone.get('end_page', 999) >= start_page
        ))
        
        # Build metadata from subsections
        metadata = {
            'section_id': section['section_id'],
            'section_title': section['title'],
            'section_level': section['heading_level'],
            'has_subsections': len(section.get('subsections', [])) > 0,
            'subsection_count': len(section.get('subsections', [])),
            'numbering': section.get('numbering'),
        }
        
        # Create PolicyChunk with correct signature
        return PolicyChunk(
            chunk_id=chunk_id,
            text=chunk_text,
            start_page=start_page,
            end_page=end_page,
            policy_ids=[section['section_id']],  # Single policy/section per chunk
            content_zones=zone_types,
            token_count=token_count,
            has_definitions='definitions' in zone_types,
            has_complete_context=True,  # Structure-aware chunks have complete context
            continuity_preserved=True,  # Parent-child in same chunk
            metadata=metadata,
        )
    
    def _split_by_subsections(
        self,
        section_pages: List[Dict],
        section: Dict,
        content_zones: List[Dict],
        pdf_metadata: PDFMetadata,
        start_chunk_id: int,
    ) -> List[PolicyChunk]:
        """
        Split a large section by subsections when it exceeds max chunk size.
        
        Args:
            section_pages: Pages for this section
            section: Section metadata
            content_zones: Content zones
            pdf_metadata: PDF metadata
            start_chunk_id: Starting chunk ID
        
        Returns:
            List of PolicyChunk objects
        """
        chunks = []
        chunk_id = start_chunk_id
        
        subsections = section.get('subsections', [])
        
        if not subsections:
            # No subsections, split by page ranges
            logger.warning(f"No subsections found for '{section['title']}', splitting by pages")
            page_ranges = self._split_pages_into_ranges(section_pages, target_tokens=3000)
            
            for page_range in page_ranges:
                chunk = self._create_chunk_from_page_range(
                    chunk_id,
                    page_range,
                    section,
                    content_zones,
                    pdf_metadata,
                )
                chunks.append(chunk)
                chunk_id += 1
        else:
            # Split by subsections
            logger.info(f"Splitting '{section['title']}' into {len(subsections)} subsection chunks")
            
            for subsec in subsections:
                subsec_pages = [
                    p for p in section_pages
                    if subsec['start_page'] <= p['page_number'] <= subsec['end_page']
                ]
                
                if not subsec_pages:
                    continue
                
                # Create pseudo-section for subsection
                subsec_as_section = {
                    'section_id': subsec['subsection_id'],
                    'title': subsec['title'],
                    'start_page': subsec['start_page'],
                    'end_page': subsec['end_page'],
                    'heading_level': subsec['heading_level'],
                    'numbering': subsec.get('numbering'),
                    'parent_section': subsec.get('parent_section'),
                    'subsections': [],  # Subsections don't have sub-subsections
                }
                
                chunk = self._create_chunk_from_section(
                    chunk_id,
                    subsec_pages,
                    subsec_as_section,
                    content_zones,
                    pdf_metadata,
                    {},
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    def _split_pages_into_ranges(
        self,
        pages: List[Dict],
        target_tokens: int = 3000,
    ) -> List[List[Dict]]:
        """
        Split pages into ranges based on target token count.
        
        Args:
            pages: List of page dicts
            target_tokens: Target tokens per range
        
        Returns:
            List of page ranges
        """
        ranges = []
        current_range = []
        current_tokens = 0
        
        for page in pages:
            page_tokens = self._count_tokens(page['text'])
            
            if current_tokens + page_tokens > target_tokens and current_range:
                # Start new range
                ranges.append(current_range)
                current_range = [page]
                current_tokens = page_tokens
            else:
                current_range.append(page)
                current_tokens += page_tokens
        
        # Add final range
        if current_range:
            ranges.append(current_range)
        
        return ranges
    
    def _create_chunk_from_page_range(
        self,
        chunk_id: int,
        page_range: List[Dict],
        section: Dict,
        content_zones: List[Dict],
        pdf_metadata: PDFMetadata,
    ) -> PolicyChunk:
        """
        Create a chunk from a page range.
        
        Args:
            chunk_id: Chunk ID
            page_range: List of pages
            section: Parent section
            content_zones: Content zones
            pdf_metadata: PDF metadata
        
        Returns:
            PolicyChunk object
        """
        start_page = page_range[0]['page_number']
        end_page = page_range[-1]['page_number']
        
        # Build text
        lines = [f"=== {section['title']} (Part {chunk_id + 1}) ==="]
        lines.append(f"Pages: {start_page}-{end_page}")
        lines.append("")
        
        for page in page_range:
            lines.append(f"[Page {page['page_number']}]")
            lines.append(page['text'])
            lines.append("")
        
        chunk_text = "\n".join(lines)
        token_count = self._count_tokens(chunk_text)
        
        # Determine zone types
        zone_types = list(set(
            zone.get('zone_type', 'policy')
            for zone in content_zones
            if zone.get('start_page', 0) <= end_page and zone.get('end_page', 999) >= start_page
        ))
        
        # Build metadata
        metadata = {
            'section_id': section['section_id'],
            'section_title': section['title'],
            'is_partial': True,
            'part_number': chunk_id + 1,
        }
        
        # Create PolicyChunk with correct signature
        return PolicyChunk(
            chunk_id=chunk_id,
            text=chunk_text,
            start_page=start_page,
            end_page=end_page,
            policy_ids=[section['section_id']],
            content_zones=zone_types,
            token_count=token_count,
            has_definitions='definitions' in zone_types,
            has_complete_context=False,  # Partial chunk
            continuity_preserved=True,
            metadata=metadata,
        )
    
    # =========================================================================
    # Step 3 (Old): Policy-Aware Chunk Creation (Fallback)
    # =========================================================================
    
    async def _create_policy_aware_chunks(
        self,
        policy_pages: List[Dict[str, Any]],
        policy_boundaries: List[Dict[str, Any]],
        content_zones: List[Dict[str, Any]],
        pdf_metadata: Dict[str, Any],
        doc_metadata: Dict[str, Any],
    ) -> List[PolicyChunk]:
        """
        Create chunks that respect policy boundaries and content zones.
        
        Algorithm:
        1. Group pages by policy boundaries
        2. Respect content zone boundaries
        3. Use recommended_chunk_boundaries from metadata
        4. Preserve semantic continuity (handle multi-page policies)
        5. Include necessary context (definitions, prerequisites)
        6. Enforce token limits while maintaining semantic coherence
        
        Args:
            policy_pages: Filtered pages containing policy content
            policy_boundaries: Detected policy boundaries
            content_zones: Semantic content zones
            pdf_metadata: PDF metadata
            doc_metadata: Document analysis metadata
        
        Returns:
            List of PolicyChunk objects
        """
        logger.info("Creating policy-aware chunks...")
        
        # Get recommended chunk boundaries from metadata
        recommended_boundaries = doc_metadata.get(
            "recommended_chunk_boundaries", []
        )
        logger.info(
            f"Using {len(recommended_boundaries)} recommended chunk boundaries"
        )
        
        chunks = []
        chunk_id = 0
        
        # Strategy: Group by policy boundaries first
        if policy_boundaries:
            logger.info(
                "Using policy-boundary-based chunking strategy "
                f"({len(policy_boundaries)} boundaries)"
            )
            
            # Sort boundaries by start page
            sorted_boundaries = sorted(
                policy_boundaries, key=lambda b: b["start_page"]
            )
            
            for boundary in sorted_boundaries:
                # Get pages for this policy
                policy_pages_subset = [
                    p for p in policy_pages
                    if boundary["start_page"] <= p["page_number"] <= boundary["end_page"]
                ]
                
                if not policy_pages_subset:
                    logger.warning(
                        f"No pages found for policy {boundary.get('policy_id')} "
                        f"(pages {boundary['start_page']}-{boundary['end_page']})"
                    )
                    continue
                
                # Build chunk text
                chunk_text = self._build_chunk_text(
                    policy_pages_subset, boundary, pdf_metadata
                )
                
                token_count = self._count_tokens(chunk_text)
                
                # Check if chunk exceeds max size
                if token_count > self.max_chunk_size:
                    logger.warning(
                        f"Policy {boundary.get('policy_id')} exceeds max chunk size "
                        f"({token_count} > {self.max_chunk_size}), splitting..."
                    )
                    # Split this policy into multiple chunks
                    sub_chunks = self._split_large_policy(
                        policy_pages_subset,
                        boundary,
                        content_zones,
                        pdf_metadata,
                        chunk_id,
                    )
                    chunks.extend(sub_chunks)
                    chunk_id += len(sub_chunks)
                else:
                    # Create single chunk for this policy
                    chunk = self._create_chunk_from_policy(
                        chunk_id,
                        policy_pages_subset,
                        boundary,
                        content_zones,
                        pdf_metadata,
                        doc_metadata,
                    )
                    chunks.append(chunk)
                    logger.info(
                        f"Created chunk {chunk_id} for policy {boundary.get('policy_id')}: "
                        f"pages {chunk.start_page}-{chunk.end_page}, {token_count} tokens"
                    )
                    chunk_id += 1
        
        else:
            # Fallback: Use content zones if no policy boundaries detected
            logger.info("No policy boundaries, using content-zone-based chunking")
            chunks = self._chunk_by_content_zones(
                policy_pages, content_zones, pdf_metadata, doc_metadata
            )
        
        logger.info(f"Created {len(chunks)} policy-aware chunks")
        return chunks
    
    def _build_chunk_text(
        self,
        pages: List[Dict[str, Any]],
        boundary: Dict[str, Any],
        pdf_metadata: Dict[str, Any],
    ) -> str:
        """
        Build chunk text from pages, including context preservation.
        
        Args:
            pages: Pages for this chunk
            boundary: Policy boundary info
            pdf_metadata: PDF metadata
        
        Returns:
            Formatted chunk text
        """
        lines = []
        
        # Add policy header if available
        if boundary.get("heading_text"):
            lines.append(f"=== {boundary['heading_text']} ===\n")
        
        # Add page content
        for page_dict in pages:
            page_num = page_dict["page_number"]
            text = page_dict.get("text", "")
            
            lines.append(f"--- Page {page_num} ---")
            lines.append(text)
            
            # Add table content if present
            tables = page_dict.get("tables", [])
            if tables:
                for idx, table_meta in enumerate(tables):
                    lines.append(f"\n[Table {idx + 1} on page {page_num}]")
                    # Table content would be extracted from table_meta
        
        return "\n\n".join(lines)
    
    def _create_chunk_from_policy(
        self,
        chunk_id: int,
        pages: List[Dict[str, Any]],
        boundary: Dict[str, Any],
        content_zones: List[Dict[str, Any]],
        pdf_metadata: Dict[str, Any],
        doc_metadata: Dict[str, Any],
    ) -> PolicyChunk:
        """
        Create a PolicyChunk from a single policy boundary.
        
        Args:
            chunk_id: Chunk identifier
            pages: Pages for this chunk
            boundary: Policy boundary
            content_zones: All content zones
            pdf_metadata: PDF metadata
            doc_metadata: Document metadata
        
        Returns:
            PolicyChunk instance
        """
        chunk_text = self._build_chunk_text(pages, boundary, pdf_metadata)
        token_count = self._count_tokens(chunk_text)
        
        # Find content zones that overlap with this chunk
        chunk_zones = [
            zone["zone_id"]
            for zone in content_zones
            if self._zones_overlap(boundary, zone)
        ]
        
        # Check for definitions in chunk
        has_definitions = self._contains_definitions(chunk_text)
        
        # Check continuity preservation
        continuity_preserved = not boundary.get("is_multi_page", False) or \
                               self._check_continuity_preserved(pages, boundary)
        
        # Check context completeness
        has_complete_context = self._check_context_completeness(
            chunk_text, boundary, doc_metadata
        )
        
        return PolicyChunk(
            chunk_id=chunk_id,
            text=chunk_text,
            start_page=boundary["start_page"],
            end_page=boundary["end_page"],
            policy_ids=[boundary.get("policy_id", f"policy_{chunk_id}")],
            content_zones=chunk_zones,
            token_count=token_count,
            has_definitions=has_definitions,
            has_complete_context=has_complete_context,
            continuity_preserved=continuity_preserved,
            metadata={
                "heading": boundary.get("heading_text"),
                "is_multi_page": boundary.get("is_multi_page", False),
                "boundary_type": "policy",
            },
        )
    
    def _split_large_policy(
        self,
        pages: List[Dict[str, Any]],
        boundary: Dict[str, Any],
        content_zones: List[Dict[str, Any]],
        pdf_metadata: Dict[str, Any],
        start_chunk_id: int,
    ) -> List[PolicyChunk]:
        """
        Split a large policy that exceeds max chunk size into multiple chunks.
        
        Strategy:
        - Try to split at content zone boundaries
        - Preserve semantic coherence
        - Add overlap for context
        
        Args:
            pages: Pages for this policy
            boundary: Policy boundary
            content_zones: Content zones
            pdf_metadata: PDF metadata
            start_chunk_id: Starting chunk ID
        
        Returns:
            List of PolicyChunk objects
        """
        logger.info(
            f"Splitting large policy {boundary.get('policy_id')} into sub-chunks"
        )
        
        chunks = []
        chunk_id = start_chunk_id
        
        # Find content zones within this policy
        policy_zones = [
            zone for zone in content_zones
            if self._zones_overlap(boundary, zone)
        ]
        
        if policy_zones:
            # Split by zones
            logger.debug(f"Splitting by {len(policy_zones)} content zones")
            for zone in policy_zones:
                zone_pages = [
                    p for p in pages
                    if zone["start_page"] <= p["page_number"] <= zone["end_page"]
                ]
                
                if not zone_pages:
                    continue
                
                chunk_text = self._build_chunk_text(zone_pages, boundary, pdf_metadata)
                token_count = self._count_tokens(chunk_text)
                
                # If zone still too large, split by pages
                if token_count > self.max_chunk_size:
                    page_chunks = self._split_by_pages(
                        zone_pages, boundary, pdf_metadata, chunk_id
                    )
                    chunks.extend(page_chunks)
                    chunk_id += len(page_chunks)
                else:
                    chunk = PolicyChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        start_page=zone["start_page"],
                        end_page=zone["end_page"],
                        policy_ids=[boundary.get("policy_id", f"policy_{chunk_id}")],
                        content_zones=[zone["zone_id"]],
                        token_count=token_count,
                        has_definitions=self._contains_definitions(chunk_text),
                        has_complete_context=False,  # Partial policy
                        continuity_preserved=True,
                        metadata={
                            "heading": boundary.get("heading_text"),
                            "is_multi_page": True,
                            "boundary_type": "zone_split",
                            "zone_id": zone["zone_id"],
                        },
                    )
                    chunks.append(chunk)
                    chunk_id += 1
        else:
            # No zones, split by pages as last resort
            logger.debug("No content zones, splitting by pages")
            chunks = self._split_by_pages(pages, boundary, pdf_metadata, chunk_id)
        
        logger.info(f"Split into {len(chunks)} sub-chunks")
        return chunks
    
    def _split_by_pages(
        self,
        pages: List[Dict[str, Any]],
        boundary: Dict[str, Any],
        pdf_metadata: Dict[str, Any],
        start_chunk_id: int,
    ) -> List[PolicyChunk]:
        """
        Split pages into chunks (last resort for oversized content).
        
        Args:
            pages: Pages to split
            boundary: Policy boundary
            pdf_metadata: PDF metadata
            start_chunk_id: Starting chunk ID
        
        Returns:
            List of PolicyChunk objects
        """
        chunks = []
        chunk_id = start_chunk_id
        current_pages = []
        current_tokens = 0
        
        for page in pages:
            page_text = self._build_chunk_text([page], boundary, pdf_metadata)
            page_tokens = self._count_tokens(page_text)
            
            if current_tokens + page_tokens > self.target_chunk_size and current_pages:
                # Create chunk
                chunk_text = self._build_chunk_text(current_pages, boundary, pdf_metadata)
                chunk = PolicyChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_page=current_pages[0]["page_number"],
                    end_page=current_pages[-1]["page_number"],
                    policy_ids=[boundary.get("policy_id", f"policy_{chunk_id}")],
                    content_zones=[],
                    token_count=self._count_tokens(chunk_text),
                    has_definitions=self._contains_definitions(chunk_text),
                    has_complete_context=False,
                    continuity_preserved=False,
                    metadata={
                        "heading": boundary.get("heading_text"),
                        "is_multi_page": True,
                        "boundary_type": "page_split_forced",
                    },
                )
                chunks.append(chunk)
                chunk_id += 1
                current_pages = []
                current_tokens = 0
            
            current_pages.append(page)
            current_tokens += page_tokens
        
        # Final chunk
        if current_pages:
            chunk_text = self._build_chunk_text(current_pages, boundary, pdf_metadata)
            chunk = PolicyChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_page=current_pages[0]["page_number"],
                end_page=current_pages[-1]["page_number"],
                policy_ids=[boundary.get("policy_id", f"policy_{chunk_id}")],
                content_zones=[],
                token_count=self._count_tokens(chunk_text),
                has_definitions=self._contains_definitions(chunk_text),
                has_complete_context=False,
                continuity_preserved=False,
                metadata={
                    "heading": boundary.get("heading_text"),
                    "is_multi_page": True,
                    "boundary_type": "page_split_forced",
                },
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_content_zones(
        self,
        pages: List[Dict[str, Any]],
        content_zones: List[Dict[str, Any]],
        pdf_metadata: Dict[str, Any],
        doc_metadata: Dict[str, Any],
    ) -> List[PolicyChunk]:
        """
        Fallback chunking strategy using content zones when no policy boundaries exist.
        
        Args:
            pages: Policy pages
            content_zones: Content zones
            pdf_metadata: PDF metadata
            doc_metadata: Document metadata
        
        Returns:
            List of PolicyChunk objects
        """
        logger.info("Chunking by content zones (fallback strategy)")
        
        chunks = []
        chunk_id = 0
        
        for zone in content_zones:
            zone_pages = [
                p for p in pages
                if zone["start_page"] <= p["page_number"] <= zone["end_page"]
            ]
            
            if not zone_pages:
                continue
            
            # Create pseudo-boundary for zone
            zone_boundary = {
                "policy_id": zone.get("zone_id", f"zone_{chunk_id}"),
                "start_page": zone["start_page"],
                "end_page": zone["end_page"],
                "heading_text": zone.get("zone_type", "Content Zone"),
                "is_multi_page": zone["end_page"] > zone["start_page"],
            }
            
            chunk_text = self._build_chunk_text(zone_pages, zone_boundary, pdf_metadata)
            token_count = self._count_tokens(chunk_text)
            
            if token_count > self.max_chunk_size:
                # Split zone
                sub_chunks = self._split_by_pages(
                    zone_pages, zone_boundary, pdf_metadata, chunk_id
                )
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
            else:
                chunk = PolicyChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_page=zone["start_page"],
                    end_page=zone["end_page"],
                    policy_ids=[zone.get("zone_id", f"zone_{chunk_id}")],
                    content_zones=[zone.get("zone_id", f"zone_{chunk_id}")],
                    token_count=token_count,
                    has_definitions=self._contains_definitions(chunk_text),
                    has_complete_context=True,  # Assume zones are complete
                    continuity_preserved=True,
                    metadata={
                        "heading": zone.get("zone_type"),
                        "is_multi_page": zone["end_page"] > zone["start_page"],
                        "boundary_type": "content_zone",
                    },
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    # =========================================================================
    # Step 4: Context Completeness Validation
    # =========================================================================
    
    def _validate_context_completeness(
        self,
        chunks: List[PolicyChunk],
        doc_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate that chunks contain complete context (definitions, prerequisites).
        
        Checks:
        - Definitions are included or referenced
        - Prerequisites are present
        - Cross-references are resolved
        - Policy flow is maintained
        
        Args:
            chunks: List of PolicyChunk objects
            doc_metadata: Document metadata
        
        Returns:
            Validation results dictionary
        """
        logger.info("Validating context completeness for chunks...")
        
        complete_chunks = 0
        incomplete_chunks = []
        definition_coverage = 0
        
        for chunk in chunks:
            issues = []
            
            # Check for definitions
            if not chunk.has_definitions:
                issues.append("missing_definitions")
            else:
                definition_coverage += 1
            
            # Check for complete context flag
            if not chunk.has_complete_context:
                issues.append("incomplete_context")
            
            # Check continuity
            if not chunk.continuity_preserved:
                issues.append("continuity_broken")
            
            if issues:
                incomplete_chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "issues": issues,
                })
            else:
                complete_chunks += 1
        
        validation_result = {
            "total_chunks": len(chunks),
            "complete_chunks": complete_chunks,
            "incomplete_chunks": len(incomplete_chunks),
            "definition_coverage_pct": (definition_coverage / len(chunks) * 100)
            if chunks else 0,
            "incomplete_chunk_details": incomplete_chunks,
        }
        
        logger.info(
            f"Context validation: {complete_chunks}/{len(chunks)} chunks complete, "
            f"definition coverage: {validation_result['definition_coverage_pct']:.1f}%"
        )
        
        return validation_result
    
    # =========================================================================
    # Step 5: Duplicate Policy Detection
    # =========================================================================
    
    async def _detect_duplicate_policies(
        self, chunks: List[PolicyChunk]
    ) -> List[DuplicatePolicyCandidate]:
        """
        Detect potential duplicate policies across chunks using multi-signal approach.
        
        Uses multiple signals to avoid false positives:
        1. Position check (skip adjacent chunks - they're neighbors, not duplicates)
        2. Page distance check (skip same/adjacent pages)
        3. Title similarity (high precision check)
        4. Token-based content similarity (fallback)
        
        Only flags as duplicate if MULTIPLE signals agree.
        
        Args:
            chunks: List of PolicyChunk objects
        
        Returns:
            List of DuplicatePolicyCandidate objects
        """
        logger.info("Detecting duplicate policies with multi-signal approach...")
        
        duplicates = []
        skipped_adjacent = 0
        skipped_same_page = 0
        skipped_different_titles = 0
        
        # Compare each pair of chunks
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i + 1:], start=i + 1):
                
                # SIGNAL 1: Position Check (CRITICAL for false positive prevention)
                # Adjacent chunks share context naturally - they are NOT duplicates
                if abs(i - j) == 1:
                    logger.debug(
                        f"Skipping adjacent chunks {i} and {j} "
                        f"(position difference = 1, naturally share context)"
                    )
                    skipped_adjacent += 1
                    continue
                
                # SIGNAL 2: Page Distance Check
                # Chunks on same or adjacent pages are likely different sections
                chunk1_start = chunk1.metadata.get('page_range', [0])[0] if isinstance(
                    chunk1.metadata.get('page_range'), list
                ) else chunk1.start_page
                chunk2_start = chunk2.metadata.get('page_range', [0])[0] if isinstance(
                    chunk2.metadata.get('page_range'), list
                ) else chunk2.start_page
                
                page_distance = abs(chunk1_start - chunk2_start)
                if page_distance <= 1:
                    logger.debug(
                        f"Skipping same/adjacent page chunks {i} and {j} "
                        f"(page distance = {page_distance}, likely different sections)"
                    )
                    skipped_same_page += 1
                    continue
                
                # SIGNAL 3: Title Similarity (High Precision)
                title1 = chunk1.metadata.get('section_title', '') or chunk1.metadata.get('heading', '')
                title2 = chunk2.metadata.get('section_title', '') or chunk2.metadata.get('heading', '')
                
                if title1 and title2:
                    title_similarity = self._calculate_title_similarity(title1, title2)
                    if title_similarity < 0.7:
                        logger.debug(
                            f"Skipping chunks {i} and {j} with different titles "
                            f"('{title1[:30]}...' vs '{title2[:30]}...', similarity={title_similarity:.2f})"
                        )
                        skipped_different_titles += 1
                        continue
                else:
                    title_similarity = 0.0
                
                # SIGNAL 4: Token-Based Content Similarity (Fallback)
                token_similarity = self._calculate_chunk_similarity(chunk1, chunk2)
                
                # FINAL DECISION: Only flag if BOTH high title similarity AND high content similarity
                # This prevents false positives from adjacent sections
                is_duplicate = False
                
                if title_similarity > 0.8 and token_similarity > 0.85:
                    # Very similar title AND content → likely duplicate
                    is_duplicate = True
                    final_similarity = (title_similarity + token_similarity) / 2
                    reason = "high title + content similarity"
                elif title_similarity > 0.9 and token_similarity > 0.75:
                    # Almost identical title, high content overlap → likely duplicate
                    is_duplicate = True
                    final_similarity = (title_similarity * 0.6 + token_similarity * 0.4)
                    reason = "very high title similarity"
                elif token_similarity > 0.95:
                    # Extremely high content similarity (even if titles differ slightly)
                    is_duplicate = True
                    final_similarity = token_similarity
                    reason = "very high content similarity"
                
                if is_duplicate:
                    merge_recommendation = self._generate_merge_recommendation(
                        chunk1, chunk2, final_similarity
                    )
                    
                    duplicate = DuplicatePolicyCandidate(
                        policy_id_1=chunk1.policy_ids[0] if chunk1.policy_ids else f"chunk_{i}",
                        policy_id_2=chunk2.policy_ids[0] if chunk2.policy_ids else f"chunk_{j}",
                        similarity_score=final_similarity,
                        chunk_id_1=chunk1.chunk_id,
                        chunk_id_2=chunk2.chunk_id,
                        merge_recommendation=merge_recommendation,
                    )
                    duplicates.append(duplicate)
                    
                    # Mark the second chunk as a duplicate in metadata
                    chunk2.metadata['is_duplicate'] = True
                    chunk2.metadata['duplicate_of'] = chunk1.chunk_id
                    chunk2.metadata['duplicate_similarity'] = final_similarity
                    chunk2.metadata['duplicate_reason'] = reason
                    
                    logger.info(
                        f"Found TRUE duplicate: chunks {chunk1.chunk_id} & {chunk2.chunk_id}, "
                        f"title_sim={title_similarity:.2f}, token_sim={token_similarity:.2f}, "
                        f"final={final_similarity:.2f} ({reason}). Marking chunk {chunk2.chunk_id} as duplicate."
                    )
        
        logger.info(
            f"Duplicate detection complete: {len(duplicates)} TRUE duplicates found, "
            f"{skipped_adjacent} adjacent pairs skipped, "
            f"{skipped_same_page} same-page pairs skipped, "
            f"{skipped_different_titles} different-title pairs skipped"
        )
        return duplicates
    
    def _calculate_chunk_similarity(
        self, chunk1: PolicyChunk, chunk2: PolicyChunk
    ) -> float:
        """
        Calculate similarity between two chunks.
        
        Uses simple token-based similarity (can be improved with embeddings).
        
        Args:
            chunk1: First chunk
            chunk2: Second chunk
        
        Returns:
            Similarity score (0-1)
        """
        # Tokenize both chunks
        tokens1 = set(self._tokenize_for_similarity(chunk1.text))
        tokens2 = set(self._tokenize_for_similarity(chunk2.text))
        
        # Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two titles using token-based comparison.
        
        Titles are more precise indicators of content than full text,
        so we use a stricter comparison.
        
        Args:
            title1: First title
            title2: Second title
        
        Returns:
            Similarity score (0-1)
        """
        if not title1 or not title2:
            return 0.0
        
        # Normalize titles
        title1_lower = title1.lower().strip()
        title2_lower = title2.lower().strip()
        
        # Exact match
        if title1_lower == title2_lower:
            return 1.0
        
        # Tokenize titles
        tokens1 = set(self._tokenize_for_similarity(title1))
        tokens2 = set(self._tokenize_for_similarity(title2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity for titles
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _tokenize_for_similarity(self, text: str) -> List[str]:
        """
        Tokenize text for similarity comparison.
        
        Args:
            text: Text to tokenize
        
        Returns:
            List of tokens
        """
        # Simple word tokenization (can be improved)
        words = re.findall(r'\b\w+\b', text.lower())
        # Remove common stop words for better similarity
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return [w for w in words if w not in stop_words]
    
    def _generate_merge_recommendation(
        self, chunk1: PolicyChunk, chunk2: PolicyChunk, similarity: float
    ) -> str:
        """
        Generate recommendation for merging duplicate policies.
        
        Args:
            chunk1: First chunk
            chunk2: Second chunk
            similarity: Similarity score
        
        Returns:
            Merge recommendation string
        """
        if similarity > 0.9:
            return "MERGE: High similarity, likely exact duplicates"
        elif similarity > 0.8:
            return "REVIEW: Very similar, may be duplicates with minor variations"
        else:
            return "MONITOR: Moderate similarity, review for potential overlap"
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _zones_overlap(
        self, boundary: Dict[str, Any], zone: Dict[str, Any]
    ) -> bool:
        """Check if a policy boundary overlaps with a content zone."""
        return not (
            boundary["end_page"] < zone["start_page"]
            or boundary["start_page"] > zone["end_page"]
        )
    
    def _contains_definitions(self, text: str) -> bool:
        """
        Check if text contains definition keywords.
        
        Args:
            text: Text to check
        
        Returns:
            True if definitions found
        """
        definition_patterns = [
            r'\bdefinition',
            r'\bdefined as\b',
            r'\bmeans\b',
            r'\brefers to\b',
            r'\bshall mean\b',
        ]
        
        text_lower = text.lower()
        for pattern in definition_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _check_continuity_preserved(
        self, pages: List[Dict[str, Any]], boundary: Dict[str, Any]
    ) -> bool:
        """
        Check if continuity is preserved across pages in a multi-page policy.
        
        Args:
            pages: Pages in chunk
            boundary: Policy boundary
        
        Returns:
            True if continuity preserved
        """
        # For now, assume continuity preserved if all pages in boundary are included
        page_numbers = {p["page_number"] for p in pages}
        expected_pages = set(range(boundary["start_page"], boundary["end_page"] + 1))
        
        return expected_pages.issubset(page_numbers)
    
    def _check_context_completeness(
        self,
        chunk_text: str,
        boundary: Dict[str, Any],
        doc_metadata: Dict[str, Any],
    ) -> bool:
        """
        Check if chunk has complete context (definitions, prerequisites).
        
        Args:
            chunk_text: Chunk text
            boundary: Policy boundary
            doc_metadata: Document metadata
        
        Returns:
            True if context is complete
        """
        # Check for definitions
        has_defs = self._contains_definitions(chunk_text)
        
        # Check for references to other sections
        # If policy references other sections, context may be incomplete
        reference_patterns = [
            r'see section',
            r'refer to',
            r'as defined in',
            r'pursuant to',
        ]
        
        has_external_refs = any(
            re.search(pattern, chunk_text.lower())
            for pattern in reference_patterns
        )
        
        # Context is complete if it has definitions and no external references
        # (or if it's a self-contained policy)
        return has_defs or not has_external_refs
    
    # =========================================================================
    # Statistics Generation
    # =========================================================================
    
    def _generate_statistics(
        self,
        chunks: List[PolicyChunk],
        filtered_pages: List[int],
        policy_pages: List[Dict[str, Any]],
        duplicate_candidates: List[DuplicatePolicyCandidate],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about the chunking process.
        
        Args:
            chunks: Created chunks
            filtered_pages: Filtered page numbers
            policy_pages: Policy pages processed
            duplicate_candidates: Duplicate policy candidates
        
        Returns:
            Statistics dictionary
        """
        total_tokens = sum(c.token_count for c in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        
        # Count chunks by boundary type
        boundary_types = defaultdict(int)
        for chunk in chunks:
            boundary_type = chunk.metadata.get("boundary_type", "unknown")
            boundary_types[boundary_type] += 1
        
        # Count policies
        unique_policies = set()
        for chunk in chunks:
            unique_policies.update(chunk.policy_ids)
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": avg_tokens,
            "max_tokens": max((c.token_count for c in chunks), default=0),
            "min_tokens": min((c.token_count for c in chunks), default=0),
            "filtered_pages_count": len(filtered_pages),
            "policy_pages_count": len(policy_pages),
            "unique_policies_count": len(unique_policies),
            "duplicate_candidates_count": len(duplicate_candidates),
            "chunks_with_definitions": sum(1 for c in chunks if c.has_definitions),
            "chunks_with_complete_context": sum(
                1 for c in chunks if c.has_complete_context
            ),
            "boundary_type_distribution": dict(boundary_types),
        }
    
    # =========================================================================
    # Public Utility Methods
    # =========================================================================
    
    def get_chunk_summary(self, result: ChunkingResult) -> Dict[str, Any]:
        """
        Get summary of chunking results for logging/display.
        
        Args:
            result: ChunkingResult object
        
        Returns:
            Summary dictionary
        """
        return result.statistics
