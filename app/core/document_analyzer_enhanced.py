"""
Enhanced document intelligence analyzer with page-level content classification,
policy boundary detection, and content zone mapping.
"""
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from langchain_openai import ChatOpenAI
from config.settings import settings
from app.utils.logger import get_logger
from app.models.schemas import (
    EnhancedPDFPage,
    EnhancedPDFMetadata,
    PageContentType,
    PageAnalysis,
    PolicyBoundary,
    ContentZone,
    PolicyFlowNode,
    ReferenceInfo,
    EnhancedDocumentMetadata,
    DocumentType,
    HeadingInfo
)
from datetime import datetime

logger = get_logger(__name__)


class EnhancedDocumentAnalyzer:
    """
    Enhanced document analyzer providing:
    - Page-level content classification
    - Policy boundary detection
    - Content zone mapping
    - Semantic continuity tracking
    - Policy flow graph
    - Reference detection
    """

    # Keywords for content type detection
    CONTENT_TYPE_KEYWORDS = {
        PageContentType.DEFINITIONS: {
            'definitions', 'defined terms', 'terminology', 'glossary', 'means',
            'shall mean', 'is defined as', 'refers to'
        },
        PageContentType.EXCLUSIONS: {
            'exclusions', 'limitations', 'not covered', 'excluded', 'does not cover',
            'limitations and exclusions', 'what is not covered'
        },
        PageContentType.COVERAGE: {
            'covered services', 'coverage', 'benefits', 'covered benefits',
            'what is covered', 'eligible services', 'covered procedures'
        },
        PageContentType.ELIGIBILITY: {
            'eligibility', 'eligible', 'qualification', 'qualifies', 'who is eligible',
            'member eligibility', 'enrollment'
        },
        PageContentType.PROCEDURES: {
            'procedure', 'process', 'how to', 'steps to', 'filing a claim',
            'appeals process', 'authorization'
        },
        PageContentType.TABLE_OF_CONTENTS: {
            'table of contents', 'contents', 'section', 'page number'
        },
        PageContentType.REFERENCES: {
            'references', 'see also', 'refer to', 'cross reference', 'reference'
        },
        PageContentType.BIBLIOGRAPHY: {
            'bibliography', 'works cited', 'citations', 'sources'
        },
        PageContentType.INDEX: {
            'index', 'alphabetical index'
        },
        PageContentType.APPENDIX: {
            'appendix', 'attachment', 'exhibit', 'schedule'
        },
        PageContentType.ADMINISTRATIVE: {
            'cover page', 'disclaimer', 'notice', 'copyright', 'proprietary',
            'confidential', 'internal use', 'disclaimer of liability'
        }
    }

    # Major heading patterns indicating policy starts
    POLICY_START_PATTERNS = [
        r'^\d+\.\s+[A-Z]',  # Numbered headings (1. POLICY)
        r'^[A-Z\s]{3,}$',  # All caps headings
        r'^\d+\.\d+\s+[A-Z]',  # Sub-numbered (1.1 COVERAGE)
        r'^POLICY\s*:',  # Explicit POLICY: prefix
        r'^COVERAGE\s*:',  # Explicit COVERAGE: prefix
    ]

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize enhanced document analyzer.
        
        Args:
            llm: Optional pre-configured LLM client for classification
        """
        self.llm = llm

    async def analyze_document(
        self,
        pages: List[EnhancedPDFPage],
        pdf_metadata: EnhancedPDFMetadata,
        llm: Optional[ChatOpenAI] = None
    ) -> EnhancedDocumentMetadata:
        """
        Perform comprehensive document analysis.

        Args:
            pages: List of EnhancedPDFPage objects
            pdf_metadata: Enhanced PDF metadata with structure info
            llm: Optional LLM for advanced classification

        Returns:
            EnhancedDocumentMetadata with complete analysis
        """
        logger.info("Starting enhanced document analysis...")
        start_time = datetime.utcnow()

        if llm is not None:
            self.llm = llm

        # Step 1: Per-page content classification
        logger.info("Classifying page content types...")
        page_analyses = []
        for page in pages:
            analysis = await self._classify_page_content(page, pdf_metadata)
            page_analyses.append(analysis)

        # Step 2: Policy boundary detection
        logger.info("Detecting policy boundaries...")
        policy_boundaries = self._detect_policy_boundaries(
            pages, pdf_metadata.headings, page_analyses
        )
        # Update page analyses with policy boundaries
        self._assign_boundaries_to_pages(page_analyses, policy_boundaries)

        # Step 2.5: Enhanced policy document structure analysis
        logger.info("Analyzing policy document structure...")
        document_structure = self.analyze_policy_document_structure(pages, pdf_metadata)

        # Step 3: Content zone mapping
        logger.info("Mapping content zones...")
        content_zones = self._map_content_zones(page_analyses)

        # Step 4: Semantic continuity detection
        logger.info("Detecting semantic continuity...")
        self._detect_continuity_markers(pages, page_analyses)

        # Step 5: Build policy flow map
        logger.info("Building policy flow map...")
        policy_flow_map = self._build_policy_flow_map(
            pages, policy_boundaries, content_zones, pdf_metadata.headings
        )

        # Step 6: Reference detection
        logger.info("Detecting references...")
        references = self._detect_references(pages)

        # Step 7: Calculate statistics and guidance
        logger.info("Calculating statistics and chunking guidance...")
        stats = self._calculate_statistics(page_analyses, content_zones)
        
        # Step 8: Generate chunking guidance
        chunk_boundaries, pages_to_filter, special_handling = self._generate_chunking_guidance(
            page_analyses, content_zones, policy_flow_map
        )

        # Step 9: Document type and complexity (from original analyzer)
        doc_type = await self._determine_document_type(pages[:3], page_analyses[:3])
        complexity = self._calculate_complexity(pages, page_analyses, content_zones)
        structure_type = self._determine_structure_type(pdf_metadata, content_zones)

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Build enhanced metadata
        metadata = EnhancedDocumentMetadata(
            # Basic info
            document_type=doc_type,
            total_pages=len(pages),
            complexity_score=complexity,
            has_images=any(len(p.images) > 0 for p in pages),
            has_tables=any(len(p.tables) > 0 for p in pages),
            structure_type=structure_type,
            language="en",
            processing_time_seconds=processing_time,
            
            # Enhanced analysis
            page_analyses=page_analyses,
            content_zones=content_zones,
            policy_flow_map=policy_flow_map,
            policy_boundaries=policy_boundaries,
            document_structure=document_structure,  # NEW: Enhanced structure analysis
            
            # Statistics
            policy_pages_count=stats['policy_pages'],
            admin_pages_count=stats['admin_pages'],
            mixed_pages_count=stats['mixed_pages'],
            
            # Zones
            main_policy_zone=stats.get('main_policy_zone'),
            definitions_zone=stats.get('definitions_zone'),
            exclusions_zone=stats.get('exclusions_zone'),
            references_zone=stats.get('references_zone'),
            
            # References
            references=references,
            has_internal_references=any(r.reference_type == 'internal' for r in references),
            has_external_references=any(r.reference_type == 'external' for r in references),
            
            # Chunking guidance
            recommended_chunk_boundaries=chunk_boundaries,
            pages_to_filter=pages_to_filter,
            pages_requiring_special_handling=special_handling,
            
            # Quality
            overall_extractability_score=stats['extractability_score'],
            estimated_policy_count=len(policy_flow_map),
            confidence_in_structure=stats['structure_confidence']
        )

        logger.info(
            f"Enhanced analysis complete: type={doc_type}, policies={len(policy_flow_map)}, "
            f"zones={len(content_zones)}, complexity={complexity:.2f}"
        )

        return metadata

    async def _classify_page_content(
        self,
        page: EnhancedPDFPage,
        pdf_metadata: EnhancedPDFMetadata
    ) -> PageAnalysis:
        """
        Classify the content type of a single page.

        Args:
            page: EnhancedPDFPage to classify
            pdf_metadata: Document metadata for context

        Returns:
            PageAnalysis with content classification
        """
        # Initialize scores
        content_type_scores = {ct: 0.0 for ct in PageContentType}
        
        text_lower = page.text.lower()
        text_lines = page.text.strip().split('\n')

        # Keyword-based scoring
        for content_type, keywords in self.CONTENT_TYPE_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            content_type_scores[content_type] = matches / len(keywords)

        # Check if page is in TOC (from PDF metadata)
        is_toc = any(
            toc.page_number == page.page_number
            for toc in pdf_metadata.toc_entries
        )
        if is_toc:
            content_type_scores[PageContentType.TABLE_OF_CONTENTS] = 1.0

        # Check position in document
        position_ratio = page.page_number / pdf_metadata.total_pages
        if position_ratio < 0.1:  # First 10% likely admin/TOC
            content_type_scores[PageContentType.ADMINISTRATIVE] += 0.3
            content_type_scores[PageContentType.TABLE_OF_CONTENTS] += 0.2
        elif position_ratio > 0.9:  # Last 10% likely appendix/index
            content_type_scores[PageContentType.APPENDIX] += 0.3
            content_type_scores[PageContentType.INDEX] += 0.3
            content_type_scores[PageContentType.REFERENCES] += 0.2

        # Check for list-heavy content (definitions often list-like)
        if len([line for line in text_lines if line.strip().startswith(('•', '-', '*'))]) > 10:
            content_type_scores[PageContentType.DEFINITIONS] += 0.2

        # Check headings on this page
        page_headings = [h for h in pdf_metadata.headings if h.page_number == page.page_number]
        for heading in page_headings:
            heading_lower = heading.text.lower()
            for content_type, keywords in self.CONTENT_TYPE_KEYWORDS.items():
                if any(kw in heading_lower for kw in keywords):
                    content_type_scores[content_type] += 0.4

        # Use page importance score to boost policy content
        if page.importance_score > 0.7:
            content_type_scores[PageContentType.POLICY_CONTENT] += 0.3

        # Default to policy content if no strong signals
        if max(content_type_scores.values()) < 0.3:
            content_type_scores[PageContentType.POLICY_CONTENT] = 0.5

        # Determine primary and secondary types
        sorted_types = sorted(content_type_scores.items(), key=lambda x: x[1], reverse=True)
        primary_type = sorted_types[0][0]
        secondary_types = [t for t, score in sorted_types[1:4] if score > 0.2]

        # Calculate content ratios
        policy_ratio = self._calculate_policy_ratio(page)
        admin_ratio = self._calculate_admin_ratio(page)
        filterable_ratio = self._calculate_filterable_ratio(primary_type, policy_ratio)

        # Detect structural elements
        has_headings = len(page_headings) > 0
        has_numbered = bool(re.search(r'^\s*\d+\.', page.text, re.MULTILINE))
        has_bullets = bool(re.search(r'^\s*[•\-\*]', page.text, re.MULTILINE))

        # Determine chunk priority
        chunk_priority = self._calculate_chunk_priority(
            primary_type, policy_ratio, page.importance_score
        )

        # Should include in extraction?
        should_include = primary_type not in {
            PageContentType.TABLE_OF_CONTENTS,
            PageContentType.INDEX,
            PageContentType.ADMINISTRATIVE
        } and filterable_ratio < 0.8

        return PageAnalysis(
            page_number=page.page_number,
            primary_content_type=primary_type,
            secondary_content_types=secondary_types,
            policy_content_ratio=policy_ratio,
            administrative_content_ratio=admin_ratio,
            filterable_content_ratio=filterable_ratio,
            policy_boundary=PolicyBoundary(page_number=page.page_number),  # Filled in later
            has_headings=has_headings,
            has_numbered_items=has_numbered,
            has_bullet_points=has_bullets,
            has_tables=len(page.tables) > 0,
            has_references=bool(re.search(r'\bsee\s+(page|section|chapter)\b', page.text, re.IGNORECASE)),
            is_dense_text=len(text_lines) > 40 and not has_bullets,
            is_list_heavy=has_bullets and len([l for l in text_lines if l.strip().startswith(('•', '-', '*'))]) > 15,
            is_table_heavy=len(page.tables) > 2,
            should_include_in_extraction=should_include,
            chunk_priority=chunk_priority
        )

    def _detect_policy_boundaries(
        self,
        pages: List[EnhancedPDFPage],
        headings: List[HeadingInfo],
        page_analyses: List[PageAnalysis]
    ) -> List[PolicyBoundary]:
        """
        Detect where policies start and end.

        Args:
            pages: All pages
            headings: Heading information
            page_analyses: Page analyses

        Returns:
            List of PolicyBoundary objects
        """
        boundaries = []
        
        # Group headings by level to identify major sections
        major_headings = [h for h in headings if h.level <= 2]
        
        for i, heading in enumerate(major_headings):
            # Check if this looks like a policy start
            is_policy_start = self._is_policy_heading(heading.text)
            
            if not is_policy_start:
                continue
            
            # Determine policy extent
            start_page = heading.page_number
            end_page = major_headings[i + 1].page_number - 1 if i + 1 < len(major_headings) else pages[-1].page_number
            
            # Create boundaries for each page in the policy
            for page_num in range(start_page, end_page + 1):
                policy_id = f"policy_{heading.text[:30].replace(' ', '_')}_{start_page}"
                
                boundary = PolicyBoundary(
                    page_number=page_num,
                    policy_starts_here=(page_num == start_page),
                    policy_continues_from_previous=(page_num > start_page),
                    policy_ends_here=(page_num == end_page),
                    policy_continues_to_next=(page_num < end_page),
                    policy_ids=[policy_id],
                    heading_at_start=heading.text if page_num == start_page else None
                )
                boundaries.append(boundary)
        
        # Handle pages without explicit boundaries
        pages_with_boundaries = {b.page_number for b in boundaries}
        for page_num in range(1, len(pages) + 1):
            if page_num not in pages_with_boundaries:
                # Create default boundary
                boundary = PolicyBoundary(
                    page_number=page_num,
                    policy_ids=["unassigned"]
                )
                boundaries.append(boundary)
        
        return sorted(boundaries, key=lambda b: b.page_number)

    def _map_content_zones(self, page_analyses: List[PageAnalysis]) -> List[ContentZone]:
        """
        Map contiguous zones of similar content.

        Args:
            page_analyses: Page analyses

        Returns:
            List of ContentZone objects
        """
        zones = []
        
        if not page_analyses:
            return zones
        
        # Start first zone
        current_zone_type = page_analyses[0].primary_content_type
        current_zone_start = 1
        zone_id_counter = 0
        
        for i, analysis in enumerate(page_analyses):
            page_num = i + 1
            
            # Check if zone type changes
            if analysis.primary_content_type != current_zone_type:
                # End current zone
                zone = ContentZone(
                    zone_id=f"zone_{zone_id_counter}",
                    zone_type=current_zone_type,
                    start_page=current_zone_start,
                    end_page=page_num - 1,
                    page_count=(page_num - 1) - current_zone_start + 1,
                    average_importance=sum(
                        page_analyses[j].chunk_priority 
                        for j in range(current_zone_start - 1, page_num - 1)
                    ) / ((page_num - 1) - current_zone_start + 1),
                    should_extract_policies=(current_zone_type == PageContentType.POLICY_CONTENT)
                )
                zones.append(zone)
                
                # Start new zone
                current_zone_type = analysis.primary_content_type
                current_zone_start = page_num
                zone_id_counter += 1
        
        # Add final zone
        zone = ContentZone(
            zone_id=f"zone_{zone_id_counter}",
            zone_type=current_zone_type,
            start_page=current_zone_start,
            end_page=len(page_analyses),
            page_count=len(page_analyses) - current_zone_start + 1,
            average_importance=sum(
                page_analyses[j].chunk_priority 
                for j in range(current_zone_start - 1, len(page_analyses))
            ) / (len(page_analyses) - current_zone_start + 1),
            should_extract_policies=(current_zone_type == PageContentType.POLICY_CONTENT)
        )
        zones.append(zone)
        
        return zones

    def _detect_continuity_markers(
        self,
        pages: List[EnhancedPDFPage],
        page_analyses: List[PageAnalysis]
    ):
        """
        Detect semantic continuity across pages (modifies page_analyses in place).

        Args:
            pages: All pages
            page_analyses: Page analyses to update
        """
        for i, (page, analysis) in enumerate(zip(pages, page_analyses)):
            text = page.text.strip()
            
            if not text:
                continue
            
            # Check if starts mid-sentence (lowercase start, no sentence-ending punctuation before)
            first_char = text[0] if text else ''
            starts_mid = first_char.islower() or text.startswith(('and', 'or', 'but', 'which', 'that'))
            analysis.starts_mid_sentence = starts_mid
            
            # Check if ends mid-sentence (no punctuation, or ends with comma)
            last_char = text[-1] if text else ''
            ends_mid = last_char not in '.!?' or text.endswith(',')
            analysis.ends_mid_sentence = ends_mid
            
            # Check for list continuation (starts with lowercase letter and number/bullet)
            has_list_cont = bool(re.match(r'^\s*([a-z]\)|\d+\.|\-|\•)', text))
            analysis.has_list_continuation = has_list_cont

    def _build_policy_flow_map(
        self,
        pages: List[EnhancedPDFPage],
        policy_boundaries: List[PolicyBoundary],
        content_zones: List[ContentZone],
        headings: List[HeadingInfo]
    ) -> List[PolicyFlowNode]:
        """
        Build policy flow graph tracking policies across pages.

        Args:
            pages: All pages
            policy_boundaries: Policy boundaries
            content_zones: Content zones
            headings: Headings

        Returns:
            List of PolicyFlowNode objects
        """
        policy_map = {}
        
        # Group boundaries by policy_id
        for boundary in policy_boundaries:
            for policy_id in boundary.policy_ids:
                if policy_id == "unassigned":
                    continue
                
                if policy_id not in policy_map:
                    policy_map[policy_id] = {
                        'pages': [],
                        'heading': boundary.heading_at_start or policy_id
                    }
                
                policy_map[policy_id]['pages'].append(boundary.page_number)
        
        # Build flow nodes
        flow_nodes = []
        for policy_id, data in policy_map.items():
            pages_list = sorted(data['pages'])
            start_page = pages_list[0]
            end_page = pages_list[-1]
            
            # Check for tables/lists in policy span
            has_tables = any(len(pages[p - 1].tables) > 0 for p in pages_list if p <= len(pages))
            has_lists = any(
                bool(re.search(r'^\s*[•\-\*\d+\.]', pages[p - 1].text, re.MULTILINE))
                for p in pages_list if p <= len(pages)
            )
            
            # Detect heading types
            policy_headings = [h for h in headings if h.page_number in pages_list]
            heading_texts_lower = [h.text.lower() for h in policy_headings]
            has_definitions = any('definition' in h for h in heading_texts_lower)
            has_exclusions = any('exclusion' in h or 'limitation' in h for h in heading_texts_lower)
            
            node = PolicyFlowNode(
                policy_id=policy_id,
                policy_title=data['heading'],
                start_page=start_page,
                end_page=end_page,
                spans_multiple_pages=(end_page > start_page),
                continuation_points=pages_list[1:-1] if len(pages_list) > 2 else [],
                has_tables=has_tables,
                has_lists=has_lists,
                has_definitions=has_definitions,
                has_exclusions=has_exclusions,
                estimated_confidence=0.8  # Default confidence
            )
            flow_nodes.append(node)
        
        return flow_nodes

    def _detect_references(self, pages: List[EnhancedPDFPage]) -> List[ReferenceInfo]:
        """
        Detect internal and external references.

        Args:
            pages: All pages

        Returns:
            List of ReferenceInfo objects
        """
        references = []
        
        # Patterns for internal references
        internal_patterns = [
            r'\bsee\s+page\s+(\d+)\b',
            r'\bsee\s+section\s+(\d+\.?\d*)\b',
            r'\brefer\s+to\s+page\s+(\d+)\b',
            r'\b\(see\s+page\s+(\d+)\)',
        ]
        
        for page in pages:
            text = page.text
            
            # Find internal references
            for pattern in internal_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    ref = ReferenceInfo(
                        reference_type='internal',
                        source_page=page.page_number,
                        reference_text=match.group(0),
                        target_page=int(match.group(1)) if match.group(1).isdigit() else None
                    )
                    references.append(ref)
        
        return references

    def _calculate_statistics(
        self,
        page_analyses: List[PageAnalysis],
        content_zones: List[ContentZone]
    ) -> Dict[str, Any]:
        """
        Calculate statistics from analyses.

        Args:
            page_analyses: Page analyses
            content_zones: Content zones

        Returns:
            Statistics dictionary
        """
        policy_pages = sum(
            1 for a in page_analyses
            if a.primary_content_type == PageContentType.POLICY_CONTENT
        )
        admin_pages = sum(
            1 for a in page_analyses
            if a.primary_content_type == PageContentType.ADMINISTRATIVE
        )
        mixed_pages = sum(
            1 for a in page_analyses
            if a.primary_content_type == PageContentType.MIXED
        )
        
        # Find specific zones
        main_policy_zone = next(
            (z for z in content_zones if z.zone_type == PageContentType.POLICY_CONTENT),
            None
        )
        definitions_zone = next(
            (z for z in content_zones if z.zone_type == PageContentType.DEFINITIONS),
            None
        )
        exclusions_zone = next(
            (z for z in content_zones if z.zone_type == PageContentType.EXCLUSIONS),
            None
        )
        references_zone = next(
            (z for z in content_zones if z.zone_type == PageContentType.REFERENCES),
            None
        )
        
        # Calculate extractability score (based on policy content ratio)
        total_pages = len(page_analyses)
        extractability = policy_pages / total_pages if total_pages > 0 else 0.5
        
        # Structure confidence (based on clear zone boundaries)
        structure_confidence = min(1.0, len(content_zones) / (total_pages / 10))
        
        return {
            'policy_pages': policy_pages,
            'admin_pages': admin_pages,
            'mixed_pages': mixed_pages,
            'main_policy_zone': main_policy_zone,
            'definitions_zone': definitions_zone,
            'exclusions_zone': exclusions_zone,
            'references_zone': references_zone,
            'extractability_score': extractability,
            'structure_confidence': structure_confidence
        }

    def _generate_chunking_guidance(
        self,
        page_analyses: List[PageAnalysis],
        content_zones: List[ContentZone],
        policy_flow_map: List[PolicyFlowNode]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Generate chunking guidance for downstream processing.

        Args:
            page_analyses: Page analyses
            content_zones: Content zones
            policy_flow_map: Policy flow map

        Returns:
            Tuple of (chunk_boundaries, pages_to_filter, pages_requiring_special_handling)
        """
        # Recommended chunk boundaries (based on policy boundaries)
        chunk_boundaries = [
            (node.start_page, node.end_page)
            for node in policy_flow_map
        ]
        
        # Pages to filter (admin, TOC, index, etc.)
        pages_to_filter = [
            a.page_number for a in page_analyses
            if a.primary_content_type in {
                PageContentType.TABLE_OF_CONTENTS,
                PageContentType.INDEX,
                PageContentType.ADMINISTRATIVE,
                PageContentType.BIBLIOGRAPHY
            }
        ]
        
        # Pages requiring special handling (table-heavy, list-heavy)
        special_handling = [
            a.page_number for a in page_analyses
            if a.requires_special_handling or a.is_table_heavy or a.has_list_continuation
        ]
        
        return chunk_boundaries, pages_to_filter, special_handling

    # ===== Helper Methods =====

    def _assign_boundaries_to_pages(
        self,
        page_analyses: List[PageAnalysis],
        policy_boundaries: List[PolicyBoundary]
    ):
        """Assign policy boundaries to page analyses (in place)."""
        boundary_map = {b.page_number: b for b in policy_boundaries}
        
        for analysis in page_analyses:
            if analysis.page_number in boundary_map:
                analysis.policy_boundary = boundary_map[analysis.page_number]

    def _is_policy_heading(self, heading_text: str) -> bool:
        """Check if heading looks like a policy start."""
        for pattern in self.POLICY_START_PATTERNS:
            if re.match(pattern, heading_text.strip()):
                return True
        
        # Check for policy-related keywords
        heading_lower = heading_text.lower()
        policy_keywords = {'policy', 'coverage', 'benefit', 'procedure', 'treatment', 'service'}
        return any(kw in heading_lower for kw in policy_keywords)

    def _calculate_policy_ratio(self, page: EnhancedPDFPage) -> float:
        """Calculate policy content ratio."""
        # Use importance score as proxy
        return page.importance_score

    def _calculate_admin_ratio(self, page: EnhancedPDFPage) -> float:
        """Calculate administrative content ratio."""
        # Inverse of policy ratio
        return 1.0 - page.importance_score

    def _calculate_filterable_ratio(
        self,
        primary_type: PageContentType,
        policy_ratio: float
    ) -> float:
        """Calculate filterable content ratio."""
        if primary_type in {
            PageContentType.TABLE_OF_CONTENTS,
            PageContentType.INDEX,
            PageContentType.BIBLIOGRAPHY,
            PageContentType.ADMINISTRATIVE
        }:
            return 0.9
        
        return 1.0 - policy_ratio

    def _calculate_chunk_priority(
        self,
        primary_type: PageContentType,
        policy_ratio: float,
        importance_score: float
    ) -> float:
        """Calculate chunk priority."""
        base_priority = 0.5
        
        if primary_type == PageContentType.POLICY_CONTENT:
            base_priority = 0.9
        elif primary_type in {PageContentType.DEFINITIONS, PageContentType.COVERAGE}:
            base_priority = 0.8
        elif primary_type in {PageContentType.EXCLUSIONS, PageContentType.ELIGIBILITY}:
            base_priority = 0.7
        
        # Adjust with importance score
        return min(1.0, base_priority * importance_score)

    async def _determine_document_type(
        self,
        sample_pages: List[EnhancedPDFPage],
        sample_analyses: List[PageAnalysis]
    ) -> DocumentType:
        """
        Determine document type using content analysis and optional LLM.

        Args:
            sample_pages: Sample pages
            sample_analyses: Sample page analyses

        Returns:
            DocumentType
        """
        # Check for insurance-specific content types
        has_insurance_types = any(
            a.primary_content_type in {
                PageContentType.COVERAGE,
                PageContentType.ELIGIBILITY,
                PageContentType.EXCLUSIONS
            }
            for a in sample_analyses
        )
        
        if has_insurance_types:
            return DocumentType.INSURANCE
        
        # Keyword-based classification
        sample_text = ' '.join(p.text for p in sample_pages).lower()
        
        if any(kw in sample_text for kw in ['insurance', 'coverage', 'policy', 'benefit']):
            return DocumentType.INSURANCE
        elif any(kw in sample_text for kw in ['regulation', 'compliance', 'regulatory']):
            return DocumentType.REGULATORY
        elif any(kw in sample_text for kw in ['contract', 'agreement', 'legal']):
            return DocumentType.LEGAL
        
        return DocumentType.UNKNOWN

    def _calculate_complexity(
        self,
        pages: List[EnhancedPDFPage],
        page_analyses: List[PageAnalysis],
        content_zones: List[ContentZone]
    ) -> float:
        """
        Calculate document complexity score.

        Args:
            pages: All pages
            page_analyses: Page analyses
            content_zones: Content zones

        Returns:
            Complexity score (0-1)
        """
        factors = []
        
        # Factor 1: Page count (0-1, normalized to 100 pages)
        page_factor = min(1.0, len(pages) / 100)
        factors.append(page_factor)
        
        # Factor 2: Number of content zones (more zones = more complex)
        zone_factor = min(1.0, len(content_zones) / 10)
        factors.append(zone_factor)
        
        # Factor 3: Mixed content ratio
        mixed_ratio = sum(
            1 for a in page_analyses
            if a.primary_content_type == PageContentType.MIXED
        ) / len(page_analyses)
        factors.append(mixed_ratio)
        
        # Factor 4: Table density
        table_density = sum(len(p.tables) for p in pages) / len(pages)
        table_factor = min(1.0, table_density / 3)
        factors.append(table_factor)
        
        # Factor 5: Has definitions/exclusions (increases complexity)
        has_complex_sections = any(
            z.zone_type in {PageContentType.DEFINITIONS, PageContentType.EXCLUSIONS}
            for z in content_zones
        )
        factors.append(0.7 if has_complex_sections else 0.3)
        
        return sum(factors) / len(factors)

    def _determine_structure_type(
        self,
        pdf_metadata: EnhancedPDFMetadata,
        content_zones: List[ContentZone]
    ) -> str:
        """
        Determine document structure type.

        Args:
            pdf_metadata: PDF metadata
            content_zones: Content zones

        Returns:
            Structure type string
        """
        has_toc = len(pdf_metadata.toc_entries) > 0
        has_headings = len(pdf_metadata.headings) > 0
        has_sections = len(pdf_metadata.section_boundaries) > 0
        zone_count = len(content_zones)
        
        if has_toc and has_headings and zone_count > 5:
            return "highly_structured"
        elif has_headings and has_sections:
            return "moderately_structured"
        elif has_headings or zone_count > 2:
            return "basic_structure"
        else:
            return "unstructured"

    def analyze_policy_document_structure(
        self,
        pages: List[EnhancedPDFPage],
        pdf_metadata: EnhancedPDFMetadata
    ) -> Dict[str, Any]:
        """
        Analyze insurance policy document structure to identify true policy boundaries.
        
        This method identifies:
        - Major policy sections (I, II, III or 1., 2., 3.)
        - Subsection hierarchy (A., B., C. or 1.1, 1.2)
        - Administrative pages (cover, TOC, disclaimers)
        - Numbering schemes and heading patterns
        
        Args:
            pages: List of EnhancedPDFPage objects
            pdf_metadata: Enhanced PDF metadata with headings
        
        Returns:
            Dictionary containing:
                - major_sections: List of major policy sections
                - administrative_pages: List of non-policy page numbers
                - policy_type: Type of policy document
                - complexity: Document complexity rating
                - recommended_chunking: Chunking strategy
        """
        logger.info("Analyzing policy document structure...")
        
        # Step 1: Detect heading hierarchy
        heading_hierarchy = self._detect_heading_hierarchy(pages, pdf_metadata)
        
        # Step 2: Identify major sections (level 1 policies)
        major_sections = self._identify_major_sections(heading_hierarchy, pages)
        
        # Step 3: Identify subsections for each major section
        for section in major_sections:
            section['subsections'] = self._identify_subsections(
                section, heading_hierarchy, pages
            )
        
        # Step 4: Identify administrative pages
        admin_pages = self._identify_administrative_pages(pages, major_sections)
        
        # Step 5: Determine policy document type
        policy_type = self._determine_policy_type(pages, major_sections)
        
        # Step 6: Calculate complexity
        complexity = self._calculate_document_complexity(major_sections, pages)
        
        # Step 7: Recommend chunking strategy
        chunking_strategy = self._recommend_chunking_strategy(
            major_sections, complexity, len(pages)
        )
        
        logger.info(
            f"Structure analysis complete: {len(major_sections)} major sections, "
            f"{len(admin_pages)} admin pages, type={policy_type}, complexity={complexity}"
        )
        
        return {
            'major_sections': major_sections,
            'administrative_pages': admin_pages,
            'policy_type': policy_type,
            'complexity': complexity,
            'recommended_chunking': chunking_strategy,
            'total_sections': len(major_sections),
            'avg_section_length': sum(s['end_page'] - s['start_page'] + 1 for s in major_sections) / len(major_sections) if major_sections else 0
        }
    
    def _detect_heading_hierarchy(
        self,
        pages: List[EnhancedPDFPage],
        pdf_metadata: EnhancedPDFMetadata
    ) -> List[Dict[str, Any]]:
        """
        Detect heading hierarchy from document structure using multiple signals.
        
        Enhanced in Phase 3.5 to use:
        - Numbering patterns (Roman, letters, decimals)
        - Semantic keywords (major policy vs subsection vs administrative)
        - Font metadata (size, formatting)
        - Contextual position
        - ALL CAPS vs Title Case analysis
        
        Detects:
        - Level 1 (Major sections): True policy sections with coverage criteria
        - Level 2 (Subsections): Requirements, considerations, guidelines
        - Level 3 (Administrative): Billing, coding, documentation (filtered out)
        
        Args:
            pages: List of pages
            pdf_metadata: PDF metadata with headings
        
        Returns:
            List of classified headings with hierarchy levels
        """
        # Semantic keyword filters
        ADMIN_KEYWORDS = {
            'place of service', 'billing', 'coding', 'benefit application',
            'required documentation', 'claims', 'authorization process',
            'reference', 'sources', 'citations'
        }
        
        MAJOR_POLICY_KEYWORDS = {
            'medical necessity', 'coverage criteria', 'eligibility',
            'indications', 'contraindications', 'exclusions', 'limitations',
            'covered services', 'medically necessary'
        }
        
        SUBSECTION_KEYWORDS = {
            'requirements', 'considerations', 'guidelines', 'procedures',
            'selection', 'evaluation', 'criteria', 'assessment'
        }
        
        REFERENCE_KEYWORDS = {
            'society', 'association', 'organization', 'committee',
            'references', 'bibliography', 'sources'
        }
        
        classified_headings = []
        
        # Patterns for different heading levels
        level_1_patterns = [
            r'^([IVX]+)\.\s+(.+)$',  # Roman numerals: "I. COVERAGE"
            r'^(\d+)\.\s+([A-Z][A-Z\s]{2,})$',  # Numbers + ALL CAPS: "1. MEDICAL NECESSITY"
            r'^([A-Z][A-Z\s]{10,})$',  # Long ALL CAPS: "COVERAGE CRITERIA"
        ]
        
        level_2_patterns = [
            r'^([A-Z])\.\s+(.+)$',  # Letters: "A. Adult Criteria"
            r'^(\d+)\.(\d+)\s+(.+)$',  # Decimals: "1.1 Initial Surgery"
        ]
        
        level_3_patterns = [
            r'^([a-z])\.\s+(.+)$',  # Small letters: "a. BMI requirements"
            r'^(\d+)\.(\d+)\.(\d+)\s+(.+)$',  # Triple decimals: "1.1.1 Age restrictions"
        ]
        
        for heading in pdf_metadata.headings:
            heading_text = heading.text.strip()
            heading_lower = heading_text.lower()
            
            classified = {
                'text': heading_text,
                'page_number': heading.page_number,
                'level': 0,
                'numbering': None,
                'title': heading_text,
                'pattern_match': None,
                'is_administrative': False,
                'is_reference': False,
            }
            
            # First check if this is administrative/reference (filter out later)
            if any(kw in heading_lower for kw in ADMIN_KEYWORDS):
                classified['level'] = 3
                classified['is_administrative'] = True
                classified['pattern_match'] = 'administrative'
            elif any(kw in heading_lower for kw in REFERENCE_KEYWORDS):
                classified['level'] = 3
                classified['is_reference'] = True
                classified['pattern_match'] = 'reference'
            
            # Check for major policy keywords
            elif any(kw in heading_lower for kw in MAJOR_POLICY_KEYWORDS):
                classified['level'] = 1
                classified['pattern_match'] = 'semantic_major_policy'
            
            # Check level 1 patterns (Roman numerals, ALL CAPS)
            elif not classified['level']:
                for pattern in level_1_patterns:
                    match = re.match(pattern, heading_text)
                    if match:
                        classified['level'] = 1
                        classified['numbering'] = match.group(1)
                        classified['title'] = match.group(2) if len(match.groups()) > 1 else heading_text
                        classified['pattern_match'] = 'level_1_pattern'
                        break
            
            # Check level 2 patterns if not level 1
            if classified['level'] == 0:
                # Check for subsection keywords
                if any(kw in heading_lower for kw in SUBSECTION_KEYWORDS):
                    classified['level'] = 2
                    classified['pattern_match'] = 'semantic_subsection'
                else:
                    for pattern in level_2_patterns:
                        match = re.match(pattern, heading_text)
                        if match:
                            classified['level'] = 2
                            if len(match.groups()) == 2:
                                classified['numbering'] = match.group(1)
                                classified['title'] = match.group(2)
                            else:  # Decimal numbering (1.1)
                                classified['numbering'] = f"{match.group(1)}.{match.group(2)}"
                                classified['title'] = match.group(3)
                            classified['pattern_match'] = 'level_2_pattern'
                            break
            
            # Check level 3 patterns if still unclassified
            if classified['level'] == 0:
                for pattern in level_3_patterns:
                    match = re.match(pattern, heading_text)
                    if match:
                        classified['level'] = 3
                        if len(match.groups()) == 2:
                            classified['numbering'] = match.group(1)
                            classified['title'] = match.group(2)
                        else:  # Triple decimal (1.1.1)
                            classified['numbering'] = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
                            classified['title'] = match.group(4)
                        classified['pattern_match'] = 'level_3_pattern'
                        break
            
            # If still unclassified, use heuristics
            if classified['level'] == 0:
                # Very long ALL CAPS likely level 1
                if heading_text.isupper() and len(heading_text) > 15:
                    classified['level'] = 1
                    classified['pattern_match'] = 'heuristic_long_caps'
                # Short ALL CAPS likely administrative
                elif heading_text.isupper() and len(heading_text.split()) <= 3:
                    classified['level'] = 3
                    classified['is_administrative'] = True
                    classified['pattern_match'] = 'heuristic_short_caps'
                # Title case with good length likely level 2
                elif heading_text[0].isupper() and len(heading_text.split()) > 2:
                    classified['level'] = 2
                    classified['pattern_match'] = 'heuristic_title'
                else:
                    classified['level'] = 2  # Default to level 2
                    classified['pattern_match'] = 'default'
            
            classified_headings.append(classified)
        
        # Filter out administrative and reference headings
        policy_headings = [
            h for h in classified_headings
            if not h.get('is_administrative') and not h.get('is_reference')
        ]
        
        logger.info(
            f"Classified {len(classified_headings)} headings: "
            f"Level 1: {sum(1 for h in policy_headings if h['level'] == 1)}, "
            f"Level 2: {sum(1 for h in policy_headings if h['level'] == 2)}, "
            f"Level 3: {sum(1 for h in policy_headings if h['level'] == 3)}, "
            f"Filtered out: {len(classified_headings) - len(policy_headings)} administrative/reference headings"
        )
        
        return policy_headings  # Return only policy headings
    
    def _identify_major_sections(
        self,
        heading_hierarchy: List[Dict[str, Any]],
        pages: List[EnhancedPDFPage]
    ) -> List[Dict[str, Any]]:
        """
        Identify major policy sections (level 1 headings).
        
        Args:
            heading_hierarchy: Classified headings
            pages: List of pages
        
        Returns:
            List of major section dictionaries
        """
        major_sections = []
        level_1_headings = [h for h in heading_hierarchy if h['level'] == 1]
        
        for i, heading in enumerate(level_1_headings):
            # Determine section boundaries
            start_page = heading['page_number']
            
            # End page is either:
            # 1. One page before next level 1 heading (but minimum = start_page for same-page sections)
            # 2. Last page of document
            if i + 1 < len(level_1_headings):
                next_heading_page = level_1_headings[i + 1]['page_number']
                
                # CRITICAL FIX: Handle consecutive headings on same page
                if next_heading_page == start_page:
                    # Multiple major sections on same page - section ends on same page
                    end_page = start_page
                    logger.info(
                        f"Section '{heading['title']}' shares page {start_page} with next section. "
                        f"Ending on same page."
                    )
                else:
                    # Normal case: end one page before next section
                    end_page = next_heading_page - 1
            else:
                end_page = len(pages)
            
            # Safety check: Ensure end_page >= start_page
            if end_page < start_page:
                logger.warning(
                    f"Section '{heading['title']}' calculated invalid range "
                    f"({start_page}-{end_page}). Setting end_page = start_page"
                )
                end_page = start_page
            
            # Create section ID from title
            section_id = self._create_section_id(heading['title'], heading['numbering'])
            
            section = {
                'section_id': section_id,
                'title': heading['title'],
                'numbering': heading['numbering'],
                'start_page': start_page,
                'end_page': end_page,
                'heading_level': 1,
                'page_count': end_page - start_page + 1,
                'subsections': []  # Will be filled later
            }
            
            major_sections.append(section)
        
        logger.info(f"Identified {len(major_sections)} major sections (filtered from {len(heading_hierarchy)} total headings)")
        
        return major_sections
    
    def _identify_subsections(
        self,
        parent_section: Dict[str, Any],
        heading_hierarchy: List[Dict[str, Any]],
        pages: List[EnhancedPDFPage]
    ) -> List[Dict[str, Any]]:
        """
        Identify subsections within a major section.
        
        Args:
            parent_section: Parent major section
            heading_hierarchy: Classified headings
            pages: List of pages
        
        Returns:
            List of subsection dictionaries
        """
        subsections = []
        
        # Find level 2 and 3 headings within parent section's page range
        section_headings = [
            h for h in heading_hierarchy
            if h['level'] in [2, 3] and 
               parent_section['start_page'] <= h['page_number'] <= parent_section['end_page']
        ]
        
        # Sort by page number and level
        section_headings.sort(key=lambda h: (h['page_number'], h['level']))
        
        for i, heading in enumerate(section_headings):
            start_page = heading['page_number']
            
            # End page logic:
            # 1. If there's a next heading of same/higher level (within section), end before it
            # 2. Otherwise, end at parent section end
            end_page = parent_section['end_page']
            for j in range(i + 1, len(section_headings)):
                next_heading = section_headings[j]
                if next_heading['level'] <= heading['level']:
                    end_page = next_heading['page_number'] - 1
                    break
            
            subsection_id = self._create_section_id(
                heading['title'],
                heading['numbering'],
                parent_id=parent_section['section_id']
            )
            
            subsection = {
                'subsection_id': subsection_id,
                'title': heading['title'],
                'numbering': heading['numbering'],
                'start_page': start_page,
                'end_page': end_page,
                'heading_level': heading['level'],
                'page_count': end_page - start_page + 1,
                'parent_section': parent_section['section_id']
            }
            
            subsections.append(subsection)
        
        logger.debug(
            f"Identified {len(subsections)} subsections in '{parent_section['title']}'"
        )
        
        return subsections
    
    def _create_section_id(
        self,
        title: str,
        numbering: Optional[str],
        parent_id: Optional[str] = None
    ) -> str:
        """
        Create a unique, descriptive section ID.
        
        Args:
            title: Section title
            numbering: Section numbering (e.g., "I", "A", "1.1")
            parent_id: Optional parent section ID
        
        Returns:
            Section ID in snake_case
        """
        # Clean and convert title to snake_case
        # Remove common prefixes
        clean_title = re.sub(r'^(Section|Chapter|Part|Article)\s+', '', title, flags=re.IGNORECASE)
        
        # Convert to snake_case
        snake_case = re.sub(r'[^\w\s-]', '', clean_title.lower())
        snake_case = re.sub(r'[-\s]+', '_', snake_case)
        snake_case = snake_case.strip('_')
        
        # Limit length
        if len(snake_case) > 50:
            words = snake_case.split('_')
            snake_case = '_'.join(words[:5])  # Take first 5 words
        
        # Add parent prefix if provided
        if parent_id:
            # Extract key terms from parent (first 2 words)
            parent_words = parent_id.split('_')[:2]
            snake_case = f"{'_'.join(parent_words)}_{snake_case}"
        
        return snake_case
    
    def _identify_administrative_pages(
        self,
        pages: List[EnhancedPDFPage],
        major_sections: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Identify administrative pages (cover, TOC, disclaimers, etc.).
        
        Args:
            pages: List of pages
            major_sections: Identified major sections
        
        Returns:
            List of administrative page numbers
        """
        admin_pages = []
        
        # Keywords that indicate administrative content
        admin_keywords = {
            'table of contents', 'contents', 'copyright', 'proprietary',
            'confidential', 'internal use', 'disclaimer', 'notice',
            'cover page', 'title page', 'revision history'
        }
        
        # Check each page
        for i, page in enumerate(pages, start=1):
            text_lower = page.text.lower()
            
            # Check for admin keywords
            has_admin_keywords = sum(1 for kw in admin_keywords if kw in text_lower) >= 2
            
            # Check if page is very short (likely cover/disclaimer)
            is_very_short = len(page.text.strip().split()) < 50
            
            # Check if page has no policy content (not in any major section)
            in_major_section = any(
                s['start_page'] <= i <= s['end_page']
                for s in major_sections
            )
            
            # Mark as administrative if:
            # 1. Has multiple admin keywords
            # 2. Very short and not in major section
            # 3. Page 1 and very short (likely cover)
            if has_admin_keywords or (is_very_short and not in_major_section) or (i == 1 and is_very_short):
                admin_pages.append(i)
        
        logger.debug(f"Identified {len(admin_pages)} administrative pages: {admin_pages}")
        
        return admin_pages
    
    def _determine_policy_type(
        self,
        pages: List[EnhancedPDFPage],
        major_sections: List[Dict[str, Any]]
    ) -> str:
        """
        Determine the type of policy document.
        
        Args:
            pages: List of pages
            major_sections: Major sections
        
        Returns:
            Policy type string
        """
        # Combine all section titles
        all_titles = ' '.join(s['title'].lower() for s in major_sections)
        combined_text = ' '.join(p.text.lower() for p in pages[:3])  # First 3 pages
        
        # Check for policy type indicators
        if 'medical necessity' in all_titles or 'coverage criteria' in all_titles:
            return 'medical_coverage'
        elif 'bariatric' in combined_text or 'surgery' in all_titles:
            return 'surgical_procedure'
        elif 'exclusion' in all_titles or 'limitation' in all_titles:
            return 'exclusions'
        elif 'benefit' in all_titles or 'covered service' in all_titles:
            return 'benefits'
        else:
            return 'general_policy'
    
    def _calculate_document_complexity(
        self,
        major_sections: List[Dict[str, Any]],
        pages: List[EnhancedPDFPage]
    ) -> str:
        """
        Calculate document complexity rating.
        
        Args:
            major_sections: Major sections
            pages: List of pages
        
        Returns:
            Complexity rating: 'low', 'medium', 'high'
        """
        # Factors:
        # 1. Number of major sections
        section_count = len(major_sections)
        
        # 2. Total subsections
        subsection_count = sum(len(s.get('subsections', [])) for s in major_sections)
        
        # 3. Average section length
        avg_section_length = sum(s['page_count'] for s in major_sections) / len(major_sections) if major_sections else 0
        
        # 4. Document length
        doc_length = len(pages)
        
        # Calculate score
        complexity_score = 0
        
        if section_count > 5:
            complexity_score += 2
        elif section_count > 3:
            complexity_score += 1
        
        if subsection_count > 10:
            complexity_score += 2
        elif subsection_count > 5:
            complexity_score += 1
        
        if avg_section_length > 3:
            complexity_score += 1
        
        if doc_length > 10:
            complexity_score += 1
        
        # Classify
        if complexity_score >= 5:
            return 'high'
        elif complexity_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _recommend_chunking_strategy(
        self,
        major_sections: List[Dict[str, Any]],
        complexity: str,
        total_pages: int
    ) -> str:
        """
        Recommend chunking strategy based on document structure.
        
        Args:
            major_sections: Major sections
            complexity: Complexity rating
            total_pages: Total pages
        
        Returns:
            Recommended chunking strategy
        """
        if complexity == 'high' and len(major_sections) > 5:
            return 'section_based_with_subsections'
        elif complexity == 'medium' or len(major_sections) > 3:
            return 'section_based'
        elif total_pages < 5:
            return 'full_document'
        else:
            return 'page_range_based'

