"""
Intelligent chunking system for managing large documents that exceed LLM context windows.
Uses LLM assistance to identify policy boundaries and prevent splitting policies across chunks.
"""
import re
import tiktoken
import json
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from app.utils.logger import get_logger
from app.core.pdf_processor import PDFPage
from config.settings import settings

logger = get_logger(__name__)


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


class ChunkingStrategy:
    """Intelligent chunking strategy that respects semantic boundaries."""

    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None,
        use_llm: bool = True,
        llm: Optional[ChatOpenAI] = None,
    ):
        """
        Initialize chunking strategy.

        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap size in tokens
            use_llm: Whether to use LLM for intelligent boundary detection
            llm: Optional pre-configured LLM client
        """
        self.target_chunk_size = chunk_size or settings.target_chunk_tokens
        self.max_chunk_size = settings.max_chunk_tokens
        self.overlap = overlap or settings.chunk_overlap
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.use_llm = use_llm
        self.llm = llm
        logger.info(
            f"Init chunking: target={self.target_chunk_size}, max={self.max_chunk_size}, "
            f"overlap={self.overlap}, llm={use_llm}"
        )

    async def _analyze_document_structure_with_llm(
        self, pages: List[PDFPage]
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze document structure and identify policy boundaries.

        Args:
            pages: List of PDFPage objects

        Returns:
            Dictionary with structure analysis including policy boundaries
        """
        logger.info(f"Analyze structure: {len(pages)} pages, sampling first 3")

        sample_pages = pages[:3]
        sample_text = "\n\n".join([f"--- Page {p.page_number} ---\n{p.text}" for p in sample_pages])

        tokens = self.encoding.encode(sample_text)
        original_token_count = len(tokens)
        if len(tokens) > 6000:
            sample_text = self.encoding.decode(tokens[:6000])
            logger.debug(f"Truncated sample: {original_token_count} -> 6000 tokens")
        else:
            logger.debug(f"Sample tokens: {original_token_count}")

        prompt = f"""Analyze this policy document and identify its structure to help with intelligent chunking.

Document Sample (first pages):
{sample_text}

Please provide a JSON response with:
1. "document_type": Type of policy document (insurance, legal, regulatory, etc.)
2. "policy_markers": List of patterns that indicate the START of a new policy/section (e.g., numbered sections like "1.", "Section 1", article numbers, etc.)
3. "hierarchy_indicators": Patterns that show sub-policies (e.g., "a.", "i.", bullet points)
4. "boundary_rules": Rules for where policies typically end (e.g., before next numbered section, at specific keywords)
5. "context_keywords": Important keywords that provide context (e.g., "Coverage", "Exclusions", "Definitions")

Focus on identifying clear, unambiguous patterns that indicate policy boundaries.

Return ONLY valid JSON, no markdown formatting."""

        try:
            if self.llm is None:
                logger.warning("LLM not initialized, using heuristics")
                return {
                    "document_type": "unknown",
                    "policy_markers": [r"^\d+\.", r"^Section \d+", r"^Article \d+"],
                    "hierarchy_indicators": [r"^\s*[a-z]\.", r"^\s*[ivx]+\.", r"^\s*\([0-9]+\)"],
                    "boundary_rules": ["before_next_marker"],
                    "context_keywords": ["coverage", "exclusion", "definition", "requirement"]
                }

            logger.debug("Calling LLM API for structure analysis...")
            response = await self.llm.ainvoke(prompt)

            logger.debug("Received structure analysis response from LLM")
            content = response.content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                logger.debug("Removing markdown code blocks from LLM response...")
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            logger.debug("Parsing structure analysis JSON...")
            structure_analysis = json.loads(content)
            logger.info(f"LLM structure analysis complete - Document type: {structure_analysis.get('document_type')}, Policy markers: {len(structure_analysis.get('policy_markers', []))}, Hierarchy indicators: {len(structure_analysis.get('hierarchy_indicators', []))}")
            logger.debug(f"Identified policy markers: {structure_analysis.get('policy_markers', [])}")

            return structure_analysis

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in structure analysis: {e}. Falling back to heuristics.")
            return {
                "document_type": "unknown",
                "policy_markers": [r"^\d+\.", r"^Section \d+", r"^Article \d+"],
                "hierarchy_indicators": [r"^\s*[a-z]\.", r"^\s*[ivx]+\.", r"^\s*\([0-9]+\)"],
                "boundary_rules": ["before_next_marker"],
                "context_keywords": ["coverage", "exclusion", "definition", "requirement"]
            }
        except Exception as e:
            logger.warning(f"LLM structure analysis failed: {e}. Falling back to heuristics.")
            return {
                "document_type": "unknown",
                "policy_markers": [r"^\d+\.", r"^Section \d+", r"^Article \d+"],
                "hierarchy_indicators": [r"^\s*[a-z]\.", r"^\s*[ivx]+\.", r"^\s*\([0-9]+\)"],
                "boundary_rules": ["before_next_marker"],
                "context_keywords": ["coverage", "exclusion", "definition", "requirement"]
            }

    async def _identify_policy_boundaries_with_llm(
        self, text: str, structure_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to identify specific policy boundaries in the text.

        Args:
            text: Full document text
            structure_analysis: Structure analysis from _analyze_document_structure_with_llm

        Returns:
            List of policy boundaries with start/end positions
        """
        logger.info("Using LLM to identify policy boundaries...")

        # Split text into lines with positions
        lines = text.split('\n')

        # Find potential boundaries using regex patterns from structure analysis
        boundaries = []
        policy_markers = structure_analysis.get('policy_markers', [])

        position = 0
        for line_num, line in enumerate(lines):
            # Check if line matches any policy marker pattern
            for pattern in policy_markers:
                try:
                    if re.match(pattern, line.strip()):
                        boundaries.append({
                            'line_number': line_num,
                            'position': position,
                            'text': line.strip(),
                            'type': 'policy_start'
                        })
                        break
                except re.error:
                    continue

            position += len(line) + 1  # +1 for newline

        logger.info(f"Identified {len(boundaries)} potential policy boundaries")
        return boundaries

    async def chunk_document(
        self, pages: List[PDFPage], structure: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Chunk document intelligently based on structure and semantic boundaries.
        Uses LLM to identify policy boundaries and prevent splitting policies.

        Args:
            pages: List of PDFPage objects
            structure: Document structure information

        Returns:
            List of DocumentChunk objects
        """
        logger.info(f"Start chunking {len(pages)} pages: target={self.target_chunk_size}, max={self.max_chunk_size}")

        chunks = []

        # Use LLM-assisted chunking if enabled
        if self.use_llm:
            try:
                logger.info("Using LLM-assisted intelligent chunking strategy...")

                # Step 1: Analyze document structure with LLM
                logger.debug("Step 1: Analyzing document structure with LLM...")
                structure_analysis = await self._analyze_document_structure_with_llm(pages)

                # Step 2: Get full document text
                logger.debug("Step 2: Extracting full document text...")
                full_text = "\n\n".join([
                    f"--- Page {p.page_number} ---\n{p.text}"
                    for p in pages
                ])
                logger.debug(f"Full text extracted: {len(full_text)} characters")

                # Step 3: Identify policy boundaries
                logger.debug("Step 3: Identifying policy boundaries...")
                boundaries = await self._identify_policy_boundaries_with_llm(
                    full_text, structure_analysis
                )

                # Step 4: Chunk by policy boundaries
                if boundaries:
                    logger.info(f"Step 4: Creating chunks from {len(boundaries)} policy boundaries...")
                    chunks = self._chunk_by_policy_boundaries(
                        pages, boundaries, structure_analysis
                    )
                    logger.info(f"LLM-assisted chunking complete - Created {len(chunks)} chunks respecting policy boundaries")
                else:
                    logger.warning("No policy boundaries found by LLM, falling back to section-based chunking")
                    chunks = None

            except Exception as e:
                logger.warning(f"LLM-assisted chunking failed: {e}. Falling back to traditional chunking.")
                chunks = None

        # Fall back to traditional chunking if LLM chunking failed or is disabled
        if not chunks:
            logger.info("Using traditional section-based/page-based chunking...")
            logger.debug("Identifying sections from document structure...")
            sections = self._identify_sections(pages, structure)

            if sections:
                logger.info(f"Found {len(sections)} sections - Creating section-based chunks...")
                chunks = self._chunk_by_sections(pages, sections)
                logger.info(f"Section-based chunking complete - Created {len(chunks)} chunks")
            else:
                logger.info("No sections found - Using page-based chunking...")
                chunks = self._chunk_by_pages(pages)
                logger.info(f"Page-based chunking complete - Created {len(chunks)} chunks")

        # Calculate statistics
        total_tokens = sum(c.token_count for c in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        logger.info(f"Chunking complete - Total chunks: {len(chunks)}, Total tokens: {total_tokens:,}, Avg tokens/chunk: {avg_tokens:.0f}")

        return chunks

    def _chunk_by_policy_boundaries(
        self,
        pages: List[PDFPage],
        boundaries: List[Dict[str, Any]],
        structure_analysis: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Create chunks based on LLM-identified policy boundaries with hard limits.
        Prefers policy boundaries but enforces max chunk size.

        Args:
            pages: List of PDFPage objects
            boundaries: List of policy boundaries from LLM
            structure_analysis: Structure analysis from LLM

        Returns:
            List of DocumentChunk objects
        """
        logger.info(f"Chunk by boundaries: {len(boundaries)} found, max_size={self.max_chunk_size}")

        full_text = "\n\n".join([f"--- Page {p.page_number} ---\n{p.text}" for p in pages])

        chunks = []
        chunk_id = 0
        current_chunk_text = ""
        current_start_page = 1
        current_section_title = "Document Start"
        last_boundary_position = 0
        sorted_boundaries = sorted(boundaries, key=lambda x: x['position'])

        for i, boundary in enumerate(sorted_boundaries):
            segment_text = full_text[last_boundary_position:boundary['position']]
            potential_text = current_chunk_text + segment_text
            token_count = self._count_tokens(potential_text)

            # Check against target size
            should_split = token_count >= self.target_chunk_size and current_chunk_text

            # Force split if exceeding max size
            if token_count > self.max_chunk_size:
                logger.warning(f"Chunk {chunk_id} exceeds max ({token_count} > {self.max_chunk_size}), force split")
                should_split = True

            if should_split:
                end_page = self._find_page_number_at_position(full_text, last_boundary_position, pages)
                chunk_tokens = self._count_tokens(current_chunk_text)

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=current_chunk_text,
                    start_page=current_start_page,
                    end_page=end_page,
                    section_context=current_section_title[:80],
                    token_count=chunk_tokens,
                    metadata={
                        "boundary_count": i - chunk_id,
                        "chunking_method": "llm_policy_boundaries"
                    },
                )
                chunks.append(chunk)
                logger.info(f"Created chunk {chunk_id}: pages {current_start_page}-{end_page}, tokens={chunk_tokens}")

                overlap_text = self._get_contextual_overlap(current_chunk_text, structure_analysis)
                current_chunk_text = overlap_text + segment_text
                current_start_page = self._find_page_number_at_position(full_text, boundary['position'], pages)
                current_section_title = boundary.get('text', 'Policy Section')
                chunk_id += 1
            else:
                current_chunk_text = potential_text
                if not current_chunk_text.strip():
                    current_section_title = boundary.get('text', 'Policy Section')

            last_boundary_position = boundary['position']

        # Handle remaining text
        remaining_text = full_text[last_boundary_position:]
        if remaining_text.strip():
            current_chunk_text += remaining_text

        # Force split final chunk if too large
        if current_chunk_text.strip():
            final_tokens = self._count_tokens(current_chunk_text)
            if final_tokens > self.max_chunk_size:
                logger.warning(f"Final chunk too large ({final_tokens}), page-based split")
                page_chunks = self._split_by_pages(current_chunk_text, current_start_page, pages, chunk_id)
                chunks.extend(page_chunks)
            else:
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=current_chunk_text,
                    start_page=current_start_page,
                    end_page=pages[-1].page_number,
                    section_context=current_section_title[:80],
                    token_count=final_tokens,
                    metadata={"boundary_count": len(sorted_boundaries) - chunk_id, "chunking_method": "llm_policy_boundaries"},
                )
                chunks.append(chunk)
                logger.info(f"Created final chunk {chunk_id}: pages {current_start_page}-{pages[-1].page_number}, tokens={final_tokens}")

        logger.info(f"Created {len(chunks)} chunks from {len(boundaries)} boundaries")
        return chunks

    def _split_by_pages(
        self, text: str, start_page: int, all_pages: List[PDFPage], start_chunk_id: int
    ) -> List[DocumentChunk]:
        """Split oversized text by pages to enforce max chunk size."""
        page_markers = list(re.finditer(r'--- Page (\d+) ---', text))
        chunks = []
        chunk_id = start_chunk_id
        current_text = ""
        current_start_idx = 0

        for marker in page_markers:
            page_num = int(marker.group(1))
            segment = text[current_start_idx:marker.start()]
            potential = current_text + segment

            if self._count_tokens(potential) > self.target_chunk_size and current_text:
                chunk_tokens = self._count_tokens(current_text)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=current_text,
                    start_page=start_page,
                    end_page=page_num - 1,
                    section_context=f"Pages {start_page}-{page_num-1}",
                    token_count=chunk_tokens,
                    metadata={"chunking_method": "page_split_forced"}
                ))
                logger.info(f"Force-split chunk {chunk_id}: pages {start_page}-{page_num-1}, {chunk_tokens} tokens")
                current_text = segment
                start_page = page_num
                chunk_id += 1
            else:
                current_text = potential

            current_start_idx = marker.start()

        if current_text.strip():
            chunk_tokens = self._count_tokens(current_text)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=current_text,
                start_page=start_page,
                end_page=all_pages[-1].page_number,
                section_context=f"Pages {start_page}-{all_pages[-1].page_number}",
                token_count=chunk_tokens,
                metadata={"chunking_method": "page_split_forced"}
            ))

        return chunks

    def _find_page_number_at_position(
        self, full_text: str, position: int, pages: List[PDFPage]
    ) -> int:
        """
        Find the page number at a specific character position in the full text.

        Args:
            full_text: Full document text
            position: Character position
            pages: List of PDFPage objects

        Returns:
            Page number
        """
        # Find page markers before this position
        text_before = full_text[:position]
        page_markers = re.findall(r'--- Page (\d+) ---', text_before)

        if page_markers:
            return int(page_markers[-1])

        return pages[0].page_number if pages else 1

    def _get_contextual_overlap(
        self, text: str, structure_analysis: Dict[str, Any]
    ) -> str:
        """
        Get contextual overlap text that includes important context.
        Tries to include complete sentences and key context keywords.

        Args:
            text: Full chunk text
            structure_analysis: Structure analysis from LLM

        Returns:
            Overlap text with context
        """
        # Get last N tokens as base overlap
        tokens = self.encoding.encode(text)

        if len(tokens) <= self.overlap:
            return text

        # Get more tokens than overlap to work with
        extended_overlap_size = int(self.overlap * 1.5)
        overlap_tokens = tokens[-extended_overlap_size:]
        overlap_text = self.encoding.decode(overlap_tokens)

        # Find the last complete sentence or paragraph
        sentences = re.split(r'[.!?]\s+', overlap_text)

        if len(sentences) > 1:
            # Keep the last complete sentences up to overlap size
            result = ""
            for sentence in reversed(sentences[:-1]):  # Exclude last incomplete sentence
                potential = sentence + ". " + result
                if self._count_tokens(potential) <= self.overlap:
                    result = potential
                else:
                    break

            if result.strip():
                return result

        # Fall back to token-based overlap
        final_overlap_tokens = tokens[-self.overlap:]
        return self.encoding.decode(final_overlap_tokens)

    def _identify_sections(
        self, pages: List[PDFPage], structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify major sections in the document.

        Args:
            pages: List of PDFPage objects
            structure: Document structure information

        Returns:
            List of section dictionaries
        """
        sections = []

        if not structure.get("has_numbered_sections"):
            return sections

        # Try to identify section boundaries
        section_pattern = structure.get("section_pattern")

        if section_pattern:
            for page in pages:
                # Find section headers in page text
                matches = re.finditer(section_pattern, page.text, re.MULTILINE)

                for match in matches:
                    # Extract section title
                    line_start = page.text.rfind("\n", 0, match.start()) + 1
                    line_end = page.text.find("\n", match.end())
                    section_title = page.text[line_start:line_end].strip()

                    sections.append({
                        "title": section_title,
                        "page_number": page.page_number,
                        "position": match.start(),
                    })

        logger.info(f"Identified {len(sections)} sections in document")
        return sections

    def _chunk_by_sections(
        self, pages: List[PDFPage], sections: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """
        Create chunks based on identified sections.

        Args:
            pages: List of PDFPage objects
            sections: List of identified sections

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        current_chunk_text = ""
        current_start_page = 1
        current_section = "Introduction"
        chunk_id = 0

        for i, section in enumerate(sections):
            section_page = section["page_number"]

            # Collect text from current position to this section
            section_text = ""
            for page in pages:
                if current_start_page <= page.page_number < section_page:
                    section_text += f"\n\n--- Page {page.page_number} ---\n{page.text}"

            # Add to current chunk
            current_chunk_text += section_text

            # Check if we should create a chunk
            token_count = self._count_tokens(current_chunk_text)

            if token_count >= self.target_chunk_size or i == len(sections) - 1:
                # Create chunk
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=current_chunk_text,
                    start_page=current_start_page,
                    end_page=section_page - 1,
                    section_context=current_section,
                    token_count=token_count,
                    metadata={"section_count": i - chunk_id + 1},
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk_text)
                current_chunk_text = overlap_text
                current_start_page = section_page
                current_section = section["title"]
                chunk_id += 1

        # Handle remaining pages
        remaining_text = ""
        last_section_page = sections[-1]["page_number"] if sections else 1

        for page in pages:
            if page.page_number >= last_section_page:
                remaining_text += f"\n\n--- Page {page.page_number} ---\n{page.text}"

        if remaining_text:
            current_chunk_text += remaining_text
            token_count = self._count_tokens(current_chunk_text)

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=current_chunk_text,
                start_page=current_start_page,
                end_page=pages[-1].page_number,
                section_context=current_section,
                token_count=token_count,
                metadata={"section_count": 1},
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_pages(self, pages: List[PDFPage]) -> List[DocumentChunk]:
        """
        Create chunks based on page boundaries (fallback method).

        Args:
            pages: List of PDFPage objects

        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        current_chunk_text = ""
        current_start_page = 1
        chunk_id = 0

        for page in pages:
            page_text = f"\n\n--- Page {page.page_number} ---\n{page.text}"

            # Add tables if present
            if page.tables:
                for table_idx, table in enumerate(page.tables):
                    page_text += f"\n\n[Table {table_idx + 1}]\n"
                    page_text += self._format_table(table)

            # Check if adding this page would exceed chunk size
            potential_text = current_chunk_text + page_text
            token_count = self._count_tokens(potential_text)

            if token_count >= self.target_chunk_size and current_chunk_text:
                # Create chunk from accumulated text
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=current_chunk_text,
                    start_page=current_start_page,
                    end_page=page.page_number - 1,
                    section_context=f"Pages {current_start_page}-{page.page_number - 1}",
                    token_count=self._count_tokens(current_chunk_text),
                    metadata={"page_count": page.page_number - current_start_page},
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk_text)
                current_chunk_text = overlap_text + page_text
                current_start_page = page.page_number
                chunk_id += 1
            else:
                current_chunk_text = potential_text

        # Create final chunk
        if current_chunk_text:
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=current_chunk_text,
                start_page=current_start_page,
                end_page=pages[-1].page_number,
                section_context=f"Pages {current_start_page}-{pages[-1].page_number}",
                token_count=self._count_tokens(current_chunk_text),
                metadata={"page_count": pages[-1].page_number - current_start_page + 1},
            )
            chunks.append(chunk)

        return chunks

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        return len(self.encoding.encode(text))

    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text from end of chunk.

        Args:
            text: Full chunk text

        Returns:
            Overlap text
        """
        # Get last N tokens as overlap
        tokens = self.encoding.encode(text)

        if len(tokens) <= self.overlap:
            return text

        overlap_tokens = tokens[-self.overlap:]
        overlap_text = self.encoding.decode(overlap_tokens)

        return overlap_text

    def _format_table(self, table: List[List[str]]) -> str:
        """
        Format table as text.

        Args:
            table: Table data (list of rows)

        Returns:
            Formatted table string
        """
        if not table:
            return ""

        # Simple table formatting
        formatted_rows = []
        for row in table:
            formatted_row = " | ".join([str(cell or "") for cell in row])
            formatted_rows.append(formatted_row)

        return "\n".join(formatted_rows)

    def get_chunk_summary(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Get summary statistics about chunks.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Summary dictionary
        """
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0

        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": avg_tokens,
            "max_tokens": max((chunk.token_count for chunk in chunks), default=0),
            "min_tokens": min((chunk.token_count for chunk in chunks), default=0),
        }
