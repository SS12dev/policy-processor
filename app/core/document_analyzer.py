"""
Document intelligence analyzer for understanding document types and characteristics.
"""
import re
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from config.settings import settings
from app.utils.logger import get_logger
from app.core.pdf_processor import PDFPage
from app.models.schemas import DocumentType, DocumentMetadata
from datetime import datetime

logger = get_logger(__name__)


class DocumentAnalyzer:
    """Analyzes documents to understand their type, structure, and complexity."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize document analyzer.
        
        Args:
            llm: Optional pre-configured LLM client. If not provided, will be initialized on first use.
        """
        self.llm = llm

    async def analyze_document(
        self, pages: List[PDFPage], structure: Dict[str, Any], llm: Optional[ChatOpenAI] = None
    ) -> DocumentMetadata:
        """
        Analyze document to determine type, complexity, and characteristics.

        Args:
            pages: List of PDFPage objects
            structure: Document structure information
            llm: Optional LLM client to use for analysis

        Returns:
            DocumentMetadata object
        """
        logger.info("Analyzing document characteristics...")

        # Use provided LLM or fall back to instance LLM
        if llm is not None:
            self.llm = llm

        start_time = datetime.utcnow()

        # Get sample text from first few pages
        sample_text = self._get_sample_text(pages, max_pages=3)

        # Determine document type
        doc_type = await self._determine_document_type(sample_text)

        # Calculate complexity score
        complexity = self._calculate_complexity(pages, structure)

        # Check for images and tables
        has_images = any(len(page.images) > 0 for page in pages)
        has_tables = any(len(page.tables) > 0 for page in pages)

        # Determine structure type
        structure_type = self._determine_structure_type(structure)

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        metadata = DocumentMetadata(
            document_type=doc_type,
            total_pages=len(pages),
            complexity_score=complexity,
            has_images=has_images,
            has_tables=has_tables,
            structure_type=structure_type,
            language="en",
            processing_time_seconds=processing_time,
        )

        logger.info(
            f"Document analysis complete: type={doc_type}, complexity={complexity:.2f}, "
            f"structure={structure_type}"
        )

        return metadata

    def _get_sample_text(self, pages: List[PDFPage], max_pages: int = 3) -> str:
        """
        Get sample text from document.

        Args:
            pages: List of PDFPage objects
            max_pages: Maximum number of pages to sample

        Returns:
            Sample text
        """
        sample_pages = pages[:max_pages]
        sample_text = "\n\n".join([p.text for p in sample_pages])
        return sample_text[:8000]  # Limit to 8000 chars

    async def _determine_document_type(self, sample_text: str) -> DocumentType:
        """
        Determine the type of policy document using LLM.

        Args:
            sample_text: Sample text from document

        Returns:
            DocumentType enum
        """
        try:
            prompt = f"""Analyze the following text from a policy document and determine its type.

Text sample:
{sample_text}

Classify the document into ONE of these categories:
- insurance: Insurance policies, coverage documents, claims procedures
- legal: Legal agreements, contracts, terms of service
- regulatory: Government regulations, compliance documents, rules
- corporate: Corporate policies, employee handbooks, internal procedures
- healthcare: Healthcare policies, medical procedures, patient rights
- financial: Financial policies, investment guidelines, banking procedures
- unknown: Cannot determine type

Respond with ONLY the category name (e.g., "insurance" or "legal"), nothing else."""

            if self.llm is None:
                logger.warning("LLM not initialized, using UNKNOWN document type")
                return DocumentType.UNKNOWN

            response = await self.llm.ainvoke(prompt)
            doc_type_str = response.content.strip().lower()

            # Map to DocumentType enum
            type_mapping = {
                "insurance": DocumentType.INSURANCE,
                "legal": DocumentType.LEGAL,
                "regulatory": DocumentType.REGULATORY,
                "corporate": DocumentType.CORPORATE,
                "healthcare": DocumentType.HEALTHCARE,
                "financial": DocumentType.FINANCIAL,
            }

            return type_mapping.get(doc_type_str, DocumentType.UNKNOWN)

        except Exception as e:
            logger.error(f"Error determining document type: {e}")
            return DocumentType.UNKNOWN

    def _calculate_complexity(self, pages: List[PDFPage], structure: Dict[str, Any]) -> float:
        """
        Calculate document complexity score (0-1).

        Args:
            pages: List of PDFPage objects
            structure: Document structure information

        Returns:
            Complexity score between 0 and 1
        """
        complexity_factors = []

        # Factor 1: Page count (more pages = more complex)
        page_count_score = min(len(pages) / 50.0, 1.0)
        complexity_factors.append(page_count_score)

        # Factor 2: Average text length per page
        avg_text_length = sum(len(p.text) for p in pages) / len(pages) if pages else 0
        text_length_score = min(avg_text_length / 5000.0, 1.0)
        complexity_factors.append(text_length_score)

        # Factor 3: Presence of tables and images
        tables_score = 0.3 if any(len(p.tables) > 0 for p in pages) else 0
        images_score = 0.2 if any(len(p.images) > 0 for p in pages) else 0
        complexity_factors.append(tables_score + images_score)

        # Factor 4: Hierarchical structure complexity
        structure_score = 0.0
        if structure.get("has_hierarchy"):
            structure_score += 0.3
        if structure.get("has_numbered_sections"):
            structure_score += 0.2
        if structure.get("has_appendices"):
            structure_score += 0.1
        complexity_factors.append(min(structure_score, 1.0))

        # Factor 5: Vocabulary complexity (unique words, technical terms)
        vocab_score = self._calculate_vocabulary_complexity(pages)
        complexity_factors.append(vocab_score)

        # Calculate weighted average
        weights = [0.15, 0.25, 0.15, 0.25, 0.20]
        complexity = sum(f * w for f, w in zip(complexity_factors, weights))

        return min(max(complexity, 0.0), 1.0)

    def _calculate_vocabulary_complexity(self, pages: List[PDFPage]) -> float:
        """
        Calculate vocabulary complexity based on unique words and term frequency.

        Args:
            pages: List of PDFPage objects

        Returns:
            Vocabulary complexity score (0-1)
        """
        # Sample text from random pages
        sample_size = min(5, len(pages))
        sample_text = " ".join([pages[i].text for i in range(0, len(pages), len(pages) // sample_size)])

        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', sample_text.lower())

        if not words:
            return 0.0

        # Calculate unique word ratio
        unique_ratio = len(set(words)) / len(words) if words else 0

        # Check for technical/legal terms
        technical_patterns = [
            r'\b(pursuant|thereof|herein|aforementioned|notwithstanding)\b',
            r'\b(shall|must|required|prohibited|obligated)\b',
            r'\b(premium|deductible|copay|liability|indemnity)\b',
            r'\b(compliance|regulatory|statutory|provision)\b',
        ]

        technical_count = sum(
            len(re.findall(pattern, sample_text, re.IGNORECASE))
            for pattern in technical_patterns
        )

        technical_ratio = min(technical_count / 100.0, 1.0)

        # Combine factors
        vocab_complexity = (unique_ratio * 0.6) + (technical_ratio * 0.4)

        return min(max(vocab_complexity, 0.0), 1.0)

    def _determine_structure_type(self, structure: Dict[str, Any]) -> str:
        """
        Determine the structure type description.

        Args:
            structure: Document structure information

        Returns:
            Structure type description
        """
        structure_types = []

        if structure.get("has_numbered_sections"):
            structure_types.append("numbered")

        if structure.get("has_hierarchy"):
            structure_types.append("hierarchical")

        if structure.get("has_toc"):
            structure_types.append("with-toc")

        if structure.get("has_appendices"):
            structure_types.append("with-appendices")

        if not structure_types:
            structure_types.append("unstructured")

        return ", ".join(structure_types)

    def should_use_gpt4(self, metadata: DocumentMetadata, use_gpt4_flag: bool) -> bool:
        """
        Determine if GPT-4 should be used based on complexity.

        Args:
            metadata: Document metadata
            use_gpt4_flag: User preference for GPT-4

        Returns:
            True if GPT-4 should be used
        """
        # Use GPT-4 if explicitly requested or if complexity is high
        if use_gpt4_flag:
            return True

        # Use GPT-4 for complex documents
        if metadata.complexity_score > 0.7:
            logger.info(f"High complexity ({metadata.complexity_score:.2f}), recommending GPT-4")
            return True

        return False

    def identify_administrative_sections(self, pages: List[PDFPage]) -> List[int]:
        """
        Identify pages that contain administrative content (TOC, index, bibliography).

        Args:
            pages: List of PDFPage objects

        Returns:
            List of page numbers to potentially skip
        """
        admin_pages = []

        for page in pages:
            text_lower = page.text.lower()

            # Check for common administrative section indicators
            admin_indicators = [
                "table of contents",
                "bibliography",
                "references",
                "index",
                "appendix",
                "glossary",
                "acknowledgments",
                "preface",
            ]

            # If page is mostly these indicators, mark as administrative
            for indicator in admin_indicators:
                if indicator in text_lower and len(page.text.strip()) < 1000:
                    admin_pages.append(page.page_number)
                    break

        logger.info(f"Identified {len(admin_pages)} administrative pages")
        return admin_pages
