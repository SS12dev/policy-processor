"""
Data models and schemas for policy document processing.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from datetime import datetime


# ===== PDF Processing Models =====

class ImageMetadata(BaseModel):
    """Metadata for an extracted image."""
    image_id: str = Field(..., description="Unique image identifier (hash)")
    image_index: int = Field(..., description="Image index on page")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    format: str = Field(..., description="Image format (PNG, JPEG, etc.)")
    thumbnail_base64: Optional[str] = Field(default=None, description="Low-res thumbnail (max 200x200)")
    position: Dict[str, float] = Field(default_factory=dict, description="Position on page (x0, y0, x1, y1)")
    size_bytes: int = Field(..., description="Original image size in bytes")


class TableMetadata(BaseModel):
    """Metadata for an extracted table."""
    table_index: int = Field(..., description="Table index on page")
    row_count: int = Field(..., description="Number of rows")
    col_count: int = Field(..., description="Number of columns")
    preview_rows: List[List[str]] = Field(default_factory=list, description="First 3 rows for preview")
    position: Dict[str, float] = Field(default_factory=dict, description="Position on page")
    confidence: float = Field(default=0.8, description="Table detection confidence")


class HeadingInfo(BaseModel):
    """Information about a heading or section title."""
    text: str = Field(..., description="Heading text")
    level: int = Field(..., description="Estimated heading level (1-6)")
    page_number: int = Field(..., description="Page where heading appears")
    char_start: int = Field(..., description="Start character position in page text")
    char_end: int = Field(..., description="End character position in page text")
    font_size: Optional[float] = Field(default=None, description="Font size if available")
    is_bold: bool = Field(default=False, description="Whether text is bold")
    is_all_caps: bool = Field(default=False, description="Whether text is all caps")


class TOCEntry(BaseModel):
    """Table of Contents entry."""
    title: str = Field(..., description="Section title")
    page_number: int = Field(..., description="Page number from TOC")
    level: int = Field(default=1, description="Heading level")
    actual_page_number: Optional[int] = Field(default=None, description="Validated actual page")


class SectionBoundary(BaseModel):
    """Boundary information for a document section."""
    section_id: str = Field(..., description="Section identifier")
    title: str = Field(..., description="Section title")
    start_page: int = Field(..., description="Starting page number")
    end_page: int = Field(..., description="Ending page number")
    heading_info: Optional[HeadingInfo] = Field(default=None, description="Associated heading")


class PageLayoutInfo(BaseModel):
    """Layout information for a page."""
    width: float = Field(..., description="Page width in points")
    height: float = Field(..., description="Page height in points")
    orientation: str = Field(..., description="portrait or landscape")
    column_count: int = Field(default=1, description="Number of text columns")
    has_headers: bool = Field(default=False, description="Has header region")
    has_footers: bool = Field(default=False, description="Has footer region")
    margin_top: float = Field(default=0.0, description="Top margin estimate")
    margin_bottom: float = Field(default=0.0, description="Bottom margin estimate")
    text_density: float = Field(default=0.0, description="Text density (chars per sq inch)")


class EnhancedPDFPage(BaseModel):
    """Enhanced PDF page with comprehensive metadata."""
    page_id: str = Field(..., description="Unique page identifier (doc_hash:pageN)")
    page_number: int = Field(..., description="Page number (1-indexed)")
    
    # Text content
    text: str = Field(..., description="Extracted text content")
    text_preview: str = Field(..., description="First 250 chars for preview")
    char_count: int = Field(..., description="Total character count")
    approx_tokens: int = Field(..., description="Approximate token count")
    
    # Images
    images_count: int = Field(default=0, description="Number of images on page")
    images: List[ImageMetadata] = Field(default_factory=list, description="Image metadata")
    
    # Tables
    tables_count: int = Field(default=0, description="Number of tables on page")
    tables: List[TableMetadata] = Field(default_factory=list, description="Table metadata")
    
    # Layout and structure
    layout_info: PageLayoutInfo = Field(..., description="Layout information")
    headings: List[HeadingInfo] = Field(default_factory=list, description="Headings found on page")
    
    # OCR information
    is_scanned: bool = Field(default=False, description="Whether page is scanned/image-based")
    ocr_performed: bool = Field(default=False, description="Whether OCR was run")
    ocr_confidence: Optional[float] = Field(default=None, description="OCR confidence score (0-1)")
    ocr_language: Optional[str] = Field(default=None, description="OCR language used")
    
    # Importance scoring
    importance_score: float = Field(default=0.5, description="Page importance score (0-1)")
    has_policy_content: bool = Field(default=True, description="Contains actual policy content")
    
    # Error tracking
    processing_errors: List[str] = Field(default_factory=list, description="Errors during processing")
    warnings: List[str] = Field(default_factory=list, description="Warnings during processing")


class EnhancedPDFMetadata(BaseModel):
    """Comprehensive PDF document metadata."""
    document_hash: str = Field(..., description="SHA256 hash of document")
    total_pages: int = Field(..., description="Total page count")
    
    # Content flags
    has_images: bool = Field(default=False, description="Document contains images")
    has_tables: bool = Field(default=False, description="Document contains tables")
    has_toc: bool = Field(default=False, description="Has table of contents")
    is_encrypted: bool = Field(default=False, description="Document is/was encrypted")
    
    # PDF metadata
    pdf_info: Dict[str, str] = Field(default_factory=dict, description="PDF metadata (title, author, etc.)")
    pdf_version: Optional[str] = Field(default=None, description="PDF version")
    
    # Structure
    headings: List[HeadingInfo] = Field(default_factory=list, description="All document headings")
    toc_entries: List[TOCEntry] = Field(default_factory=list, description="Table of contents entries")
    section_boundaries: List[SectionBoundary] = Field(default_factory=list, description="Section boundaries")
    
    # Statistics
    total_images: int = Field(default=0, description="Total images across document")
    total_tables: int = Field(default=0, description="Total tables across document")
    scanned_pages_count: int = Field(default=0, description="Number of scanned pages")
    page_token_estimates: List[int] = Field(default_factory=list, description="Token count per page")
    
    # Quality metrics
    ocr_quality_score: float = Field(default=1.0, description="Overall OCR quality (0-1)")
    extraction_quality: float = Field(default=1.0, description="Text extraction quality (0-1)")
    
    # Error tracking
    page_errors: Dict[int, str] = Field(default_factory=dict, description="Errors by page number")
    processing_warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    
    # Processing stats
    extraction_methods_used: List[str] = Field(default_factory=list, description="Extraction methods used")
    processing_time_seconds: float = Field(..., description="Total processing time")
    ocr_time_seconds: float = Field(default=0.0, description="Time spent on OCR")


# ===== Document Analysis Models =====

class PageContentType(str, Enum):
    """Types of content found on a page."""
    POLICY_CONTENT = "policy_content"           # Main policy text
    DEFINITIONS = "definitions"                 # Definitions section
    EXCLUSIONS = "exclusions"                   # Exclusions/limitations
    COVERAGE = "coverage"                       # Coverage details
    ELIGIBILITY = "eligibility"                 # Eligibility criteria
    PROCEDURES = "procedures"                   # Procedures/processes
    TABLE_OF_CONTENTS = "table_of_contents"     # TOC
    REFERENCES = "references"                   # References section
    BIBLIOGRAPHY = "bibliography"               # Bibliography
    INDEX = "index"                            # Index
    APPENDIX = "appendix"                      # Appendix
    ADMINISTRATIVE = "administrative"           # Admin content (cover, disclaimers)
    MIXED = "mixed"                            # Multiple content types
    UNKNOWN = "unknown"                        # Cannot determine


class PolicyBoundary(BaseModel):
    """Information about policy boundaries within a page."""
    page_number: int = Field(..., description="Page number")
    policy_starts_here: bool = Field(default=False, description="New policy starts on this page")
    policy_continues_from_previous: bool = Field(default=False, description="Policy continues from previous page")
    policy_ends_here: bool = Field(default=False, description="Policy ends on this page")
    policy_continues_to_next: bool = Field(default=False, description="Policy continues to next page")
    split_point_char: Optional[int] = Field(default=None, description="Character offset where policy splits")
    policy_ids: List[str] = Field(default_factory=list, description="Policy IDs present on this page")
    heading_at_start: Optional[str] = Field(default=None, description="Heading text if policy starts here")


class PageAnalysis(BaseModel):
    """Detailed analysis of a single page."""
    page_number: int = Field(..., description="Page number")
    
    # Content classification
    primary_content_type: PageContentType = Field(..., description="Primary content type")
    secondary_content_types: List[PageContentType] = Field(default_factory=list, description="Other content types")
    
    # Content quality scores
    policy_content_ratio: float = Field(default=0.0, description="Ratio of policy content (0-1)")
    administrative_content_ratio: float = Field(default=0.0, description="Ratio of admin content (0-1)")
    filterable_content_ratio: float = Field(default=0.0, description="Ratio of content to filter (0-1)")
    
    # Policy boundaries
    policy_boundary: PolicyBoundary = Field(..., description="Policy boundary information")
    
    # Continuity markers
    starts_mid_sentence: bool = Field(default=False, description="Starts with incomplete sentence")
    ends_mid_sentence: bool = Field(default=False, description="Ends with incomplete sentence")
    has_list_continuation: bool = Field(default=False, description="Has continuing list items")
    
    # Structural elements
    has_headings: bool = Field(default=False, description="Page contains headings")
    has_numbered_items: bool = Field(default=False, description="Has numbered items/lists")
    has_bullet_points: bool = Field(default=False, description="Has bullet points")
    has_tables: bool = Field(default=False, description="Contains tables")
    has_references: bool = Field(default=False, description="Contains references")
    
    # Content characteristics
    is_dense_text: bool = Field(default=False, description="Dense paragraph text")
    is_list_heavy: bool = Field(default=False, description="Mostly lists")
    is_table_heavy: bool = Field(default=False, description="Mostly tables")
    
    # Filtering hints
    should_include_in_extraction: bool = Field(default=True, description="Include in policy extraction")
    requires_special_handling: bool = Field(default=False, description="Needs special processing")
    chunk_priority: float = Field(default=0.5, description="Priority for chunking (0-1)")


class ContentZone(BaseModel):
    """A zone of similar content within the document."""
    zone_id: str = Field(..., description="Unique zone identifier")
    zone_type: PageContentType = Field(..., description="Type of content in this zone")
    start_page: int = Field(..., description="Starting page number")
    end_page: int = Field(..., description="Ending page number")
    page_count: int = Field(..., description="Number of pages in zone")
    
    # Zone characteristics
    heading_text: Optional[str] = Field(default=None, description="Zone heading/title")
    has_subsections: bool = Field(default=False, description="Contains subsections")
    nesting_level: int = Field(default=0, description="Nesting depth (0=top-level)")
    
    # Content quality
    average_importance: float = Field(default=0.5, description="Average page importance in zone")
    should_extract_policies: bool = Field(default=True, description="Extract policies from this zone")
    
    # Related zones
    parent_zone_id: Optional[str] = Field(default=None, description="Parent zone if nested")
    child_zone_ids: List[str] = Field(default_factory=list, description="Child zones")


class PolicyFlowNode(BaseModel):
    """A node in the policy flow graph."""
    policy_id: str = Field(..., description="Policy identifier")
    policy_title: str = Field(..., description="Policy title")
    
    # Location
    start_page: int = Field(..., description="Starting page")
    end_page: int = Field(..., description="Ending page")
    spans_multiple_pages: bool = Field(default=False, description="Spans multiple pages")
    
    # Continuation tracking
    continuation_points: List[int] = Field(default_factory=list, description="Pages where policy continues")
    has_interruptions: bool = Field(default=False, description="Has interruptions (tables, etc.)")
    
    # Content characteristics
    has_tables: bool = Field(default=False, description="Contains tables")
    has_lists: bool = Field(default=False, description="Contains lists")
    has_definitions: bool = Field(default=False, description="Contains definitions")
    has_exclusions: bool = Field(default=False, description="Contains exclusions")
    
    # Hierarchy
    nesting_level: int = Field(default=0, description="Nesting level (0=root)")
    parent_policy_id: Optional[str] = Field(default=None, description="Parent policy ID")
    child_policy_ids: List[str] = Field(default_factory=list, description="Child policy IDs")
    
    # Quality
    estimated_confidence: float = Field(default=0.7, description="Estimated extraction confidence")


class ReferenceInfo(BaseModel):
    """Information about a reference within the document."""
    reference_type: str = Field(..., description="internal|external|citation")
    source_page: int = Field(..., description="Page where reference appears")
    reference_text: str = Field(..., description="Reference text")
    target_page: Optional[int] = Field(default=None, description="Target page if internal")
    target_section: Optional[str] = Field(default=None, description="Target section")


class EnhancedDocumentMetadata(BaseModel):
    """Enhanced document metadata with comprehensive analysis."""
    
    # Basic document info (from previous DocumentMetadata)
    document_type: 'DocumentType' = Field(..., description="Type of policy document")
    total_pages: int = Field(..., description="Total number of pages")
    complexity_score: float = Field(..., description="Document complexity (0-1)")
    has_images: bool = Field(default=False, description="Whether document contains images")
    has_tables: bool = Field(default=False, description="Whether document contains tables")
    structure_type: str = Field(..., description="Document structure (numbered, hierarchical, etc.)")
    language: str = Field(default="en", description="Document language")
    processing_time_seconds: float = Field(..., description="Total processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Enhanced analysis
    page_analyses: List[PageAnalysis] = Field(default_factory=list, description="Per-page analysis")
    content_zones: List[ContentZone] = Field(default_factory=list, description="Content zones")
    policy_flow_map: List[PolicyFlowNode] = Field(default_factory=list, description="Policy flow graph")
    policy_boundaries: List[PolicyBoundary] = Field(default_factory=list, description="Detected policy boundaries")
    document_structure: Optional[Dict[str, Any]] = Field(default=None, description="Enhanced policy document structure analysis")
    
    # Content statistics
    policy_pages_count: int = Field(default=0, description="Pages with policy content")
    admin_pages_count: int = Field(default=0, description="Administrative pages")
    mixed_pages_count: int = Field(default=0, description="Mixed content pages")
    
    # Zone statistics
    main_policy_zone: Optional[ContentZone] = Field(default=None, description="Main policy content zone")
    definitions_zone: Optional[ContentZone] = Field(default=None, description="Definitions zone")
    exclusions_zone: Optional[ContentZone] = Field(default=None, description="Exclusions zone")
    references_zone: Optional[ContentZone] = Field(default=None, description="References zone")
    
    # Reference tracking
    references: List[ReferenceInfo] = Field(default_factory=list, description="Document references")
    has_internal_references: bool = Field(default=False, description="Has internal cross-references")
    has_external_references: bool = Field(default=False, description="Has external references")
    
    # Chunking guidance
    recommended_chunk_boundaries: List[Tuple[int, int]] = Field(default_factory=list, description="Recommended (start, end) page pairs")
    pages_to_filter: List[int] = Field(default_factory=list, description="Pages to filter/exclude")
    pages_requiring_special_handling: List[int] = Field(default_factory=list, description="Pages needing special processing")
    
    # Quality indicators
    overall_extractability_score: float = Field(default=0.7, description="How extractable is this document (0-1)")
    estimated_policy_count: int = Field(default=0, description="Estimated number of policies")
    confidence_in_structure: float = Field(default=0.7, description="Confidence in structure analysis (0-1)")


class DocumentType(str, Enum):
    """Types of policy documents."""
    INSURANCE = "insurance"
    LEGAL = "legal"
    REGULATORY = "regulatory"
    CORPORATE = "corporate"
    HEALTHCARE = "healthcare"
    FINANCIAL = "financial"
    UNKNOWN = "unknown"


class ProcessingStage(str, Enum):
    """Stages of document processing."""
    SUBMITTED = "submitted"
    PARSING_PDF = "parsing_pdf"
    ANALYZING_DOCUMENT = "analyzing_document"
    CHUNKING = "chunking"
    EXTRACTING_POLICIES = "extracting_policies"
    GENERATING_TREES = "generating_trees"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestionType(str, Enum):
    """Types of eligibility questions."""
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC_RANGE = "numeric_range"
    TEXT_INPUT = "text_input"
    DATE = "date"
    CONDITIONAL = "conditional"


class NodeType(str, Enum):
    """Types of decision tree nodes."""
    QUESTION = "question"
    DECISION = "decision"
    OUTCOME = "outcome"
    ROUTER = "router"
    GROUP = "group"


class OutcomeType(str, Enum):
    """Types of outcome nodes."""
    APPROVED = "approved"
    DENIED = "denied"
    REFER_TO_MANUAL = "refer_to_manual"
    PENDING_REVIEW = "pending_review"
    REQUIRES_DOCUMENTATION = "requires_documentation"


class LogicGroupType(str, Enum):
    """Types of logic grouping."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    XOR = "XOR"


class ComparisonOperator(str, Enum):
    """Comparison operators for routing conditions."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    IN_RANGE = "in_range"
    CONTAINS = "contains"
    MATCHES_PATTERN = "matches_pattern"


class ProcessingOptions(BaseModel):
    """Options for document processing."""
    use_gpt4: bool = Field(default=False, description="Use GPT-4 for complex sections")
    enable_streaming: bool = Field(default=True, description="Enable real-time progress streaming")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence score")
    max_depth: int = Field(default=10, description="Maximum hierarchy depth to extract")


class ProcessingRequest(BaseModel):
    """Request to process a policy document."""
    document: str = Field(..., description="Base64-encoded PDF document")
    processing_options: Optional[ProcessingOptions] = Field(default_factory=ProcessingOptions)


class SourceReference(BaseModel):
    """Reference to source material in the document."""
    page_number: int = Field(..., description="Page number in the document")
    section: str = Field(..., description="Section identifier or title")
    quoted_text: str = Field(..., description="Exact text from the source")
    line_numbers: Optional[List[int]] = Field(default=None, description="Line numbers if available")


class DocumentMetadata(BaseModel):
    """Metadata about the processed document."""
    document_type: DocumentType = Field(..., description="Type of policy document")
    total_pages: int = Field(..., description="Total number of pages")
    complexity_score: float = Field(..., description="Document complexity (0-1)")
    has_images: bool = Field(default=False, description="Whether document contains images")
    has_tables: bool = Field(default=False, description="Whether document contains tables")
    structure_type: str = Field(..., description="Document structure (numbered, hierarchical, etc.)")
    language: str = Field(default="en", description="Document language")
    processing_time_seconds: float = Field(..., description="Total processing time")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PolicyCondition(BaseModel):
    """A condition within a policy."""
    condition_id: str = Field(..., description="Unique identifier for the condition")
    description: str = Field(..., description="Human-readable description")
    logic_type: str = Field(..., description="AND, OR, NOT, etc.")
    source_references: List[SourceReference] = Field(default_factory=list)
    confidence_score: float = Field(..., description="Confidence in extraction (0-1)")


class SubPolicy(BaseModel):
    """A sub-policy within a larger policy."""
    policy_id: str = Field(..., description="Unique identifier")
    title: str = Field(..., description="Policy title or heading")
    description: str = Field(..., description="Policy description")
    level: int = Field(..., description="Hierarchy level (0 is root)")
    conditions: List[PolicyCondition] = Field(default_factory=list)
    source_references: List[SourceReference] = Field(default_factory=list)
    parent_id: Optional[str] = Field(default=None, description="Parent policy ID")
    children: List['SubPolicy'] = Field(default_factory=list, description="Child policies")
    confidence_score: float = Field(..., description="Confidence in extraction (0-1)")


class PolicyHierarchy(BaseModel):
    """Complete hierarchical structure of all policies."""
    root_policies: List[SubPolicy] = Field(..., description="Top-level policies")
    total_policies: int = Field(..., description="Total number of policies extracted")
    max_depth: int = Field(..., description="Maximum depth of hierarchy")
    definitions: Dict[str, str] = Field(default_factory=dict, description="Key terms and definitions")


class QuestionOption(BaseModel):
    """An option for a multiple choice question."""
    option_id: str = Field(..., description="Unique option identifier")
    label: str = Field(..., description="Option label")
    value: Any = Field(..., description="Option value")
    leads_to_node: Optional[str] = Field(default=None, description="Next node ID if selected")
    routes_to_tree: Optional[str] = Field(default=None, description="Routes to child policy tree ID")
    routes_to_policy: Optional[str] = Field(default=None, description="Routes to child policy title")


class RoutingRule(BaseModel):
    """Defines how to route based on answer value."""
    answer_value: Union[str, int, float, bool] = Field(..., description="Answer value to match")
    comparison: ComparisonOperator = Field(default=ComparisonOperator.EQUALS, description="How to compare")
    next_node_id: str = Field(..., description="Next node to route to")
    condition_expression: Optional[str] = Field(default=None, description="Complex condition expression")
    range_min: Optional[Union[int, float]] = Field(default=None, description="Minimum for range comparison")
    range_max: Optional[Union[int, float]] = Field(default=None, description="Maximum for range comparison")


class LogicGroup(BaseModel):
    """Defines logical grouping of nodes."""
    group_id: str = Field(..., description="Unique group identifier")
    group_type: LogicGroupType = Field(..., description="Type of logical grouping")
    member_node_ids: List[str] = Field(default_factory=list, description="Node IDs in this group")
    parent_group_id: Optional[str] = Field(default=None, description="Parent group for nested logic")
    description: str = Field(default="", description="Human-readable group description")


class EligibilityQuestion(BaseModel):
    """A question in the decision tree."""
    question_id: str = Field(..., description="Unique question identifier")
    question_text: str = Field(..., description="The question to ask")
    question_type: QuestionType = Field(..., description="Type of question")
    options: Optional[List[QuestionOption]] = Field(default=None, description="Options for multiple choice")
    validation_rules: Optional[Dict[str, Any]] = Field(default=None, description="Validation rules for input")
    help_text: Optional[str] = Field(default=None, description="Additional context or help")
    source_references: List[SourceReference] = Field(default_factory=list)
    explanation: str = Field(..., description="Why this question is being asked")
    routing_rules: Optional[List[RoutingRule]] = Field(default=None, description="Explicit routing rules for this question")


class DecisionNode(BaseModel):
    """A node in the decision tree with conditional routing support."""
    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="question, decision, outcome, router, group")

    # Node content based on type
    question: Optional[EligibilityQuestion] = Field(default=None, description="Question if this is a question node")
    decision_logic: Optional[str] = Field(default=None, description="Logic expression for decision nodes")
    outcome: Optional[str] = Field(default=None, description="Outcome if this is an outcome node")
    outcome_type: Optional[str] = Field(default=None, description="approved, denied, refer_to_manual, etc.")

    # Routing configuration
    children: Dict[str, 'DecisionNode'] = Field(default_factory=dict, description="Child nodes keyed by answer")
    routing_rules: Optional[List[RoutingRule]] = Field(default=None, description="Explicit routing configuration")
    default_next_node_id: Optional[str] = Field(default=None, description="Default next node if no routing rule matches")

    # Grouping and logic
    logic_group: Optional[LogicGroup] = Field(default=None, description="Logic group this node belongs to")
    group_type: Optional[LogicGroupType] = Field(default=None, description="If this node is a group container")

    # Metadata
    source_references: List[SourceReference] = Field(default_factory=list)
    confidence_score: float = Field(..., description="Confidence in this node (0-1)")

    # Hierarchical context fields
    policy_context: Optional[Dict[str, Any]] = Field(default=None, description="Parent policy context")
    child_policy_references: List[str] = Field(default_factory=list, description="Child policies this routes to")
    parent_policy_id: Optional[str] = Field(default=None, description="Parent policy ID if this is a child")
    navigation_hint: Optional[str] = Field(default=None, description="Context hint for user")
    display_order: Optional[int] = Field(default=None, description="Order for display in UI")


class DecisionTree(BaseModel):
    """Complete decision tree for a policy with conditional routing."""
    tree_id: str = Field(..., description="Unique tree identifier")
    policy_id: str = Field(..., description="Associated policy ID")
    policy_title: str = Field(..., description="Policy title")
    root_node: DecisionNode = Field(..., description="Root node of the tree")
    questions: List[EligibilityQuestion] = Field(default_factory=list, description="Flat list of all questions in the tree")
    logic_groups: List[LogicGroup] = Field(default_factory=list, description="Logic groups for AND/OR conditions")

    # Tree statistics
    total_nodes: int = Field(..., description="Total number of nodes")
    total_paths: int = Field(..., description="Total number of possible paths")
    total_outcomes: int = Field(default=0, description="Number of outcome nodes")
    max_depth: int = Field(..., description="Maximum depth of the tree")
    confidence_score: float = Field(..., description="Overall confidence in tree (0-1)")

    # Path validation
    has_complete_routing: bool = Field(default=False, description="Whether all paths lead to outcomes")
    unreachable_nodes: List[str] = Field(default_factory=list, description="Node IDs that are unreachable")
    incomplete_routes: List[str] = Field(default_factory=list, description="Node IDs with incomplete routing")

    # Hierarchical metadata
    policy_level: int = Field(default=0, description="Hierarchy level (0=root)")
    parent_policy_id: Optional[str] = Field(default=None, description="Parent policy ID")
    child_policy_ids: List[str] = Field(default_factory=list, description="Child policy IDs")
    is_aggregator: bool = Field(default=False, description="Whether this orchestrates child trees")
    aggregation_strategy: Optional[str] = Field(default=None, description="How children are aggregated")


class ValidationIssue(BaseModel):
    """An issue found during validation."""
    severity: str = Field(..., description="error, warning, info")
    issue_type: str = Field(..., description="Type of issue")
    description: str = Field(..., description="Description of the issue")
    location: str = Field(..., description="Where the issue was found")
    suggestion: Optional[str] = Field(default=None, description="Suggested fix")


class ValidationResult(BaseModel):
    """Results of validation process."""
    is_valid: bool = Field(..., description="Whether validation passed")
    overall_confidence: float = Field(..., description="Overall confidence score (0-1)")
    issues: List[ValidationIssue] = Field(default_factory=list)
    completeness_score: float = Field(..., description="How complete the extraction is (0-1)")
    consistency_score: float = Field(..., description="Internal consistency score (0-1)")
    traceability_score: float = Field(..., description="How well traced to source (0-1)")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")
    sections_requiring_gpt4: List[str] = Field(default_factory=list, description="Sections needing GPT-4")


class ProcessingStatus(BaseModel):
    """Current status of document processing."""
    job_id: str = Field(..., description="Job identifier")
    stage: ProcessingStage = Field(..., description="Current processing stage")
    progress_percentage: float = Field(..., description="Progress (0-100)")
    current_section: Optional[str] = Field(default=None, description="Section being processed")
    message: str = Field(..., description="Status message")
    errors: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ProcessingResponse(BaseModel):
    """Complete response from document processing."""
    job_id: str = Field(..., description="Job identifier")
    status: ProcessingStage = Field(..., description="Final processing status")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    policy_hierarchy: PolicyHierarchy = Field(..., description="Extracted policy hierarchy")
    decision_trees: List[DecisionTree] = Field(..., description="Generated decision trees")
    validation_result: ValidationResult = Field(..., description="Validation results")
    processing_stats: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ===== Enhanced Chunking Models =====

class PolicyChunkMetadata(BaseModel):
    """Metadata for a policy-aware chunk."""
    chunk_id: int = Field(..., description="Chunk identifier")
    start_page: int = Field(..., description="Starting page number")
    end_page: int = Field(..., description="Ending page number")
    policy_ids: List[str] = Field(default_factory=list, description="Policy IDs in chunk")
    content_zones: List[str] = Field(default_factory=list, description="Content zone IDs")
    token_count: int = Field(..., description="Token count")
    has_definitions: bool = Field(default=False, description="Contains definitions")
    has_complete_context: bool = Field(default=False, description="Has complete context")
    continuity_preserved: bool = Field(default=True, description="Semantic continuity preserved")
    heading: Optional[str] = Field(default=None, description="Primary heading/title")
    is_multi_page: bool = Field(default=False, description="Spans multiple pages")
    boundary_type: str = Field(default="policy", description="Type of boundary (policy/zone/page)")
    zone_id: Optional[str] = Field(default=None, description="Associated zone ID if zone_split")


class DuplicatePolicyCandidate(BaseModel):
    """Candidate duplicate policy detected across chunks."""
    policy_id_1: str = Field(..., description="First policy ID")
    policy_id_2: str = Field(..., description="Second policy ID")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    chunk_id_1: int = Field(..., description="First chunk ID")
    chunk_id_2: int = Field(..., description="Second chunk ID")
    merge_recommendation: str = Field(..., description="Merge recommendation")


class ContextValidationResult(BaseModel):
    """Results of context completeness validation."""
    total_chunks: int = Field(..., description="Total chunks validated")
    complete_chunks: int = Field(..., description="Chunks with complete context")
    incomplete_chunks: int = Field(..., description="Chunks missing context")
    definition_coverage_pct: float = Field(..., description="Percentage with definitions")
    incomplete_chunk_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Details of incomplete chunks"
    )


class ChunkingStatistics(BaseModel):
    """Statistics about the chunking process."""
    total_chunks: int = Field(..., description="Total chunks created")
    total_tokens: int = Field(..., description="Total token count")
    avg_tokens_per_chunk: float = Field(..., description="Average tokens per chunk")
    max_tokens: int = Field(..., description="Maximum chunk size")
    min_tokens: int = Field(..., description="Minimum chunk size")
    filtered_pages_count: int = Field(..., description="Number of pages filtered")
    policy_pages_count: int = Field(..., description="Number of policy pages processed")
    unique_policies_count: int = Field(..., description="Number of unique policies")
    duplicate_candidates_count: int = Field(..., description="Potential duplicates found")
    chunks_with_definitions: int = Field(..., description="Chunks containing definitions")
    chunks_with_complete_context: int = Field(..., description="Chunks with complete context")
    boundary_type_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of boundary types"
    )


class EnhancedChunkingResult(BaseModel):
    """Complete result of enhanced chunking process."""
    chunks: List[PolicyChunkMetadata] = Field(..., description="Created chunks")
    filtered_pages: List[int] = Field(default_factory=list, description="Filtered page numbers")
    duplicate_candidates: List[DuplicatePolicyCandidate] = Field(
        default_factory=list,
        description="Potential duplicate policies"
    )
    context_validation: ContextValidationResult = Field(..., description="Context validation results")
    statistics: ChunkingStatistics = Field(..., description="Chunking statistics")
    chunking_method: str = Field(
        default="enhanced_policy_aware",
        description="Chunking method used"
    )


# Update forward references
SubPolicy.model_rebuild()
DecisionNode.model_rebuild()
