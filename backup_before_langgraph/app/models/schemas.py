"""
Data models and schemas for policy document processing.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime


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


class DecisionNode(BaseModel):
    """A node in the decision tree."""
    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="question, decision, outcome")
    question: Optional[EligibilityQuestion] = Field(default=None, description="Question if this is a question node")
    outcome: Optional[str] = Field(default=None, description="Outcome if this is an outcome node")
    outcome_type: Optional[str] = Field(default=None, description="approved, denied, refer_to_manual")
    children: Dict[str, 'DecisionNode'] = Field(default_factory=dict, description="Child nodes keyed by answer")
    source_references: List[SourceReference] = Field(default_factory=list)
    confidence_score: float = Field(..., description="Confidence in this node (0-1)")

    # Hierarchical context fields
    policy_context: Optional[Dict[str, Any]] = Field(default=None, description="Parent policy context")
    child_policy_references: List[str] = Field(default_factory=list, description="Child policies this routes to")
    parent_policy_id: Optional[str] = Field(default=None, description="Parent policy ID if this is a child")
    navigation_hint: Optional[str] = Field(default=None, description="Context hint for user")


class DecisionTree(BaseModel):
    """Complete decision tree for a policy."""
    tree_id: str = Field(..., description="Unique tree identifier")
    policy_id: str = Field(..., description="Associated policy ID")
    policy_title: str = Field(..., description="Policy title")
    root_node: DecisionNode = Field(..., description="Root node of the tree")
    total_nodes: int = Field(..., description="Total number of nodes")
    total_paths: int = Field(..., description="Total number of possible paths")
    max_depth: int = Field(..., description="Maximum depth of the tree")
    confidence_score: float = Field(..., description="Overall confidence in tree (0-1)")

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


# Update forward references
SubPolicy.model_rebuild()
DecisionNode.model_rebuild()
