"""
LangGraph State Definition for Policy Document Processing.

Defines the state that flows through the processing graph.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add
from models.schemas import (
    ProcessingStage,
    DocumentMetadata,
    DocumentMetadata,
    PolicyHierarchy,
    DecisionTree,
    ValidationResult,
)


class ProcessingState(TypedDict):
    """
    State that flows through the LangGraph workflow.

    This state is passed between nodes and updated as processing progresses.
    """

    # Job identification
    job_id: str

    # Input
    document_base64: str

    # Processing options
    use_gpt4: bool
    enable_streaming: bool
    confidence_threshold: float

    # Current stage and progress
    current_stage: ProcessingStage
    progress_percentage: float
    status_message: str

    # Stage 1: PDF Parsing
    pdf_bytes: Optional[bytes]
    pages: Optional[List[Any]]  # List[PDFPage]
    pdf_metadata: Optional[Dict[str, Any]]  # PDFMetadata as dict
    structure: Optional[Dict[str, Any]]

    # Stage 2: Document Analysis
    metadata: Optional[DocumentMetadata]  # Basic metadata for backward compatibility
    document_metadata: Optional[Dict[str, Any]]  # DocumentMetadata as dict
    should_use_gpt4_extraction: Optional[bool]
    should_use_gpt4_trees: bool  # Always True

    # Stage 3: Chunking
    chunks: Optional[List[Any]]  # List[Chunk]
    chunk_summary: Optional[Dict[str, Any]]
    chunking_metadata: Optional[Dict[str, Any]]  # Chunking metadata (filtered pages, duplicates, validation)

    # Stage 4: Policy Extraction
    policy_hierarchy: Optional[PolicyHierarchy]

    # Stage 5: Decision Tree Generation
    decision_trees: Optional[List[DecisionTree]]

    # Stage 6: Validation
    validation_result: Optional[ValidationResult]
    validation_passed: bool
    retry_count: int
    needs_retry: bool
    failed_policy_ids: Optional[List[str]]

    # Stage 7: Final Results
    processing_stats: Optional[Dict[str, Any]]
    processing_time_seconds: Optional[float]

    # Error handling
    errors: Annotated[List[str], add]  # Accumulate errors
    is_failed: bool

    # Logging and debugging
    logs: Annotated[List[str], add]  # Accumulate logs


def create_initial_state(
    job_id: str,
    document_base64: str,
    use_gpt4: bool = False,
    enable_streaming: bool = True,
    confidence_threshold: float = 0.7
) -> ProcessingState:
    """
    Create initial state for processing.

    Args:
        job_id: Unique job identifier
        document_base64: Base64-encoded PDF document
        use_gpt4: Whether to use GPT-4 for extraction
        enable_streaming: Enable streaming updates
        confidence_threshold: Minimum confidence threshold

    Returns:
        Initial ProcessingState
    """
    return ProcessingState(
        # Job identification
        job_id=job_id,

        # Input
        document_base64=document_base64,

        # Processing options
        use_gpt4=use_gpt4,
        enable_streaming=enable_streaming,
        confidence_threshold=confidence_threshold,

        # Current stage
        current_stage=ProcessingStage.PARSING_PDF,
        progress_percentage=0.0,
        status_message="Initializing...",

        # Stage outputs (all None initially)
        pdf_bytes=None,
        pages=None,
        pdf_metadata=None,
        structure=None,
        metadata=None,
        document_metadata=None,
        should_use_gpt4_extraction=None,
        should_use_gpt4_trees=True,  # Always use GPT-4 for trees
        chunks=None,
        chunk_summary=None,
        policy_hierarchy=None,
        decision_trees=None,
        validation_result=None,
        validation_passed=False,
        retry_count=0,
        needs_retry=False,
        failed_policy_ids=None,
        processing_stats=None,
        processing_time_seconds=None,

        # Error handling
        errors=[],
        is_failed=False,

        # Logging
        logs=[]
    )
