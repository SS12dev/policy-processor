"""
Document Analysis Agent
LangGraph-based agent for processing PDF documents.
Uses StateGraph with multiple nodes and conditional edges.
"""

from typing import TypedDict, Annotated, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from settings import settings
from utils.logger import get_logger
from nodes.pdf_parser import validate_pdf_node, parse_pdf_node
from nodes.heading_extractor import extract_headings_node
from nodes.keyword_extractor import extract_keywords_node
from nodes.llm_analyzer import analyze_document_node
from nodes.response_formatter import format_response_node

logger = get_logger(__name__)


# Define the state schema
class DocumentAnalysisState(TypedDict):
    """State schema for document analysis workflow."""
    # Input
    pdf_bytes: str  # Base64 encoded PDF
    context_id: str  # Conversation context

    # Validation
    validation_success: bool
    validation_error: str | None
    pdf_size_mb: float

    # PDF Parsing
    extracted_text: str
    page_count: int
    word_count: int
    pdf_metadata: Dict[str, Any]
    parsing_success: bool
    parsing_error: str | None

    # Heading Extraction
    headings: list[Dict[str, Any]]
    heading_count: int
    heading_extraction_success: bool
    heading_extraction_error: str | None

    # Keyword Extraction
    keywords: list[str]
    keywords_basic: list[str]
    keywords_llm: list[str]
    keyword_count: int
    keyword_extraction_success: bool
    keyword_extraction_error: str | None

    # LLM Analysis
    analysis: Dict[str, Any] | None
    analysis_success: bool
    analysis_error: str | None

    # Response
    formatted_response: Dict[str, Any]
    text_summary: str
    response_ready: bool

    # Retry tracking
    retry_count: int
    max_retries_reached: bool


# Conditional edge functions
def should_retry_parsing(state: DocumentAnalysisState) -> str:
    """Decide if PDF parsing should be retried."""
    if state.get("parsing_success", False):
        return "continue"

    retry_count = state.get("retry_count", 0)
    if retry_count < settings.max_retries:
        logger.info(f"Retrying PDF parsing (attempt {retry_count + 1}/{settings.max_retries})")
        return "retry"

    logger.error("Max retries reached for PDF parsing")
    return "failed"


def should_continue_after_extraction(state: DocumentAnalysisState) -> str:
    """
    Decide if processing should continue after extraction.
    Continue if at least one extraction method succeeded.
    """
    heading_success = state.get("heading_extraction_success", False)
    keyword_success = state.get("keyword_extraction_success", False)

    if heading_success or keyword_success:
        return "continue"

    logger.warning("Both heading and keyword extraction failed")
    return "skip_analysis"


def should_run_llm_analysis(state: DocumentAnalysisState) -> str:
    """Decide if LLM analysis should run."""
    if not settings.enable_llm_analysis:
        logger.info("LLM analysis disabled in settings")
        return "skip"

    # Only run if we have some extracted data
    has_data = (
        state.get("extracted_text") and
        (state.get("headings") or state.get("keywords"))
    )

    if has_data:
        return "analyze"

    logger.warning("Insufficient data for LLM analysis")
    return "skip"


class DocumentAnalysisAgent:
    """LangGraph agent for document analysis."""

    def __init__(self):
        """Initialize the agent with its workflow graph."""
        logger.info("Initializing Document Analysis Agent")

        # Create memory saver for state persistence
        self.memory = MemorySaver()

        # Build the workflow graph
        self.graph = self._build_graph()

        logger.info("Document Analysis Agent initialized successfully")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with nodes and edges."""
        # Create the state graph
        workflow = StateGraph(DocumentAnalysisState)

        # Add nodes
        workflow.add_node("validate_pdf", validate_pdf_node)
        workflow.add_node("parse_pdf", parse_pdf_node)
        workflow.add_node("extract_headings", extract_headings_node)
        workflow.add_node("extract_keywords", extract_keywords_node)
        workflow.add_node("analyze_document", analyze_document_node)
        workflow.add_node("format_response", format_response_node)

        # Define the workflow edges
        # Start -> Validate
        workflow.add_edge(START, "validate_pdf")

        # Validate -> Parse (with conditional retry)
        workflow.add_conditional_edges(
            "validate_pdf",
            lambda state: "parse" if state.get("validation_success") else "error",
            {
                "parse": "parse_pdf",
                "error": "format_response"  # Skip to response with error
            }
        )

        # Parse -> Retry or Continue
        workflow.add_conditional_edges(
            "parse_pdf",
            should_retry_parsing,
            {
                "continue": "extract_headings",
                "retry": "parse_pdf",  # Retry parsing
                "failed": "format_response"  # Skip to response with error
            }
        )

        # Headings and Keywords run in parallel conceptually,
        # but LangGraph executes sequentially - we go through both
        workflow.add_edge("extract_headings", "extract_keywords")

        # After extraction -> Check if we should continue
        workflow.add_conditional_edges(
            "extract_keywords",
            should_run_llm_analysis,
            {
                "analyze": "analyze_document",
                "skip": "format_response"
            }
        )

        # Analysis -> Format
        workflow.add_edge("analyze_document", "format_response")

        # Format -> End
        workflow.add_edge("format_response", END)

        # Compile the graph
        return workflow.compile(checkpointer=self.memory)

    async def process_document(
        self,
        pdf_bytes: str,
        context_id: str
    ) -> Dict[str, Any]:
        """
        Process a PDF document through the workflow.

        Args:
            pdf_bytes: Base64 encoded PDF data
            context_id: Conversation context ID

        Returns:
            Processing results
        """
        logger.info(
            "Starting document processing",
            extra={"context_id": context_id}
        )

        # Create initial state
        initial_state = {
            "pdf_bytes": pdf_bytes,
            "context_id": context_id,
            "retry_count": 0,
            "max_retries_reached": False
        }

        # Configure the run
        config = {
            "configurable": {
                "thread_id": context_id  # Use context_id as thread_id for state persistence
            }
        }

        try:
            # Execute the workflow
            result = await self.graph.ainvoke(initial_state, config)

            logger.info(
                "Document processing completed",
                extra={
                    "context_id": context_id,
                    "response_ready": result.get("response_ready", False)
                }
            )

            return result

        except Exception as e:
            logger.error(
                f"Document processing failed: {str(e)}",
                extra={
                    "context_id": context_id,
                    "error_type": type(e).__name__
                }
            )
            raise

    async def stream_document_processing(
        self,
        pdf_bytes: str,
        context_id: str
    ):
        """
        Process document with streaming updates.

        Args:
            pdf_bytes: Base64 encoded PDF data
            context_id: Conversation context ID

        Yields:
            Processing updates as they occur
        """
        logger.info(
            "Starting streaming document processing",
            extra={"context_id": context_id}
        )

        initial_state = {
            "pdf_bytes": pdf_bytes,
            "context_id": context_id,
            "retry_count": 0,
            "max_retries_reached": False
        }

        config = {
            "configurable": {
                "thread_id": context_id
            }
        }

        try:
            # Stream the workflow execution
            async for event in self.graph.astream(initial_state, config):
                # Each event contains node name and state update
                yield event

        except Exception as e:
            logger.error(
                f"Streaming document processing failed: {str(e)}",
                extra={
                    "context_id": context_id,
                    "error_type": type(e).__name__
                }
            )
            raise


# Global agent instance
_agent: DocumentAnalysisAgent | None = None


def get_agent() -> DocumentAnalysisAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = DocumentAnalysisAgent()
    return _agent
