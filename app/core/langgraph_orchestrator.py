"""
LangGraph-based orchestrator for Policy Document Processing.

This orchestrator uses LangGraph's StateGraph to manage the processing workflow
as a state machine with nodes for each processing stage.
"""
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

from langgraph.graph import StateGraph, END
from app.core.graph_state import ProcessingState, create_initial_state
from app.core.graph_nodes import (
    parse_pdf_node,
    analyze_document_node,
    chunk_document_node,
    extract_policies_node,
    generate_trees_node,
    validate_node,
    retry_failed_trees_node,
    complete_node,
    should_retry,
    check_for_errors,
)
from app.models.schemas import ProcessingRequest, ProcessingResponse, ProcessingStage
from app.utils.logger import get_logger
from app.utils.redis_client import get_redis_client

logger = get_logger(__name__)


class LangGraphOrchestrator:
    """
    LangGraph-based orchestrator that processes policy documents through
    a state machine workflow.
    """

    def __init__(self):
        """Initialize the LangGraph orchestrator."""
        self.redis_client = get_redis_client()
        self.graph = self._build_graph()
        logger.info("LangGraph orchestrator initialized")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph with all nodes and edges.

        Returns:
            Compiled StateGraph
        """
        # Create state graph
        workflow = StateGraph(ProcessingState)

        # Add all nodes
        workflow.add_node("parse_pdf", parse_pdf_node)
        workflow.add_node("analyze_document", analyze_document_node)
        workflow.add_node("chunk_document", chunk_document_node)
        workflow.add_node("extract_policies", extract_policies_node)
        workflow.add_node("generate_trees", generate_trees_node)
        workflow.add_node("validate", validate_node)
        workflow.add_node("retry_trees", retry_failed_trees_node)
        workflow.add_node("complete", complete_node)

        # Set entry point
        workflow.set_entry_point("parse_pdf")

        # Add sequential edges with error checking
        workflow.add_conditional_edges(
            "parse_pdf",
            check_for_errors,
            {
                "continue": "analyze_document",
                "error": "complete"
            }
        )

        workflow.add_conditional_edges(
            "analyze_document",
            check_for_errors,
            {
                "continue": "chunk_document",
                "error": "complete"
            }
        )

        workflow.add_conditional_edges(
            "chunk_document",
            check_for_errors,
            {
                "continue": "extract_policies",
                "error": "complete"
            }
        )

        workflow.add_conditional_edges(
            "extract_policies",
            check_for_errors,
            {
                "continue": "generate_trees",
                "error": "complete"
            }
        )

        workflow.add_conditional_edges(
            "generate_trees",
            check_for_errors,
            {
                "continue": "validate",
                "error": "complete"
            }
        )

        # Add conditional routing after validation for retry logic
        workflow.add_conditional_edges(
            "validate",
            should_retry,
            {
                "retry": "retry_trees",
                "complete": "complete"
            }
        )

        # After retry, validate again
        workflow.add_edge("retry_trees", "validate")

        # Complete node is the end
        workflow.add_edge("complete", END)

        # Compile the graph
        compiled_graph = workflow.compile()
        logger.info("LangGraph workflow compiled successfully")

        return compiled_graph

    async def process_document(
        self,
        request: ProcessingRequest,
        job_id: Optional[str] = None
    ) -> ProcessingResponse:
        """
        Process a policy document through the LangGraph workflow.

        Args:
            request: ProcessingRequest object
            job_id: Optional job ID (generated if not provided)

        Returns:
            ProcessingResponse object
        """
        # Generate job ID if not provided
        if not job_id:
            job_id = str(uuid.uuid4())

        logger.info(f"[{job_id}] Starting LangGraph document processing")
        start_time = datetime.utcnow()

        try:
            # The document is already base64-encoded string (per ProcessingRequest schema)
            # No need to encode again
            document_base64 = request.document

            # Create initial state
            initial_state = create_initial_state(
                job_id=job_id,
                document_base64=document_base64,
                use_gpt4=request.processing_options.use_gpt4,
                enable_streaming=request.processing_options.enable_streaming,
                confidence_threshold=request.processing_options.confidence_threshold,
            )

            logger.info(f"[{job_id}] Initial state created, starting graph execution")

            # Execute the graph
            final_state = await self.graph.ainvoke(initial_state)

            logger.info(f"[{job_id}] Graph execution complete")

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Check if processing failed
            if final_state.get("is_failed", False):
                logger.error(f"[{job_id}] Processing failed: {final_state.get('errors', [])}")

                # Publish failure status
                await self._publish_failure_status(
                    job_id,
                    final_state.get("errors", ["Unknown error"])
                )

                raise Exception(f"Processing failed: {final_state.get('errors', ['Unknown error'])[0]}")

            # Update processing time in metadata
            if final_state.get("metadata"):
                final_state["metadata"].processing_time_seconds = processing_time

            # Update processing stats
            if final_state.get("processing_stats"):
                final_state["processing_stats"]["processing_time_seconds"] = processing_time

            # Build response
            response = ProcessingResponse(
                job_id=job_id,
                status=final_state["current_stage"],
                metadata=final_state.get("metadata"),
                policy_hierarchy=final_state.get("policy_hierarchy"),
                decision_trees=final_state.get("decision_trees", []),
                validation_result=final_state.get("validation_result"),
                processing_stats=final_state.get("processing_stats", {}),
            )

            logger.info(
                f"[{job_id}] LangGraph processing complete in {processing_time:.2f}s - "
                f"{len(final_state.get('decision_trees', []))} trees generated"
            )

            return response

        except Exception as e:
            logger.error(f"[{job_id}] LangGraph processing failed: {e}", exc_info=True)

            # Publish failure status
            await self._publish_failure_status(job_id, [str(e)])

            raise

    async def _publish_failure_status(self, job_id: str, errors: list):
        """Publish failure status to Redis."""
        try:
            status_data = {
                "job_id": job_id,
                "stage": ProcessingStage.FAILED.value,
                "progress_percentage": 0.0,
                "message": f"Processing failed: {errors[0] if errors else 'Unknown error'}",
                "errors": errors,
            }
            self.redis_client.set_status(job_id, status_data)
            self.redis_client.publish(f"job:{job_id}:status", status_data)
        except Exception as e:
            logger.error(f"Failed to publish failure status: {e}")

    async def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current processing status.

        Args:
            job_id: Job identifier

        Returns:
            Status dict or None
        """
        status_data = self.redis_client.get_status(job_id)
        return status_data

    async def get_result(self, job_id: str) -> Optional[ProcessingResponse]:
        """
        Get processing result.

        Args:
            job_id: Job identifier

        Returns:
            ProcessingResponse object or None
        """
        result_data = self.redis_client.get_result(job_id)

        if result_data:
            return ProcessingResponse(**result_data)

        return None
