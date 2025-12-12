"""
LangGraph-based orchestrator for Policy Document Processing.

This orchestrator uses LangGraph's StateGraph to manage the processing workflow
as a state machine with nodes for each processing stage.
"""
import uuid
from typing import Optional, Dict, Any, AsyncIterator
from datetime import datetime

from langgraph.graph import StateGraph, END
from core.graph_state import ProcessingState, create_initial_state
from core.graph_nodes import (
    parse_pdf_node,
    analyze_document_node,
    chunk_document_node,
    extract_policies_node,
    generate_trees_node,
    validate_node,
    retry_failed_trees_node,
    verification_node,
    refinement_node,
    complete_node,
    should_retry,
    should_refine,
    check_for_errors,
)
from models.schemas import ProcessingRequest, ProcessingResponse, ProcessingStage
from utils.logger import get_logger

logger = get_logger(__name__)


class LangGraphOrchestrator:
    """
    LangGraph-based orchestrator that processes policy documents through
    a state machine workflow with streaming support.
    """

    def __init__(self):
        """Initialize the LangGraph orchestrator."""
        self.graph = self._build_graph()
        logger.info("LangGraph orchestrator initialized")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph with all nodes and edges.

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(ProcessingState)

        # Add all nodes
        workflow.add_node("parse_pdf", parse_pdf_node)
        workflow.add_node("analyze_document", analyze_document_node)
        workflow.add_node("chunk_document", chunk_document_node)
        workflow.add_node("extract_policies", extract_policies_node)
        workflow.add_node("generate_trees", generate_trees_node)
        workflow.add_node("validate", validate_node)
        workflow.add_node("retry_trees", retry_failed_trees_node)
        workflow.add_node("verification", verification_node)
        workflow.add_node("check_refinement", lambda state: state)  # Pass-through node for routing
        workflow.add_node("refinement", refinement_node)
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

        # Add conditional routing after validation for retry logic OR verification
        workflow.add_conditional_edges(
            "validate",
            should_retry,
            {
                "retry": "retry_trees",
                "complete": "verification"  # Changed: go to verification instead of complete
            }
        )

        # After retry, validate again
        workflow.add_edge("retry_trees", "validate")
        
        # Add verification node
        workflow.add_edge("verification", "check_refinement")
        
        # Add conditional routing for refinement
        workflow.add_conditional_edges(
            "check_refinement",
            should_refine,
            {
                "refine": "refinement",
                "reverify": "verification",  # Re-verify after refinement
                "complete": "complete"
            }
        )
        
        # Add refinement node
        workflow.add_edge("refinement", "check_refinement")

        # Complete node is the end
        workflow.add_edge("complete", END)

        # Compile the graph
        compiled_graph = workflow.compile()
        logger.info("LangGraph workflow compiled successfully with 10 nodes (added verification & refinement)")

        return compiled_graph

    async def stream_processing(
        self,
        request: ProcessingRequest,
        job_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream processing updates as the document flows through the workflow.

        Args:
            request: ProcessingRequest object
            job_id: Optional job ID (generated if not provided)

        Yields:
            Progress updates as dictionaries with status information
        """
        if not job_id:
            job_id = str(uuid.uuid4())

        logger.info(f"[{job_id}] Starting streaming document processing")
        start_time = datetime.utcnow()

        try:
            document_base64 = request.document

            initial_state = create_initial_state(
                job_id=job_id,
                document_base64=document_base64,
                use_gpt4=request.processing_options.use_gpt4,
                enable_streaming=request.processing_options.enable_streaming,
                confidence_threshold=request.processing_options.confidence_threshold,
            )

            logger.info(f"[{job_id}] Initial state created, starting streaming execution")

            # Stream state updates as graph executes
            async for chunk in self.graph.astream(initial_state, stream_mode="updates"):
                # Extract node name and updated state
                for node_name, node_state in chunk.items():
                    current_stage = node_state.get("current_stage")
                    progress = node_state.get("progress_percentage", 0.0)

                    # Yield progress update
                    update = {
                        "job_id": job_id,
                        "node": node_name,
                        "stage": current_stage.value if current_stage else "unknown",
                        "progress": progress,
                        "errors": node_state.get("errors", []),
                        "is_complete": node_name == "complete"
                    }

                    logger.debug(f"[{job_id}] Streaming update: {node_name} at {progress}%")
                    yield update

                    # If failed, stop streaming
                    if node_state.get("is_failed", False):
                        logger.error(f"[{job_id}] Processing failed at node {node_name}")
                        return

            logger.info(f"[{job_id}] Streaming execution complete")

        except Exception as e:
            logger.error(f"[{job_id}] Streaming processing failed: {e}", exc_info=True)
            yield {
                "job_id": job_id,
                "node": "error",
                "stage": ProcessingStage.FAILED.value,
                "progress": 0.0,
                "errors": [str(e)],
                "is_complete": True
            }

    async def process_document(
        self,
        request: ProcessingRequest,
        job_id: Optional[str] = None
    ) -> ProcessingResponse:
        """
        Process a policy document through the LangGraph workflow.

        This method collects all streaming updates and returns the final result.
        For progressive updates, use stream_processing() instead.

        Args:
            request: ProcessingRequest object
            job_id: Optional job ID (generated if not provided)

        Returns:
            ProcessingResponse object
        """
        if not job_id:
            job_id = str(uuid.uuid4())

        logger.info(f"[{job_id}] Starting document processing")
        start_time = datetime.utcnow()

        try:
            document_base64 = request.document

            initial_state = create_initial_state(
                job_id=job_id,
                document_base64=document_base64,
                use_gpt4=request.processing_options.use_gpt4,
                enable_streaming=request.processing_options.enable_streaming,
                confidence_threshold=request.processing_options.confidence_threshold,
            )

            logger.info(f"[{job_id}] Initial state created, starting graph execution")

            # Execute graph and collect final state
            final_state = None
            async for chunk in self.graph.astream(initial_state, stream_mode="updates"):
                # Keep updating with latest state
                for node_name, node_state in chunk.items():
                    final_state = node_state
                    logger.debug(f"[{job_id}] Completed node: {node_name}")

            if not final_state:
                raise Exception("Graph execution produced no final state")

            logger.info(f"[{job_id}] Graph execution complete")

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Check if processing failed
            if final_state.get("is_failed", False):
                errors = final_state.get("errors", ["Unknown error"])
                logger.error(f"[{job_id}] Processing failed: {errors}")
                raise Exception(f"Processing failed: {errors[0]}")

            # Update processing time in metadata (metadata is a dict)
            if final_state.get("metadata"):
                final_state["metadata"]["processing_time_seconds"] = processing_time

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
                f"[{job_id}] Processing complete in {processing_time:.2f}s - "
                f"{len(final_state.get('decision_trees', []))} trees generated"
            )

            return response

        except Exception as e:
            logger.error(f"[{job_id}] Processing failed: {e}", exc_info=True)
            raise

    async def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current processing status from state.

        Note: Status tracking is now handled through streaming updates.
        This method is kept for backward compatibility.

        Args:
            job_id: Job identifier

        Returns:
            Status dict or None
        """
        logger.warning(f"[{job_id}] Status lookup called - use stream_processing() for real-time updates")
        return None

    async def get_result(self, job_id: str) -> Optional[ProcessingResponse]:
        """
        Get processing result from storage.

        Note: Results are now returned directly from process_document().
        This method is kept for backward compatibility.

        Args:
            job_id: Job identifier

        Returns:
            ProcessingResponse object or None
        """
        logger.warning(f"[{job_id}] Result lookup called - results are returned from process_document()")
        return None
