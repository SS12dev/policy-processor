"""
A2A Agent for Policy Document Processing.

This agent provides a single unified endpoint for processing policy documents
with full streaming support and direct orchestrator integration.
"""

import json
import base64
import hashlib
from typing import Dict, Any, Optional, AsyncIterator
from datetime import datetime

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, TextPart, Role
from typing_extensions import override

from app.utils.logger import get_logger
from app.database.operations import DatabaseOperations
from app.core.orchestrator import ProcessingOrchestrator
from app.models.schemas import ProcessingRequest

logger = get_logger(__name__)


class PolicyProcessorAgent(AgentExecutor):
    """
    A2A Agent with a single processing endpoint.

    This agent handles all policy document processing with streaming support.
    """

    def __init__(self, db_ops: DatabaseOperations, orchestrator: ProcessingOrchestrator):
        """
        Initialize the agent.

        Args:
            db_ops: Database operations instance
            orchestrator: Policy processing orchestrator
        """
        self.db_ops = db_ops
        self.orchestrator = orchestrator
        logger.info("PolicyProcessorAgent initialized")

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute an incoming A2A request.

        Args:
            context: Request context containing the user's request
            event_queue: Queue for sending responses back to the client
        """
        try:
            logger.info(f"Processing A2A request: task_id={context.task_id}")

            # Extract parameters from context
            parameters = self._extract_parameters(context)

            # Check if this is a document processing request
            if "document_base64" in parameters:
                await self._process_document(context, event_queue, parameters)
            elif "job_id" in parameters and not parameters.get("document_base64"):
                # Get results for existing job
                await self._get_results(context, event_queue, parameters["job_id"])
            else:
                # Invalid request
                await self._send_error(
                    context,
                    event_queue,
                    "Invalid request. Please provide either 'document_base64' to process a new document or 'job_id' to get results."
                )

        except Exception as e:
            logger.error(f"Error executing A2A request: {e}", exc_info=True)
            await self._send_error(context, event_queue, str(e))

    async def _process_document(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        parameters: Dict[str, Any]
    ) -> None:
        """Process a policy document."""
        try:
            # Extract parameters
            document_base64 = parameters["document_base64"]
            use_gpt4 = parameters.get("use_gpt4", False)
            enable_streaming = parameters.get("enable_streaming", True)
            confidence_threshold = parameters.get("confidence_threshold", 0.7)

            # Validate and decode document
            try:
                pdf_bytes = base64.b64decode(document_base64)
                logger.info(f"Decoded PDF document: {len(pdf_bytes)} bytes")
            except Exception as e:
                await self._send_error(context, event_queue, f"Invalid base64 encoding: {str(e)}")
                return

            # Calculate document hash for deduplication
            doc_hash = hashlib.sha256(pdf_bytes).hexdigest()

            # Create processing request
            request = ProcessingRequest(
                document=document_base64,
                processing_options={
                    "use_gpt4": use_gpt4,
                    "enable_streaming": enable_streaming,
                    "confidence_threshold": confidence_threshold,
                }
            )

            # Send initial acknowledgment
            if enable_streaming:
                await self._send_message(
                    context,
                    event_queue,
                    "📄 Document received. Starting processing...\n"
                )

            # Process the document
            response = await self.orchestrator.process_document(request)

            # Save to database
            job_data = {
                "job_id": response.job_id,
                "status": response.status.value,
                "document_type": response.metadata.document_type.value if hasattr(response.metadata, 'document_type') else "unknown",
                "created_at": datetime.utcnow(),
                "started_at": datetime.utcnow(),
                "use_gpt4": use_gpt4,
                "enable_streaming": enable_streaming,
                "confidence_threshold": confidence_threshold,
            }

            # Add document data
            document_data = {
                "job_id": response.job_id,
                "content_base64": document_base64,
                "document_hash": doc_hash,
                "file_size_bytes": len(pdf_bytes),
                "mime_type": "application/pdf",
                "uploaded_at": datetime.utcnow(),
            }

            # Save job and document
            self.db_ops.save_job(job_data)
            self.db_ops.save_document(document_data)

            # If completed, save results
            if response.status.value == "completed":
                job_data["completed_at"] = datetime.utcnow()
                job_data["status"] = "completed"

                # Extract statistics
                if hasattr(response, 'metadata'):
                    job_data["total_pages"] = getattr(response.metadata, 'total_pages', None)

                if hasattr(response, 'policy_hierarchy'):
                    job_data["total_policies"] = response.policy_hierarchy.total_policies

                if hasattr(response, 'decision_trees'):
                    job_data["total_decision_trees"] = len(response.decision_trees)

                if hasattr(response, 'validation_result'):
                    job_data["validation_confidence"] = response.validation_result.overall_confidence
                    job_data["validation_passed"] = response.validation_result.is_valid

                # Update job
                self.db_ops.update_job(response.job_id, job_data)

                # Save results
                results_data = response.dict()
                self.db_ops.save_results(response.job_id, results_data)

                # Send completion message
                validation_status = "Passed" if hasattr(response, 'validation_result') and response.validation_result.is_valid else "Failed"
                message = (
                    f"Processing complete!\n\n"
                    f"**Job ID:** `{response.job_id}`\n"
                    f"**Status:** {response.status.value}\n"
                    f"**Total Policies:** {response.policy_hierarchy.total_policies if hasattr(response, 'policy_hierarchy') else 'N/A'}\n"
                    f"**Decision Trees:** {len(response.decision_trees) if hasattr(response, 'decision_trees') else 'N/A'}\n"
                    f"**Validation:** {validation_status}\n"
                )
                await self._send_message(context, event_queue, message)

            elif response.status.value == "failed":
                job_data["status"] = "failed"
                job_data["error_message"] = getattr(response, 'error_message', 'Unknown error')
                self.db_ops.update_job(response.job_id, job_data)

                await self._send_error(
                    context,
                    event_queue,
                    f"Processing failed: {getattr(response, 'error_message', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            await self._send_error(context, event_queue, f"Processing error: {str(e)}")

    async def _get_results(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        job_id: str
    ) -> None:
        """Get results for a job."""
        try:
            results = self.db_ops.get_results(job_id)

            if not results:
                await self._send_error(context, event_queue, f"Job {job_id} not found")
                return

            # Format response
            response_text = json.dumps(results, indent=2, default=str)
            await self._send_message(context, event_queue, response_text)

        except Exception as e:
            logger.error(f"Error getting results: {e}", exc_info=True)
            await self._send_error(context, event_queue, str(e))

    async def _send_message(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        text: str
    ) -> None:
        """Send a text message through the event queue."""
        message = Message(
            message_id=f"msg_{context.task_id}_{datetime.utcnow().timestamp()}",
            role=Role.agent,
            parts=[TextPart(text=text)],
            task_id=context.task_id,
            context_id=context.context_id
        )
        event_queue.enqueue_event(message)

    async def _send_error(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        error_message: str
    ) -> None:
        """Send an error message."""
        logger.error(f"Sending error: {error_message}")
        message = Message(
            message_id=f"msg_error_{context.task_id}",
            role=Role.agent,
            parts=[TextPart(text=f"ERROR: {error_message}")],
            task_id=context.task_id,
            context_id=context.context_id
        )
        event_queue.enqueue_event(message)

    @override
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """Handle cancellation of an ongoing task."""
        logger.warning(f"Cancel requested for task: {context.task_id}")

        message = Message(
            message_id=f"msg_cancel_{context.task_id}",
            role=Role.agent,
            parts=[TextPart(text="WARNING: Task cancellation is not currently supported. Jobs will continue to completion.")],
            task_id=context.task_id,
            context_id=context.context_id
        )
        event_queue.enqueue_event(message)

    def _extract_parameters(self, context: RequestContext) -> Dict[str, Any]:
        """Extract parameters from the request context."""
        try:
            # Try to get from metadata
            if hasattr(context, 'metadata') and context.metadata:
                params = context.metadata.get('parameters', {})
                if params:
                    return params

            # Try to parse from user message
            user_input = context.get_user_input()
            if user_input:
                # Try to parse as JSON
                try:
                    return json.loads(user_input)
                except json.JSONDecodeError:
                    pass

            return {}
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
            return {}
