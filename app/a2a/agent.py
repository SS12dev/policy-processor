"""
A2A Agent for Policy Document Processing - Stateless Redis-based version.

This agent:
- Uses Redis for temporary storage (no database)
- Returns full results in A2A response
- Supports concurrent requests
- Designed for containerized deployment
"""

import json
import base64
import hashlib
from typing import Dict, Any
from datetime import datetime

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, TextPart, Role
from typing_extensions import override

from app.utils.logger import get_logger
from app.a2a.redis_storage import RedisAgentStorage
from app.core.langgraph_orchestrator import LangGraphOrchestrator
from app.models.schemas import ProcessingRequest

logger = get_logger(__name__)


class PolicyProcessorAgent(AgentExecutor):
    """
    Stateless A2A Agent with Redis-based temporary storage.

    Designed for:
    - Container deployment
    - Horizontal scaling
    - Concurrent request handling
    - No persistent database at agent layer
    """

    def __init__(self, orchestrator: LangGraphOrchestrator, redis_storage: RedisAgentStorage):
        """
        Initialize the agent.

        Args:
            orchestrator: LangGraph-based policy processing orchestrator
            redis_storage: Redis storage for temporary state
        """
        self.orchestrator = orchestrator
        self.storage = redis_storage
        logger.info("PolicyProcessorAgent initialized (stateless, Redis-based)")

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
            logger.info(f"[AGENT] Processing A2A request - Task: {context.task_id}")

            # Extract parameters from context
            parameters = self._extract_parameters(context)
            logger.info(f"[AGENT] Parameters: {self._sanitize_params_for_log(parameters)}")

            # Route to appropriate handler
            if "document_base64" in parameters:
                await self._process_document(context, event_queue, parameters)
            elif "job_id" in parameters and not parameters.get("document_base64"):
                await self._get_results(context, event_queue, parameters["job_id"])
            else:
                await self._send_error(
                    context,
                    event_queue,
                    "Invalid request. Provide 'document_base64' or 'job_id'."
                )

        except Exception as e:
            logger.error(f"Error executing request: {e}", exc_info=True)
            await self._send_error(context, event_queue, str(e))

    async def _process_document(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        parameters: Dict[str, Any]
    ) -> None:
        """Process a policy document with Redis-based state management."""
        job_id = context.task_id  # Use task_id as job_id

        try:
            # Acquire lock to prevent duplicate processing
            if not self.storage.acquire_lock(job_id, timeout_seconds=600):
                await self._send_error(
                    context,
                    event_queue,
                    f"Job {job_id} is already being processed"
                )
                return

            # Increment active jobs counter
            active_count = self.storage.increment_active_jobs()
            logger.info(f"[AGENT] Active jobs: {active_count}")

            try:
                # Extract and validate parameters
                document_base64 = parameters["document_base64"]
                use_gpt4 = parameters.get("use_gpt4", False)
                enable_streaming = parameters.get("enable_streaming", True)
                confidence_threshold = parameters.get("confidence_threshold", 0.7)

                logger.info(f"[AGENT] Job {job_id}: Starting processing")
                logger.info(f"[AGENT] Options: GPT-4={use_gpt4}, Streaming={enable_streaming}, Threshold={confidence_threshold}")

                # Validate document
                try:
                    pdf_bytes = base64.b64decode(document_base64)
                    doc_hash = hashlib.sha256(pdf_bytes).hexdigest()
                    logger.info(f"[AGENT] Document: {len(pdf_bytes)} bytes, hash: {doc_hash[:16]}...")
                except Exception as e:
                    await self._send_error(context, event_queue, f"Invalid base64: {str(e)}")
                    return

                # Save initial status to Redis
                self.storage.save_job_status(job_id, {
                    "status": "processing",
                    "started_at": datetime.utcnow().isoformat(),
                    "document_hash": doc_hash,
                    "use_gpt4": use_gpt4
                }, ttl_hours=1)

                # Create processing request
                request = ProcessingRequest(
                    document=document_base64,
                    processing_options={
                        "use_gpt4": use_gpt4,
                        "enable_streaming": enable_streaming,
                        "confidence_threshold": confidence_threshold,
                    }
                )

                # Process document
                logger.info(f"[AGENT] Job {job_id}: Calling orchestrator")
                response = await self.orchestrator.process_document(request, job_id=job_id)

                # Prepare result data
                result_data = response.model_dump(mode='json')
                result_data["document_hash"] = doc_hash
                result_data["processing_completed_at"] = datetime.utcnow().isoformat()

                # Save result to Redis (with TTL)
                self.storage.save_job_result(job_id, result_data)

                # Update status
                self.storage.save_job_status(job_id, {
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "total_policies": result_data.get("policy_hierarchy", {}).get("total_policies", 0),
                    "validation_passed": result_data.get("validation_result", {}).get("is_valid", False)
                }, ttl_hours=24)

                # Save metrics
                self.storage.save_processing_metrics({
                    "job_id": job_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "duration_seconds": result_data.get("processing_stats", {}).get("processing_time_seconds", 0),
                    "total_policies": result_data.get("policy_hierarchy", {}).get("total_policies", 0),
                    "use_gpt4": use_gpt4
                })

                # Send result in A2A response
                validation_status = "Passed" if response.validation_result.is_valid else "Failed"
                message_text = (
                    f"Processing complete!\n\n"
                    f"**Job ID:** `{job_id}`\n"
                    f"**Status:** completed\n"
                    f"**Total Policies:** {response.policy_hierarchy.total_policies}\n"
                    f"**Decision Trees:** {len(response.decision_trees)}\n"
                    f"**Validation:** {validation_status}\n\n"
                    f"**Full Results (JSON):**\n```json\n{json.dumps(result_data, indent=2, default=str)}\n```"
                )

                await self._send_message(context, event_queue, message_text)

                logger.info(f"[AGENT] Job {job_id}: Completed successfully")

            finally:
                # Release lock and decrement counter
                self.storage.release_lock(job_id)
                active_count = self.storage.decrement_active_jobs()
                logger.info(f"[AGENT] Active jobs: {active_count}")

        except Exception as e:
            logger.error(f"[AGENT] Job {job_id}: Processing error: {e}", exc_info=True)

            # Update status to failed
            self.storage.save_job_status(job_id, {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow().isoformat()
            }, ttl_hours=24)

            # Release lock
            self.storage.release_lock(job_id)
            self.storage.decrement_active_jobs()

            await self._send_error(context, event_queue, f"Processing error: {str(e)}")

    async def _get_results(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        job_id: str
    ) -> None:
        """Get results for a job from Redis."""
        try:
            logger.info(f"[AGENT] Retrieving results for job {job_id}")

            # Get result from Redis
            result_data = self.storage.get_job_result(job_id)

            if not result_data:
                # Check if job exists
                status = self.storage.get_job_status(job_id)
                if status:
                    status_msg = status.get("status", "unknown")
                    await self._send_message(
                        context,
                        event_queue,
                        f"Job {job_id} status: {status_msg}. Results not yet available or expired."
                    )
                else:
                    await self._send_error(context, event_queue, f"Job {job_id} not found")
                return

            # Format and send results
            response_text = json.dumps(result_data, indent=2, default=str)
            await self._send_message(context, event_queue, f"Results for job {job_id}:\n\n```json\n{response_text}\n```")

            logger.info(f"[AGENT] Results sent for job {job_id}")

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
        await event_queue.enqueue_event(message)

    async def _send_error(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        error_message: str
    ) -> None:
        """Send an error message."""
        logger.error(f"[AGENT] Error: {error_message}")
        message = Message(
            message_id=f"msg_error_{context.task_id}",
            role=Role.agent,
            parts=[TextPart(text=f"ERROR: {error_message}")],
            task_id=context.task_id,
            context_id=context.context_id
        )
        await event_queue.enqueue_event(message)

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
            parts=[TextPart(text="WARNING: Task cancellation not supported. Job will continue.")],
            task_id=context.task_id,
            context_id=context.context_id
        )
        await event_queue.enqueue_event(message)

    def _extract_parameters(self, context: RequestContext) -> Dict[str, Any]:
        """Extract parameters from the request context."""
        try:
            # Try metadata first
            if hasattr(context, 'metadata') and context.metadata:
                params = context.metadata.get('parameters', {})
                if params:
                    return params

            # Try to parse user input as JSON
            user_input = context.get_user_input()
            if user_input:
                try:
                    return json.loads(user_input)
                except json.JSONDecodeError:
                    pass

            return {}
        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
            return {}

    def _sanitize_params_for_log(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters for logging (hide large base64 data)."""
        sanitized = {}
        for key, value in params.items():
            if key == "document_base64" and isinstance(value, str):
                sanitized[key] = f"<base64, {len(value)} chars>"
            else:
                sanitized[key] = value
        return sanitized
