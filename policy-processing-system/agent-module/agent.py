"""
A2A Agent for Policy Document Processing - Stateless Redis-based version.

This agent:
- Uses Redis for temporary storage (no database)
- Returns full results via A2A events
- Supports concurrent requests
- Designed for containerized deployment
"""

import json
import base64
import hashlib
from typing import Dict, Any
from datetime import datetime
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a import types
from typing_extensions import override

from utils.logger import get_logger
from utils.redis_storage import RedisAgentStorage
from core.langgraph_orchestrator import LangGraphOrchestrator
from models.schemas import ProcessingRequest

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
        queue: EventQueue,
    ) -> None:
        """
        Execute an incoming A2A request.

        Args:
            context: Request context containing the user's request
            queue: Queue for sending responses back to the client
        """
        try:
            logger.info(f"[AGENT] Processing A2A request - Task: {context.task_id}")

            # Extract parameters from context
            parameters = self._extract_parameters(context)
            logger.info(f"[AGENT] Parameters: {self._sanitize_params_for_log(parameters)}")

            # Route to appropriate handler
            if "document_base64" in parameters:
                await self._process_document(context, queue, parameters)
            elif "job_id" in parameters and not parameters.get("document_base64"):
                await self._get_results(context, queue, parameters["job_id"])
            else:
                await self._send_error(
                    context,
                    queue,
                    "Invalid request. Provide 'document_base64' or 'job_id'."
                )

        except Exception as e:
            logger.error(f"Error executing request: {e}", exc_info=True)
            await self._send_error(context, queue, str(e))

    async def _process_document(
        self,
        context: RequestContext,
        queue: EventQueue,
        parameters: Dict[str, Any]
    ) -> None:
        """Process a policy document with Redis-based state management."""
        job_id = context.task_id  # Use task_id as job_id

        try:
            # Acquire lock to prevent duplicate processing
            if not self.storage.acquire_lock(job_id, timeout_seconds=600):
                await self._send_error(
                    context,
                    queue,
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

                # Send initial status
                await queue.enqueue_event(types.TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    status=types.TaskStatus(
                        state=types.TaskState.working,
                        message=self._create_status_message("Starting policy document processing...")
                    ),
                    final=False
                ))

                # Validate document
                try:
                    pdf_bytes = base64.b64decode(document_base64)
                    doc_hash = hashlib.sha256(pdf_bytes).hexdigest()
                    logger.info(f"[AGENT] Document: {len(pdf_bytes)} bytes, hash: {doc_hash[:16]}...")

                    # Send validation status
                    await queue.enqueue_event(types.TaskStatusUpdateEvent(
                        task_id=context.task_id,
                        context_id=context.context_id,
                        status=types.TaskStatus(
                            state=types.TaskState.working,
                            message=self._create_status_message(f"Validating PDF document ({len(pdf_bytes)} bytes)...")
                        ),
                        final=False
                    ))
                except Exception as e:
                    await self._send_error(context, queue, f"Invalid base64: {str(e)}")
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

                # Send processing status
                await queue.enqueue_event(types.TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    status=types.TaskStatus(
                        state=types.TaskState.working,
                        message=self._create_status_message("Processing document with LangGraph orchestrator...")
                    ),
                    final=False
                ))

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

                # Send result as artifact
                await queue.enqueue_event(types.TaskArtifactUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    artifact=types.Artifact(
                        artifact_id=f"artifact_{job_id}",
                        name="policy_processing_results.json",
                        description="Complete policy processing results with decision trees",
                        parts=[
                            types.Part(root=types.DataPart(data=result_data))
                        ]
                    ),
                    last_chunk=True,
                    append=False
                ))

                # Send final completion status
                validation_status = "Passed" if response.validation_result.is_valid else "Failed"
                completion_message = (
                    f"Processing complete! Job ID: {job_id}\n"
                    f"Total Policies: {response.policy_hierarchy.total_policies}\n"
                    f"Decision Trees: {len(response.decision_trees)}\n"
                    f"Validation: {validation_status}"
                )

                await queue.enqueue_event(types.TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    status=types.TaskStatus(
                        state=types.TaskState.completed,
                        message=self._create_status_message(completion_message)
                    ),
                    final=True
                ))

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

            # Send failure status
            await queue.enqueue_event(types.TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=types.TaskStatus(
                    state=types.TaskState.failed,
                    message=self._create_status_message(f"Processing error: {str(e)}")
                ),
                final=True
            ))

    async def _get_results(
        self,
        context: RequestContext,
        queue: EventQueue,
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
                    await queue.enqueue_event(types.TaskStatusUpdateEvent(
                        task_id=context.task_id,
                        context_id=context.context_id,
                        status=types.TaskStatus(
                            state=types.TaskState.working,
                            message=self._create_status_message(
                                f"Job {job_id} status: {status_msg}. Results not yet available or expired."
                            )
                        ),
                        final=True
                    ))
                else:
                    await self._send_error(context, queue, f"Job {job_id} not found")
                return

            # Send results as artifact
            await queue.enqueue_event(types.TaskArtifactUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                artifact=types.Artifact(
                    artifact_id=f"artifact_{job_id}_retrieved",
                    name="policy_processing_results.json",
                    description=f"Retrieved results for job {job_id}",
                    parts=[
                        types.Part(root=types.DataPart(data=result_data))
                    ]
                ),
                last_chunk=True,
                append=False
            ))

            # Send completion status
            await queue.enqueue_event(types.TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=types.TaskStatus(
                    state=types.TaskState.completed,
                    message=self._create_status_message(f"Results retrieved for job {job_id}")
                ),
                final=True
            ))

            logger.info(f"[AGENT] Results sent for job {job_id}")

        except Exception as e:
            logger.error(f"Error getting results: {e}", exc_info=True)
            await self._send_error(context, queue, str(e))

    def _create_status_message(self, text: str) -> types.Message:
        """Create a properly formatted status message."""
        return types.Message(
            message_id=f"status_{uuid4()}",
            role=types.Role.agent,
            parts=[types.Part(root=types.TextPart(text=text))]
        )

    async def _send_error(
        self,
        context: RequestContext,
        queue: EventQueue,
        error_message: str
    ) -> None:
        """Send an error status."""
        logger.error(f"[AGENT] Error: {error_message}")
        await queue.enqueue_event(types.TaskStatusUpdateEvent(
            task_id=context.task_id,
            context_id=context.context_id,
            status=types.TaskStatus(
                state=types.TaskState.failed,
                message=self._create_status_message(f"ERROR: {error_message}")
            ),
            final=True
        ))

    @override
    async def cancel(
        self,
        task_id: str,
        context_id: str
    ) -> None:
        """Handle cancellation of an ongoing task."""
        logger.warning(f"Cancel requested for task: {task_id}")
        # Note: Actual cancellation implementation would require more complex state management

    def _extract_parameters(self, context: RequestContext) -> Dict[str, Any]:
        """Extract parameters from the request context, including FilePart for documents."""
        try:
            parameters = {}
            user_input = context.message
            
            if user_input and user_input.parts:
                for part in user_input.parts:
                    # Extract file from FilePart (proper A2A way)
                    if isinstance(part.root, types.FilePart):
                        file_data = part.root.file
                        if hasattr(file_data, 'bytes') and file_data.bytes:
                            parameters["document_base64"] = file_data.bytes
                            parameters["filename"] = getattr(file_data, 'name', 'document.pdf')
                            logger.info(f"[AGENT] Extracted file from FilePart: {parameters['filename']}")
                    
                    # Extract other parameters from TextPart (JSON)
                    elif isinstance(part.root, types.TextPart):
                        try:
                            text_params = json.loads(part.root.text)
                            # Merge text parameters, but don't override file data
                            for key, value in text_params.items():
                                if key not in parameters:
                                    parameters[key] = value
                        except json.JSONDecodeError:
                            # Not JSON, ignore
                            pass

            return parameters
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


# Global agent instance
_agent: PolicyProcessorAgent | None = None


def get_agent() -> PolicyProcessorAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        from settings import settings
        from utils.poppler_installer import ensure_poppler_available
        from utils.tesseract_installer import ensure_tesseract_available

        # Ensure Poppler is available (auto-install if needed)
        logger.info("Checking Poppler installation...")
        poppler_ok = ensure_poppler_available(auto_install=True)
        if poppler_ok:
            logger.info("Poppler is ready for PDF to image conversion")
        else:
            logger.warning("Poppler not available - OCR functionality will be limited")

        # Ensure Tesseract is available (auto-install if needed)
        logger.info("Checking Tesseract OCR installation...")
        tesseract_ok = ensure_tesseract_available(auto_install=True)
        if tesseract_ok:
            logger.info("Tesseract OCR is ready for text extraction from images")
        else:
            logger.warning("Tesseract OCR not available - text extraction from scanned pages will fail")
            logger.warning("Install with: choco install tesseract (or see logs for installation instructions)")

        if poppler_ok and tesseract_ok:
            logger.info("✅ Full OCR pipeline ready (Poppler + Tesseract)")
        elif poppler_ok:
            logger.warning("⚠️  Partial OCR: Poppler OK, but Tesseract missing")
        else:
            logger.warning("⚠️  OCR unavailable: Install Poppler and Tesseract for full functionality")

        # Initialize Redis storage
        redis_storage = RedisAgentStorage(result_ttl_hours=settings.redis_result_ttl_hours)

        # Initialize LangGraph orchestrator
        orchestrator = LangGraphOrchestrator()

        # Create agent
        _agent = PolicyProcessorAgent(orchestrator, redis_storage)
        logger.info("Global PolicyProcessorAgent instance created")

    return _agent
