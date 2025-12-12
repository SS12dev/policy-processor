"""
A2A Server
Wraps the LangGraph agent and provides A2A protocol interface.
"""

import asyncio
from typing import AsyncIterator
from uuid import uuid4

from a2a import types
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import InMemoryQueueManager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from settings import settings
from agent import get_agent
from utils.logger import get_logger
from utils.metrics import get_metrics_collector, track_request

logger = get_logger(__name__)


# Rate limiter
limiter = Limiter(key_func=get_remote_address)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request size limits."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            size_bytes = int(content_length)
            if size_bytes > settings.max_request_size_bytes:
                logger.warning(
                    f"Request size ({size_bytes / (1024*1024):.2f} MB) exceeds limit",
                    extra={"remote_addr": request.client.host}
                )
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request Too Large",
                        "max_size_mb": settings.max_request_size_mb,
                        "received_size_mb": size_bytes / (1024 * 1024)
                    }
                )

        return await call_next(request)


class DocumentAnalysisAgentExecutor(AgentExecutor):
    """A2A Agent Executor that wraps the LangGraph document analysis agent."""

    def __init__(self):
        """Initialize the executor."""
        super().__init__()
        self.agent = get_agent()
        self.metrics = get_metrics_collector()

        logger.info("DocumentAnalysisAgentExecutor initialized")

    async def execute(
        self,
        request_context: RequestContext,
        queue
    ) -> None:
        """
        Execute document analysis on the user's request.

        Args:
            request_context: A2A request context with user message
            queue: Event queue for sending updates to the client
        """
        with track_request() as tracker:
            try:
                logger.info(
                    "Received analysis request",
                    extra={
                        "task_id": request_context.task_id,
                        "context_id": request_context.context_id
                    }
                )

                # Extract PDF from message
                pdf_bytes = None
                user_message = request_context.message

                for part in user_message.parts:
                    if isinstance(part.root, types.FilePart):
                        file_part = part.root
                        if isinstance(file_part.file, types.FileWithBytes):
                            pdf_bytes = file_part.file.bytes
                            logger.info(
                                "PDF file received",
                                extra={"pdf_filename": file_part.file.name}
                            )
                            break

                if not pdf_bytes:
                    logger.error("No PDF file found in request")
                    await queue.enqueue_event(types.TaskStatusUpdateEvent(
                        task_id=request_context.task_id,
                        context_id=request_context.context_id,
                        status=types.TaskStatus(
                            state=types.TaskState.failed,
                            message=self._create_status_message("No PDF file provided")
                        ),
                        final=True
                    ))
                    tracker.failure()
                    return

                # Send initial status
                await queue.enqueue_event(types.TaskStatusUpdateEvent(
                    task_id=request_context.task_id,
                    context_id=request_context.context_id,
                    status=types.TaskStatus(
                        state=types.TaskState.working,
                        message=self._create_status_message("Starting document analysis...")
                    ),
                    final=False
                ))

                # Process document with streaming
                step_count = 0
                final_result = None

                async for event in self.agent.stream_document_processing(
                    pdf_bytes=pdf_bytes,
                    context_id=request_context.context_id
                ):
                    step_count += 1

                    # Extract node name and state from event
                    for node_name, state_update in event.items():
                        # Yield status update for each node
                        status_message = self._get_status_message(node_name, state_update)

                        if status_message:
                            await queue.enqueue_event(types.TaskStatusUpdateEvent(
                                task_id=request_context.task_id,
                                context_id=request_context.context_id,
                                status=types.TaskStatus(
                                    state=types.TaskState.working,
                                    message=self._create_status_message(status_message)
                                ),
                                final=False
                            ))

                        # Save final result
                        if state_update.get("response_ready"):
                            final_result = state_update

                    # Add small delay for streaming effect (configurable)
                    if settings.stream_chunk_size:
                        await asyncio.sleep(0.1)

                # Send final response
                if final_result and final_result.get("formatted_response"):
                    response_data = final_result["formatted_response"]

                    # Create artifact with full JSON response
                    await queue.enqueue_event(types.TaskArtifactUpdateEvent(
                        task_id=request_context.task_id,
                        context_id=request_context.context_id,
                        artifact=types.Artifact(
                            artifact_id=f"artifact_{request_context.task_id}",
                            name="analysis_results.json",
                            description="Complete document analysis results",
                            parts=[
                                types.Part(root=types.DataPart(data=response_data))
                            ]
                        ),
                        last_chunk=True,
                        append=False
                    ))

                    # Final success status
                    await queue.enqueue_event(types.TaskStatusUpdateEvent(
                        task_id=request_context.task_id,
                        context_id=request_context.context_id,
                        status=types.TaskStatus(
                            state=types.TaskState.completed,
                            message=self._create_status_message("Document analysis completed successfully")
                        ),
                        final=True
                    ))

                    tracker.success()

                else:
                    # Processing failed
                    await queue.enqueue_event(types.TaskStatusUpdateEvent(
                        task_id=request_context.task_id,
                        context_id=request_context.context_id,
                        status=types.TaskStatus(
                            state=types.TaskState.failed,
                            message=self._create_status_message("Document analysis failed")
                        ),
                        final=True
                    ))

                    tracker.failure()

            except Exception as e:
                logger.error(
                    f"Document analysis failed: {str(e)}",
                    exc_info=e,
                    extra={
                        "task_id": request_context.task_id,
                        "error_type": type(e).__name__
                    }
                )

                self.metrics.record_error(type(e).__name__, str(e))

                await queue.enqueue_event(types.TaskStatusUpdateEvent(
                    task_id=request_context.task_id,
                    context_id=request_context.context_id,
                    status=types.TaskStatus(
                        state=types.TaskState.failed,
                        message=self._create_status_message(f"Error: {str(e)}")
                    ),
                    final=True
                ))

                tracker.failure()

    def _create_status_message(self, text: str) -> types.Message:
        """Create a status message object."""
        return types.Message(
            message_id=f"status_{uuid4()}",
            role=types.Role.agent,
            parts=[types.Part(root=types.TextPart(text=text))]
        )

    def _get_status_message(self, node_name: str, state: dict) -> str:
        """Get human-readable status message for a node."""
        messages = {
            "validate_pdf": "Validating PDF file...",
            "parse_pdf": "Parsing PDF and extracting text...",
            "extract_headings": "Extracting document headings...",
            "extract_keywords": "Extracting keywords...",
            "analyze_document": "Performing deep analysis with AI...",
            "format_response": "Formatting results..."
        }

        base_message = messages.get(node_name, f"Processing {node_name}...")

        # Add specific details if available
        if node_name == "parse_pdf" and state.get("page_count"):
            return f"{base_message} ({state['page_count']} pages found)"
        elif node_name == "extract_headings" and state.get("heading_count"):
            return f"{base_message} ({state['heading_count']} headings found)"
        elif node_name == "extract_keywords" and state.get("keyword_count"):
            return f"{base_message} ({state['keyword_count']} keywords found)"

        return base_message

    async def cancel(self, task_id: str, context_id: str) -> None:
        """
        Cancel a running task.

        Args:
            task_id: ID of the task to cancel
            context_id: Context ID of the task
        """
        logger.info(
            "Task cancellation requested",
            extra={"task_id": task_id, "context_id": context_id}
        )
        # Note: LangGraph doesn't have native cancellation support yet
        # This is a placeholder for future implementation
        # For now, we just log the cancellation request


def create_server() -> FastAPI:
    """
    Create and configure the A2A server.

    Returns:
        Configured FastAPI application with A2A protocol support
    """
    logger.info("Creating A2A server")

    # Create agent card
    agent_card = types.AgentCard(
        name=settings.agent_name,
        description=settings.agent_description,
        url=settings.agent_url,
        version=settings.agent_version,
        default_input_modes=["file"],
        default_output_modes=["text", "data", "artifact"],
        capabilities=types.AgentCapabilities(
            streaming=settings.enable_streaming,
            push_notifications=False
        ),
        skills=[
            types.AgentSkill(
                id="pdf_analysis",
                name="pdf_analysis",
                description="Analyze PDF documents to extract headings, keywords, and insights",
                tags=["pdf", "document-analysis", "nlp"],
                input_modes=["file"],
                output_modes=["text", "data", "artifact"]
            ),
            types.AgentSkill(
                id="heading_extraction",
                name="heading_extraction",
                description="Extract document headings and structure",
                tags=["pdf", "heading-extraction"],
                input_modes=["file"],
                output_modes=["data"]
            ),
            types.AgentSkill(
                id="keyword_extraction",
                name="keyword_extraction",
                description="Extract keywords and key phrases from documents",
                tags=["pdf", "keyword-extraction", "nlp"],
                input_modes=["file"],
                output_modes=["data"]
            )
        ]
    )

    # Create agent executor
    executor = DocumentAnalysisAgentExecutor()

    # Create task store and queue manager
    task_store = InMemoryTaskStore()
    queue_manager = InMemoryQueueManager()

    # Create request handler
    http_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        queue_manager=queue_manager
    )

    # Create A2A application
    a2a_app = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=http_handler
    )

    # Build the FastAPI app
    app = a2a_app.build()

    # Add middlewares
    app.add_middleware(RequestSizeLimitMiddleware)

    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Add rate limit to endpoints
    @app.get("/health")
    @limiter.limit(f"{settings.rate_limit_requests_per_minute}/minute")
    async def health_check(request: Request):
        """Health check endpoint."""
        return {"status": "healthy", "agent": settings.agent_name}

    @app.get("/metrics")
    @limiter.limit("10/minute")
    async def get_metrics(request: Request):
        """Get agent metrics."""
        metrics = get_metrics_collector().get_metrics()
        return metrics

    logger.info(
        "A2A server created",
        extra={
            "agent_name": settings.agent_name,
            "url": settings.agent_url,
            "streaming_enabled": settings.enable_streaming
        }
    )

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_server()

    logger.info("="*80)
    logger.info(f"Starting {settings.agent_name}")
    logger.info("="*80)
    logger.info(f"Server: {settings.agent_url}")
    logger.info(f"Agent card: {settings.agent_url}/.well-known/agent-card.json")
    logger.info(f"RPC endpoint: {settings.agent_url}/rpc")
    logger.info(f"Health check: {settings.agent_url}/health")
    logger.info(f"Metrics: {settings.agent_url}/metrics")
    logger.info("="*80)

    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.server_port,
        log_level=settings.log_level.lower(),
        timeout_keep_alive=5
    )
