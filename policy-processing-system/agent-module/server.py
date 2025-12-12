"""
A2A Server for Policy Document Processor.

FastAPI-based A2A server with streaming support and settings integration.
"""

from uuid import uuid4

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import InMemoryQueueManager
from a2a import types

from settings import settings
from utils.logger import get_logger
from agent import get_agent

logger = get_logger(__name__)


def create_agent_card() -> types.AgentCard:
    """
    Create an agent card based on settings.

    Returns:
        Agent card object
    """
    # Use localhost for the agent card URL when binding to 0.0.0.0
    # This allows clients to connect properly
    advertised_host = "localhost" if settings.server_host == "0.0.0.0" else settings.server_host
    
    return types.AgentCard(
        name=settings.agent_name,
        version=settings.agent_version,
        description=settings.agent_description,
        url=f"http://{advertised_host}:{settings.server_port}",
        capabilities=types.AgentCapabilities(
            streaming=settings.a2a_streaming,
            push_notifications=settings.a2a_push_notifications,
        ),
        default_input_modes=["text", "file"],
        default_output_modes=["text", "data", "artifact"],
        skills=[
            types.AgentSkill(
                id="process_policy",
                name="Process Policy Document",
                description="Upload and process a policy document (PDF). Extracts policies, generates decision trees, validates structure, and stores results. Can also retrieve results for previously processed documents.",
                tags=["document", "processing", "policy", "extraction"],
                input_modes=["text", "file"],
                output_modes=["text", "data", "artifact"],
                input_schema={
                    "type": "object",
                    "properties": {
                        "document_base64": {
                            "type": "string",
                            "description": "Base64-encoded PDF document to process (required for new processing)"
                        },
                        "job_id": {
                            "type": "string",
                            "description": "Job ID to retrieve results (alternative to document_base64)"
                        },
                        "use_gpt4": {
                            "type": "boolean",
                            "description": "Use GPT-4 for complex extraction (default: false)",
                            "default": False
                        },
                        "enable_streaming": {
                            "type": "boolean",
                            "description": "Enable streaming progress updates (default: true)",
                            "default": True
                        },
                        "confidence_threshold": {
                            "type": "number",
                            "description": f"Minimum confidence threshold (0.0-1.0, default: {settings.default_confidence_threshold})",
                            "default": settings.default_confidence_threshold,
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "oneOf": [
                        {"required": ["document_base64"]},
                        {"required": ["job_id"]}
                    ]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["submitted", "processing", "completed", "failed"]
                        },
                        "message": {"type": "string"},
                        "results": {
                            "type": "object",
                            "description": "Complete processing results (if status is completed)"
                        }
                    }
                }
            )
        ],
        privacy_policy=f"Results stored temporarily in Redis with {settings.redis_result_ttl_hours}h TTL. No persistent storage at agent layer.",
        terms_of_service="For authorized use only.",
        metadata={
            "framework": "A2A SDK + LangGraph + Python + Redis",
            "architecture": "Stateless LangGraph state machine with 8 nodes, Redis temporary storage, container-ready",
            "llm_provider": settings.llm_provider,
            "storage": f"Redis with TTL ({settings.redis_result_ttl_hours}h)",
            "deployment": "Container-friendly, horizontally scalable",
            "features": [
                "langgraph_state_machine",
                "stateless_architecture",
                "redis_temporary_storage",
                "concurrent_requests",
                "job_locking",
                "streaming_support",
                "decision_trees",
                "validation",
                "automatic_retry_logic"
            ]
        }
    )


def create_server():
    """
    Create the A2A server.

    Returns:
        Configured FastAPI application
    """
    try:
        logger.info("Creating A2A server...")

        # Get the global agent instance (lazy initialization)
        agent_executor = get_agent()

        # Create task store and queue manager
        task_store = InMemoryTaskStore()
        queue_manager = InMemoryQueueManager()

        # Create request handler
        http_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=task_store,
            queue_manager=queue_manager
        )

        # Create agent card
        agent_card = create_agent_card()

        # Create the A2A FastAPI application
        a2a_app = A2AFastAPIApplication(
            agent_card=agent_card,
            http_handler=http_handler
        )

        # Build the FastAPI app (uses default endpoints)
        app = a2a_app.build()

        # Add health check endpoint
        @app.get("/health")
        async def health_check():
            """Health check endpoint for monitoring."""
            return {
                "status": "healthy",
                "agent": settings.agent_name,
                "version": settings.agent_version,
                "redis": "connected"
            }

        logger.info(f"A2A server created successfully")
        logger.info(f"   - Server URL: http://{settings.server_host}:{settings.server_port}")
        logger.info(f"   - Agent card: http://{settings.server_host}:{settings.server_port}/.well-known/agent-card.json")
        logger.info(f"   - RPC endpoint: http://{settings.server_host}:{settings.server_port}/")
        logger.info(f"   - Health check: http://{settings.server_host}:{settings.server_port}/health")
        logger.info(f"   - Skills: {len(agent_card.skills)} available")
        logger.info(f"   - Redis TTL: {settings.redis_result_ttl_hours}h")

        return app

    except Exception as e:
        logger.error(f"Failed to create A2A server: {e}", exc_info=True)
        raise


# Create app instance at module level for uvicorn
app = create_server()


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 80)
    logger.info(f"Starting {settings.agent_name}")
    logger.info("=" * 80)
    logger.info(f"Server: http://{settings.server_host}:{settings.server_port}")
    logger.info(f"Agent card: http://{settings.server_host}:{settings.server_port}/.well-known/agent-card.json")
    logger.info(f"RPC endpoint: http://{settings.server_host}:{settings.server_port}/")
    logger.info("=" * 80)

    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.server_port,
        log_level=settings.log_level.lower(),
        reload=settings.server_reload
    )
