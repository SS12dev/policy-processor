"""
A2A Server for Policy Document Processor.

FastAPI-based A2A server with streaming support and settings integration.
"""

import sys
from pathlib import Path
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.events import InMemoryQueueManager
from a2a import types

from settings import settings
from utils.logger import get_logger
from a2a.agent import PolicyProcessorAgent
from a2a.redis_storage import RedisAgentStorage
from core.langgraph_orchestrator import LangGraphOrchestrator

logger = get_logger(__name__)


def create_agent_card() -> types.AgentCard:
    """
    Create an agent card based on settings.

    Returns:
        Agent card object
    """
    return types.AgentCard(
        name=settings.agent_name,
        version=settings.agent_version,
        description=settings.agent_description,
        url=f"http://{settings.server_host}:{settings.server_port}",
        capabilities=types.AgentCapabilities(
            streaming=settings.a2a_streaming,
            multiturn=settings.a2a_multiturn,
            push_notifications=settings.a2a_push_notifications,
        ),
        default_input_modes=[types.InputMode.text],
        default_output_modes=[types.OutputMode.text],
        authentication_schemes=[
            types.AuthenticationScheme(
                type="none",
                description="No authentication required"
            )
        ],
        skills=[
            types.AgentSkill(
                id="process_policy",
                name="Process Policy Document",
                description="Upload and process a policy document (PDF). Extracts policies, generates decision trees, validates structure, and stores results. Can also retrieve results for previously processed documents.",
                tags=["document", "processing", "policy", "extraction"],
                input_modes=[types.InputMode.text, types.InputMode.file],
                output_modes=[types.OutputMode.text, types.OutputMode.data],
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


def create_a2a_server(redis_storage: RedisAgentStorage):
    """
    Create the A2A server.

    Args:
        redis_storage: Redis storage for temporary state

    Returns:
        Configured FastAPI application
    """
    try:
        logger.info("Creating A2A server...")

        # Create LangGraph orchestrator
        orchestrator = LangGraphOrchestrator()

        # Create the agent executor (stateless, Redis-based)
        agent_executor = PolicyProcessorAgent(orchestrator, redis_storage)

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

        logger.info(f"A2A server created successfully")
        logger.info(f"   - Server URL: http://{settings.server_host}:{settings.server_port}")
        logger.info(f"   - Agent card: http://{settings.server_host}:{settings.server_port}/.well-known/agent-card.json")
        logger.info(f"   - RPC endpoint: http://{settings.server_host}:{settings.server_port}/")
        logger.info(f"   - Skills: {len(agent_card.skills)} available")
        logger.info(f"   - Redis TTL: {settings.redis_result_ttl_hours}h")

        return app

    except Exception as e:
        logger.error(f"Failed to create A2A server: {e}", exc_info=True)
        raise


def run_a2a_server():
    """
    Run the A2A server using settings configuration.
    """
    try:
        import uvicorn

        # Initialize Redis storage (stateless, container-friendly)
        redis_storage = RedisAgentStorage(result_ttl_hours=settings.redis_result_ttl_hours)
        logger.info(f"Redis storage initialized with {settings.redis_result_ttl_hours}h TTL")

        # Create the app
        app = create_a2a_server(redis_storage)

        logger.info("=" * 80)
        logger.info("Policy Document Processor - A2A Server")
        logger.info("=" * 80)
        logger.info(f"Server: http://{settings.server_host}:{settings.server_port}")
        logger.info(f"Agent card: http://{settings.server_host}:{settings.server_port}/.well-known/agent-card.json")
        logger.info(f"RPC endpoint: http://{settings.server_host}:{settings.server_port}/")
        logger.info(f"Health check: http://{settings.server_host}:{settings.server_port}/health")
        logger.info("=" * 80)

        # Run the server
        uvicorn.run(
            app,
            host=settings.server_host,
            port=settings.server_port,
            reload=settings.server_reload,
            log_level=settings.log_level.lower()
        )

    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        raise
    except Exception as e:
        logger.error(f"Failed to start A2A server: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_a2a_server()
