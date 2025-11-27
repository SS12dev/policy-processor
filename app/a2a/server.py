"""
A2A Server for Policy Document Processor.

Single endpoint A2A server with streaming support.
"""

import logging
from pathlib import Path

from a2a.server.apps.jsonrpc import A2AStarletteApplication
from a2a.types import AgentCard
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from app.utils.logger import get_logger
from app.a2a.agent import PolicyProcessorAgent
from app.a2a.redis_storage import RedisAgentStorage
from app.core.langgraph_orchestrator import LangGraphOrchestrator

logger = get_logger(__name__)


def create_agent_card(host: str = "0.0.0.0", port: int = 8001) -> dict:
    """
    Create an agent card with a single processing endpoint.

    Args:
        host: Server host
        port: Server port

    Returns:
        Agent card dictionary
    """
    return {
        "name": "Policy Document Processor Agent",
        "version": "4.0.0",
        "description": "Stateless policy document processor with Redis-based temporary storage. Processes PDFs and generates hierarchical decision trees with eligibility questions. Powered by LangGraph state machine. Designed for container deployment and horizontal scaling.",
        "url": f"http://{host}:{port}",
        "capabilities": {
            "streaming": True,
            "multiturn": False,
            "push_notifications": False,
        },
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "authentication_schemes": [
            {
                "type": "none",
                "description": "No authentication required"
            }
        ],
        "skills": [
            {
                "id": "process_policy",
                "name": "Process Policy Document",
                "description": "Upload and process a policy document (PDF). Extracts policies, generates decision trees, validates structure, and stores results. Can also retrieve results for previously processed documents.",
                "tags": ["document", "processing", "policy", "extraction"],
                "input_schema": {
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
                            "description": "Minimum confidence threshold (0.0-1.0, default: 0.7)",
                            "default": 0.7,
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "oneOf": [
                        {"required": ["document_base64"]},
                        {"required": ["job_id"]}
                    ]
                },
                "output_schema": {
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
            }
        ],
        "privacy_policy": "Results stored temporarily in Redis with TTL. No persistent storage at agent layer.",
        "terms_of_service": "For authorized use only.",
        "metadata": {
            "framework": "A2A SDK + LangGraph + Python + Redis",
            "architecture": "Stateless LangGraph state machine with 8 nodes, Redis temporary storage, container-ready",
            "llm_models": ["gpt-4o", "gpt-4o-mini"],
            "storage": "Redis with TTL (24h default)",
            "deployment": "Container-friendly, horizontally scalable",
            "features": [
                "langgraph_state_machine",
                "stateless_architecture",
                "redis_temporary_storage",
                "concurrent_requests",
                "job_locking",
                "single_endpoint",
                "streaming_support",
                "decision_trees",
                "validation",
                "automatic_retry_logic"
            ]
        }
    }


def create_a2a_server(
    redis_storage: RedisAgentStorage,
    host: str = "0.0.0.0",
    port: int = 8001
):
    """
    Create the A2A server.

    Args:
        redis_storage: Redis storage for temporary state
        host: Host to bind to
        port: Port to bind to

    Returns:
        Configured Starlette application
    """
    try:
        logger.info("Creating A2A server...")

        # Create LangGraph orchestrator
        orchestrator = LangGraphOrchestrator()

        # Create the agent executor (stateless, Redis-based)
        agent_executor = PolicyProcessorAgent(orchestrator, redis_storage)

        # Create task store
        task_store = InMemoryTaskStore()

        # Create request handler
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=task_store
        )

        # Create agent card
        agent_card_dict = create_agent_card(host, port)

        # Create AgentCard object
        agent_card = AgentCard.model_validate(agent_card_dict)

        # Create the A2A Starlette application
        a2a_app = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        # Build the actual Starlette ASGI app
        app = a2a_app.build(
            agent_card_url="/.well-known/agent-card",
            rpc_url="/jsonrpc"
        )

        # Add health check endpoints (async for concurrent processing)
        from starlette.routing import Route
        from starlette.responses import JSONResponse

        async def health_check(request):
            """Async health check endpoint for deployment readiness."""
            return JSONResponse({
                "status": "healthy",
                "service": "policy-processor-a2a",
                "version": "4.0.0",
                "architecture": "stateless",
                "storage": "redis",
                "langgraph": "active"
            })

        async def liveness_check(request):
            """Async liveness probe for Kubernetes/deployment."""
            return JSONResponse({"alive": True})

        # Add routes to existing app
        app.routes.append(Route("/health", health_check, methods=["GET"]))
        app.routes.append(Route("/health/liveness", liveness_check, methods=["GET"]))

        logger.info(f"A2A server created at {agent_card_dict['url']}")
        logger.info(f"   - Agent card: {agent_card_dict['url']}/.well-known/agent-card")
        logger.info(f"   - JSON-RPC: {agent_card_dict['url']}/jsonrpc")
        logger.info(f"   - Health check: {agent_card_dict['url']}/health")
        logger.info(f"   - Liveness: {agent_card_dict['url']}/health/liveness")
        logger.info(f"   - Skills: {len(agent_card_dict['skills'])} available")

        return app

    except Exception as e:
        logger.error(f"Failed to create A2A server: {e}", exc_info=True)
        return None


def run_a2a_server(
    host: str = "0.0.0.0",
    port: int = 8001,
    reload: bool = False,
    result_ttl_hours: int = 24
):
    """
    Run the A2A server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        result_ttl_hours: Hours to keep results in Redis (default: 24)
    """
    try:
        import uvicorn

        # Initialize Redis storage (stateless, container-friendly)
        redis_storage = RedisAgentStorage(result_ttl_hours=result_ttl_hours)
        logger.info(f"Redis storage initialized with {result_ttl_hours}h TTL")

        # Create the app
        app = create_a2a_server(redis_storage, host, port)

        if app is None:
            logger.error("Failed to create A2A app. Exiting.")
            return

        logger.info("=" * 80)
        logger.info("Policy Document Processor - A2A Server")
        logger.info("=" * 80)
        logger.info(f"Server URL: http://{host}:{port}")
        logger.info(f"Agent Card: http://{host}:{port}/.well-known/agent-card")
        logger.info(f"JSON-RPC: http://{host}:{port}/jsonrpc")
        logger.info("=" * 80)

        # Run the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
    except Exception as e:
        logger.error(f"Failed to start A2A server: {e}", exc_info=True)


if __name__ == "__main__":
    run_a2a_server(
        host="0.0.0.0",
        port=8001,
        reload=True,
        result_ttl_hours=24
    )
