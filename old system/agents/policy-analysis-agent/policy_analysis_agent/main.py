import logging
import sys
from pathlib import Path

import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

try:
    from .agent_executor import PolicyAnalysisAgentExecutor
except ImportError:
    from agent_executor import PolicyAnalysisAgentExecutor

try:
    from .utils.settings import agent_settings
except ImportError:
    from utils.settings import agent_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import memory-optimized stores
try:
    # Try both relative and absolute imports
    try:
        from .memory_optimized_a2a_stores import MemoryOptimizedTaskStore, MemoryOptimizedPushNotificationConfigStore
    except ImportError:
        from memory_optimized_a2a_stores import MemoryOptimizedTaskStore, MemoryOptimizedPushNotificationConfigStore
    logger.info("Using memory-optimized A2A stores")
    USING_OPTIMIZED_STORES = True
except ImportError as e:
    # Fallback to original stores
    from a2a.server.tasks import InMemoryPushNotificationConfigStore, InMemoryTaskStore
    
    # Create wrapper classes that ignore the extra parameters
    class MemoryOptimizedTaskStore(InMemoryTaskStore):
        def __init__(self, max_tasks: int = 10):
            super().__init__()
            logger.info(f"Using fallback InMemoryTaskStore (max_tasks={max_tasks} ignored)")
    
    class MemoryOptimizedPushNotificationConfigStore(InMemoryPushNotificationConfigStore):
        def __init__(self, max_configs: int = 5):
            super().__init__()
            logger.info(f"Using fallback InMemoryPushNotificationConfigStore (max_configs={max_configs} ignored)")
    
    logger.warning(f"Failed to import memory-optimized stores, using fallback wrappers: {e}")
    USING_OPTIMIZED_STORES = False

try:
    # Try relative imports first (when run as module)
    from .agent import PolicyAnalysisAgent
    from .agent_executor import PolicyAnalysisAgentExecutor
    from .utils.llm import check_llm_env_vars
    from .utils.settings import agent_settings
except ImportError:
    # Fall back to absolute imports (when run as script)
    from agent import PolicyAnalysisAgent
    from agent_executor import PolicyAnalysisAgentExecutor
    from utils.llm import check_llm_env_vars
    from utils.settings import agent_settings

def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set specific log levels for noisy libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

def create_agent_card(host: str, port: int) -> AgentCard:
    """Create agent card describing capabilities."""

    capabilities = AgentCapabilities(
        streaming=False,
        push_notifications=True,
    )

    analyze_policy_skill = AgentSkill(
        id='analyze_policy',
        name='Policy Analysis',
        description='Analyzes medical policy documents to extract criteria and generate questionnaires in one streamlined operation',
        tags=['medical', 'policy', 'prior-authorization', 'analysis', 'questionnaire'],
        examples=[
            'Analyze bariatric surgery policy and create questionnaire',
            'Extract criteria and generate questions from policy document',
            'Process medical policy for prior authorization requirements',
            'Complete policy analysis with structured output'
        ],
    )

    agent_card = AgentCard(
        name=agent_settings.AGENT_NAME,
        description=agent_settings.AGENT_DESCRIPTION,
        url=f'http://{host}:{port}/',
        version=agent_settings.AGENT_VERSION,
        default_input_modes=PolicyAnalysisAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=PolicyAnalysisAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[analyze_policy_skill],
    )

    return agent_card

# Create app at module level for uvicorn
def create_app():
    """Create the FastAPI application."""
    try:
        check_llm_env_vars()
        logger.info("LLM environment variables verified")

        agent_card = create_agent_card(agent_settings.AGENT_HOST, agent_settings.AGENT_PORT)
        logger.info(f"Agent card created: {agent_card.name}")

        httpx_client = httpx.AsyncClient()
        push_config_store = MemoryOptimizedPushNotificationConfigStore(max_configs=100)  # Increased for concurrency
        task_store = MemoryOptimizedTaskStore(max_tasks=100)  # Increased for concurrency
        
        # Log memory optimization status
        store_type = "optimized" if USING_OPTIMIZED_STORES else "fallback"
        logger.info(f"Using {store_type} stores configured for high concurrency:")
        logger.info(f"  - MemoryOptimizedTaskStore: max_tasks=100 ({store_type})")
        logger.info(f"  - MemoryOptimizedPushNotificationConfigStore: max_configs=100 ({store_type})")
        
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )

        request_handler = DefaultRequestHandler(
            agent_executor=PolicyAnalysisAgentExecutor(),
            task_store=task_store,
            push_config_store=push_config_store,
            push_sender=push_sender,
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        
        app = server.build()
        
        # Add health endpoints for Kubernetes probes
        @app.route('/health/liveness', methods=['GET'])
        async def liveness_check(request):
            """Fast 'is the process up and routing working?' check."""
            from starlette.responses import JSONResponse
            from datetime import datetime, timezone
            return JSONResponse({
                "service": agent_card.name,
                "probe": "liveness",
                "status": "ok",
                "version": agent_card.version,
                "time_utc": datetime.now(timezone.utc).isoformat()
            }, status_code=200)

        @app.route('/health/readiness', methods=['GET'])
        async def readiness_check(request):
            """Readiness probe to check if agent is ready to handle requests."""
            from starlette.responses import JSONResponse
            from datetime import datetime, timezone
            try:
                # Test that agent executor can be created
                _ = PolicyAnalysisAgentExecutor()
                return JSONResponse({
                    "service": agent_card.name,
                    "probe": "readiness",
                    "status": "ready",
                    "version": agent_card.version,
                    "time_utc": datetime.now(timezone.utc).isoformat()
                }, status_code=200)
            except Exception as e:
                return JSONResponse({
                    "service": agent_card.name,
                    "probe": "readiness",
                    "status": "not-ready",
                    "error": str(e),
                    "version": agent_card.version,
                    "time_utc": datetime.now(timezone.utc).isoformat()
                }, status_code=503)

        # Keep the old health endpoint for backward compatibility
        @app.route('/health', methods=['GET'])
        async def health_check(request):
            from starlette.responses import JSONResponse
            return JSONResponse({
                "status": "healthy",
                "agent": agent_card.name,
                "version": agent_card.version
            })

        return app
    except Exception as e:
        logger.error(f'Error during app creation: {e}', exc_info=True)
        raise

app = create_app()

@click.command()
@click.option('--host', 'host', default=agent_settings.AGENT_HOST, help='Host to run server on')
@click.option('--port', 'port', default=agent_settings.AGENT_PORT, help='Port to run server on')
@click.option('--log-level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
def main(host: str, port: int, log_level: str):
    """Start Policy Analysis Agent server with A2A protocol support."""
    try:
        # Update logging level
        setup_logging(log_level)
        logger.info("Policy Analysis Agent starting up...")
        
        # Validate port range
        if not (1 <= port <= 65535):
            raise ValueError(f"Port must be between 1-65535, got: {port}")
        
        # Log startup information
        logger.info(f"Agent: {agent_settings.AGENT_NAME} v{agent_settings.AGENT_VERSION}")
        logger.info(f"Description: {agent_settings.AGENT_DESCRIPTION}")
        logger.info(f"LLM Configuration:")
        logger.info(f"  Model: {agent_settings.PA_LLM_MODEL}")
        logger.info(f"  Endpoint: {agent_settings.PA_LLM_ENDPOINT}")
        logger.info(f"Server Configuration:")
        logger.info(f"  Host: {host}")
        logger.info(f"  Port: {port}")
        logger.info(f"  Log Level: {log_level.upper()}")
        
        # Log service endpoints
        base_url = f"http://{host}:{port}"
        logger.info(f"Service Endpoints:")
        logger.info(f"  Main: {base_url}/")
        logger.info(f"  Agent Card: {base_url}/.well-known/agent-card.json")
        logger.info(f"  Health Check: {base_url}/health")
        logger.info(f"  Liveness Probe: {base_url}/health/liveness")
        logger.info(f"  Readiness Probe: {base_url}/health/readiness")
        
        logger.info("Starting HTTP server...")
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level.lower(),
            access_log=True,
            server_header=False,
            date_header=False
        )

    except ValueError as ve:
        logger.error(f'Configuration error: {ve}')
        logger.error('Please check your .env file and environment variable configuration')
        sys.exit(1)

    except OSError as oe:
        if "Address already in use" in str(oe):
            logger.error(f'Port {port} is already in use. Please choose a different port.')
        else:
            logger.error(f'Network error: {oe}')
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping server...")
        sys.exit(0)

    except Exception as e:
        logger.error(f'Unexpected error during server startup: {e}', exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()