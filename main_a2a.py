"""
Main entry point for the A2A Server.

Runs the Policy Document Processor as an A2A-compliant agent server.
"""

from pathlib import Path

from app.utils.logger import get_logger
from app.a2a.server import run_a2a_server

logger = get_logger(__name__)


def main():
    """Run the A2A server."""
    logger.info("=" * 80)
    logger.info("Policy Document Processor - A2A Server (Stateless)")
    logger.info("=" * 80)

    # Configuration
    host = "0.0.0.0"
    port = 8001
    reload = False  # Set to True for development auto-reload
    result_ttl_hours = 24  # Redis TTL for results

    logger.info(f"Server: http://{host}:{port}")
    logger.info(f"Storage: Redis (TTL: {result_ttl_hours}h)")
    logger.info(f"Architecture: Stateless, container-ready")
    logger.info("=" * 80)

    # Run the server
    run_a2a_server(
        host=host,
        port=port,
        reload=reload,
        result_ttl_hours=result_ttl_hours
    )


if __name__ == "__main__":
    main()
