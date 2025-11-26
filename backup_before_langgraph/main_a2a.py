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
    logger.info("Policy Document Processor - A2A Server")
    logger.info("=" * 80)

    # Configuration
    db_path = "./data/policy_processor.db"
    host = "0.0.0.0"
    port = 8001
    reload = False  # Set to True for development auto-reload

    logger.info(f"Database: {db_path}")
    logger.info(f"Server: http://{host}:{port}")
    logger.info("=" * 80)

    # Run the server
    run_a2a_server(
        db_path=db_path,
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    main()
