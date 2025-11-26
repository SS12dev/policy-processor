"""
Main entry point for the Streamlit application.

Launches the Streamlit web interface for the Policy Document Processor.
"""

import subprocess
import sys
from pathlib import Path

from app.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Launch the Streamlit application."""
    logger.info("=" * 80)
    logger.info("Policy Document Processor - Streamlit Frontend")
    logger.info("=" * 80)

    # Path to the Streamlit app
    app_path = Path(__file__).parent / "app" / "streamlit_app" / "app.py"

    if not app_path.exists():
        logger.error(f"Streamlit app not found at: {app_path}")
        sys.exit(1)

    logger.info(f"Starting Streamlit server...")
    logger.info(f"App path: {app_path}")
    logger.info("=" * 80)

    # Run Streamlit
    subprocess.run([
        "streamlit", "run",
        str(app_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])


if __name__ == "__main__":
    main()
