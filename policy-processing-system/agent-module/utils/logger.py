"""
Logging configuration and utilities.
"""
import sys
import os
from loguru import logger


def setup_logger():
    """Configure the application logger."""
    logger.remove()  # Remove default handler

    # Get log level from environment or use INFO as default
    log_level = os.getenv("LOG_LEVEL", "INFO")

    # Console handler with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # File handler for persistent logs
    logger.add(
        "logs/policy_processor_{time:YYYY-MM-DD}.log",
        rotation="500 MB",
        retention="10 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    return logger


def get_logger(name: str = None):
    """Get a logger instance with optional name."""
    if name:
        return logger.bind(name=name)
    return logger


# Initialize logger on module import
setup_logger()
