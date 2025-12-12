"""
Logging Utility Module
Provides structured logging with JSON and text formats.
"""

import logging
import sys
import json
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
from logging.handlers import RotatingFileHandler

from settings import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Custom text formatter for human-readable logging."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def parse_log_rotation_size(size_str: str) -> int:
    """
    Parse log rotation size string (e.g., '10MB', '1GB') to bytes.

    Args:
        size_str: Size string

    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()
    # Check units from longest to shortest to avoid partial matches
    units = [
        ("GB", 1024 ** 3),
        ("MB", 1024 ** 2),
        ("KB", 1024),
        ("G", 1024 ** 3),
        ("M", 1024 ** 2),
        ("K", 1024),
        ("B", 1)
    ]

    for unit, multiplier in units:
        if size_str.endswith(unit):
            number = float(size_str[:-len(unit)])
            return int(number * multiplier)

    # Default to bytes if no unit
    return int(size_str)


def setup_logging():
    """Configure logging based on settings."""
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if settings.log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    # Console handler (always enabled for agent)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation (if configured)
    if settings.log_file:
        log_file = Path(settings.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        max_bytes = parse_log_rotation_size(settings.log_rotation)
        backup_count = settings.log_retention_days  # Simplified: 1 file per day

        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, settings.log_level))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Convenience logging functions
def log_info(message: str, **kwargs):
    """Log info message with extra data."""
    logger = logging.getLogger("agent")
    logger.info(message, extra=kwargs)


def log_error(message: str, exception: Exception = None, **kwargs):
    """Log error message with exception and extra data."""
    logger = logging.getLogger("agent")
    if exception:
        logger.error(message, exc_info=exception, extra=kwargs)
    else:
        logger.error(message, extra=kwargs)


def log_warning(message: str, **kwargs):
    """Log warning message with extra data."""
    logger = logging.getLogger("agent")
    logger.warning(message, extra=kwargs)


def log_debug(message: str, **kwargs):
    """Log debug message with extra data."""
    logger = logging.getLogger("agent")
    logger.debug(message, extra=kwargs)


# Initialize logging on module import
setup_logging()
