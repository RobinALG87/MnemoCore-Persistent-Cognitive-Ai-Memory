"""
MnemoCore Logging Configuration
================================
Centralized logging configuration using loguru.

Provides:
  - configure_logging(): Setup function called at application startup
  - JSON log format when LOG_FORMAT=json environment variable is set
  - Consistent logging across all modules

Usage:
    from src.core.logging_config import configure_logging, get_logger

    # At application startup:
    configure_logging(level="INFO", json_format=False)

    # In modules:
    logger = get_logger(__name__)
    logger.info("Message")
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from loguru import logger

# Remove default handler
logger.remove()

# Track if logging has been configured
_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    json_format: Optional[bool] = None,
    *,
    sink: Optional[str] = None,
) -> None:
    """
    Configure loguru logging for MnemoCore.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, use JSON format. If None, check LOG_FORMAT env var.
        sink: Optional file path for log output. If None, logs to stderr.

    Environment:
        LOG_FORMAT: Set to "json" to enable JSON formatted logs.
        LOG_LEVEL: Override log level if not specified.
    """
    global _CONFIGURED

    # Check environment for JSON format
    if json_format is None:
        json_format = os.environ.get("LOG_FORMAT", "").lower() == "json"

    # Check environment for log level
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")

    # Remove existing handlers
    logger.remove()

    # Determine sink
    log_sink = sink if sink else sys.stderr

    if json_format:
        # JSON format for production/cloud logging
        format_str = (
            '{{"timestamp": "{{time:YYYY-MM-DDTHH:mm:ss.SSSZ}}", '
            '"level": "{{level}}", '
            '"logger": "{{name}}", '
            '"function": "{{function}}", '
            '"line": {{line}}, '
            '"message": "{{message}}", '
            '"exception": "{{exception}}"}}'
        )
    else:
        # Human-readable format for development
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add handler
    logger.add(
        log_sink,
        level=level.upper(),
        format=format_str,
        colorize=not json_format and sink is None,
        enqueue=True,  # Thread-safe
        backtrace=True,
        diagnose=True,
    )

    # Intercept standard logging
    _intercept_standard_logging(level)

    _CONFIGURED = True
    logger.debug(f"Logging configured: level={level}, json_format={json_format}")


def _intercept_standard_logging(level: str) -> None:
    """
    Intercept standard library logging and redirect to loguru.

    This ensures that logs from third-party libraries using stdlib logging
    are also handled by loguru.
    """
    import logging

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                log_level = logger.level(record.levelname).name
            except ValueError:
                log_level = record.levelno

            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back  # type: ignore
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                log_level, record.getMessage()
            )

    # Configure root logger to use our handler
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Set levels for common noisy loggers
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        logging.getLogger(logger_name).setLevel(level.upper())


def get_logger(name: str = __name__):
    """
    Get a logger instance bound to the specified module name.

    Args:
        name: Module name (typically __name__).

    Returns:
        A loguru logger instance bound to the module.
    """
    # Ensure logging is configured with defaults if not already done
    if not _CONFIGURED:
        configure_logging()

    return logger.bind(name=name)


# Module-level logger for convenience
__all__ = ["configure_logging", "get_logger", "logger"]
