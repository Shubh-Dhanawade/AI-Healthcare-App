"""Logging configuration using Loguru."""

import sys
import io
from loguru import logger


def setup_logging():
    """Configure application logging."""
    logger.remove()
    
    # Console logging — wrap stdout in a UTF-8 TextIOWrapper to avoid
    # UnicodeEncodeError on Windows (CP1252) when log messages contain emoji.
    try:
        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except AttributeError:
        # Fallback if stdout.buffer is not available (e.g., IDLE)
        utf8_stdout = sys.stdout

    logger.add(
        utf8_stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=False,
    )
    
    # File logging
    logger.add(
        "./logs/healthcare_ai.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )
