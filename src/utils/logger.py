"""Logging configuration using loguru."""

import sys

from loguru import logger

# Remove default handler
logger.remove()

# Add console handler with formatting
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
)

__all__ = ["logger"]
