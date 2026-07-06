"""Logging setup for the Finance CA Assistant package."""

from __future__ import annotations

import logging
import os
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured package logger.

    The log level can be controlled with ``FINANCE_CA_LOG_LEVEL`` or by passing
    ``level`` explicitly. Handlers are added only once per logger.
    """

    logger = logging.getLogger(name)
    selected_level = (level or os.getenv("FINANCE_CA_LOG_LEVEL") or "INFO").upper()
    logger.setLevel(getattr(logging, selected_level, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)

    logger.propagate = False
    return logger

