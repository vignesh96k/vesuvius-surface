"""Logging setup and helpers for console / file experiment logs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[str | Path] = None,
    name: str = "vesuvius",
    level: int = logging.INFO,
    filename: str = "train.log",
) -> logging.Logger:
    """Configure root application logging with console and optional file handlers.

    Args:
        log_dir: Directory for log files. If ``None``, console-only logging.
        name: Logger name.
        level: Logging level.
        filename: Log file name inside ``log_dir``.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "vesuvius") -> logging.Logger:
    """Return a named logger, creating a basic one if unset.

    Args:
        name: Logger name, typically a module path.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers and not logging.getLogger("vesuvius").handlers:
        # Fallback so imports work before setup_logging is called.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
    return logger
