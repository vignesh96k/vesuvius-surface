"""Experiment directory helpers (training viz stubs live in analysis/)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def create_experiment_dirs(
    output_dir: str | Path,
    experiment_name: str,
) -> dict[str, Path]:
    """Create a standard experiment folder layout."""
    root = Path(output_dir) / experiment_name
    dirs = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "predictions": root / "predictions",
        "visualizations": root / "visualizations",
        "configs": root / "configs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment directories ready under %s", root)
    return dirs


def log_batch_preview(
    batch: dict[str, Any],
    predictions: Optional[Any] = None,
    max_items: int = 4,
) -> None:
    """Log a lightweight textual preview of a batch."""
    keys = list(batch.keys())
    n = None
    for value in batch.values():
        if hasattr(value, "shape") and len(getattr(value, "shape", [])) > 0:
            n = int(value.shape[0])
            break
    logger.debug(
        "Batch preview keys=%s n=%s predictions=%s max_items=%s",
        keys,
        n,
        predictions is not None,
        max_items,
    )
