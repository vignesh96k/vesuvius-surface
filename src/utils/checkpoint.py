"""Checkpoint save / load helpers for training resumption and inference."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints under an experiment directory.

    Args:
        checkpoint_dir: Directory where checkpoint files are written.
        monitor: Metric key used to decide the "best" checkpoint.
        mode: ``max`` keeps higher metric values; ``min`` keeps lower.
        save_top_k: Number of best checkpoints to retain (``<=0`` keeps all).
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "val/dice",
        mode: str = "max",
        save_top_k: int = 3,
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_score: Optional[float] = None
        self._saved: list[tuple[float, Path]] = []

        logger.info(
            "CheckpointManager ready at %s (monitor=%s, mode=%s, top_k=%s)",
            self.checkpoint_dir,
            self.monitor,
            self.mode,
            self.save_top_k,
        )

    def _is_improved(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def save(
        self,
        state: dict[str, Any],
        filename: str,
        metric_value: Optional[float] = None,
        is_last: bool = False,
    ) -> Path:
        """Serialize ``state`` to disk and optionally track best checkpoints.

        Args:
            state: Checkpoint payload (model, optimizer, epoch, metrics, ...).
            filename: Destination file name inside ``checkpoint_dir``.
            metric_value: Optional monitored metric for best-model bookkeeping.
            is_last: If ``True``, also copy the file to ``last.pt``.

        Returns:
            Path to the written checkpoint.
        """
        path = self.checkpoint_dir / filename
        torch.save(state, path)
        logger.info("Saved checkpoint: %s", path)

        if is_last:
            last_path = self.checkpoint_dir / "last.pt"
            shutil.copy2(path, last_path)
            logger.debug("Updated last checkpoint symlink/copy: %s", last_path)

        if metric_value is not None:
            self._saved.append((float(metric_value), path))
            if self._is_improved(float(metric_value)):
                self.best_score = float(metric_value)
                best_path = self.checkpoint_dir / "best.pt"
                shutil.copy2(path, best_path)
                logger.info(
                    "New best checkpoint (%.6f) -> %s",
                    metric_value,
                    best_path,
                )
            self._prune()

        return path

    def _prune(self) -> None:
        """Keep only the top-k checkpoints by monitored metric."""
        if self.save_top_k <= 0:
            return

        reverse = self.mode == "max"
        self._saved.sort(key=lambda item: item[0], reverse=reverse)
        to_remove = self._saved[self.save_top_k :]
        self._saved = self._saved[: self.save_top_k]

        for _, path in to_remove:
            if path.name in {"best.pt", "last.pt"}:
                continue
            if path.exists():
                path.unlink()
                logger.debug("Pruned checkpoint: %s", path)

    @staticmethod
    def load(
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load a checkpoint dictionary from disk.

        Args:
            path: Checkpoint file path.
            map_location: Device mapping passed to ``torch.load``.

        Returns:
            Deserialized checkpoint dictionary.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=map_location)
        logger.info("Loaded checkpoint from %s", path)
        return checkpoint
