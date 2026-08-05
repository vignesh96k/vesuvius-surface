"""Normalization and light 3D augmentations for Surface Detection patches."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

TransformFn = Callable[[dict[str, Any]], dict[str, Any]]


def normalize_volume(
    volume: np.ndarray,
    method: str = "zscore",
    eps: float = 1e-6,
    percentiles: tuple[float, float] = (1.0, 99.0),
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> np.ndarray:
    """Normalize a 3D volume to ``float32``."""
    arr = volume.astype(np.float32, copy=False)
    method = method.lower()

    if method in {"none", "identity"}:
        return arr
    if method == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
        return (arr - lo) / (hi - lo + eps)
    if method == "percentile":
        lo_p, hi_p = percentiles
        lo = float(np.percentile(arr, lo_p))
        hi = float(np.percentile(arr, hi_p))
        arr = np.clip(arr, lo, hi)
        return (arr - lo) / (hi - lo + eps)
    if method == "zscore":
        if mean is not None and std is not None:
            return (arr - float(mean)) / (float(std) + eps)
        return (arr - float(arr.mean())) / (float(arr.std()) + eps)
    raise ValueError(
        f"Unknown normalization '{method}'. Expected: none, minmax, zscore, percentile."
    )


# Keep old name used by analysis imports.
normalize_image = normalize_volume


class RandomFlip3D:
    """Random flips along D/H/W for image (+ label if present)."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image = sample["image"]
        label = sample.get("label")
        # image: Tensor (1, D, H, W) or ndarray (D,H,W)/(1,D,H,W)
        for axis in (1, 2, 3):  # skip channel dim when tensor CHW-like
            if isinstance(image, torch.Tensor):
                if image.ndim == 4 and torch.rand(1).item() < self.p:
                    image = torch.flip(image, dims=(axis,))
                    if label is not None and isinstance(label, torch.Tensor):
                        label = torch.flip(label, dims=(axis - 1,))
            else:
                # NumPy path: (D,H,W)
                ax = axis - 1
                if np.random.rand() < self.p:
                    image = np.flip(image, axis=ax).copy()
                    if label is not None:
                        label = np.flip(label, axis=ax).copy()
        sample["image"] = image
        if label is not None:
            sample["label"] = label
        return sample


class RandomRot90Planes3D:
    """Random 90-degree rotations in axial / coronal / sagittal planes."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        # Axes pairs on (D,H,W) for rot90; for tensor (C,D,H,W) add +1.
        self._planes = ((1, 2), (0, 2), (0, 1))

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if np.random.rand() >= self.p:
            return sample
        image = sample["image"]
        label = sample.get("label")
        k = int(np.random.randint(0, 4))
        plane = self._planes[int(np.random.randint(0, 3))]
        if isinstance(image, torch.Tensor) and image.ndim == 4:
            dims = (plane[0] + 1, plane[1] + 1)
            image = torch.rot90(image, k, dims=dims)
            if label is not None and isinstance(label, torch.Tensor):
                label = torch.rot90(label, k, dims=plane)
        else:
            image = np.rot90(image, k, axes=plane).copy()
            if label is not None:
                label = np.rot90(label, k, axes=plane).copy()
        sample["image"] = image
        if label is not None:
            sample["label"] = label
        return sample


class Compose3D:
    """Sequential transform list."""

    def __init__(self, transforms: Sequence[TransformFn]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_transforms(
    split: str,
    config: Optional[dict[str, Any]] = None,
) -> Optional[Compose3D]:
    """Build train/eval 3D transforms from config."""
    config = config or {}
    split_l = split.lower()
    if split_l in {"train", "training"}:
        cfg = config.get("train", config)
        if not cfg.get("enabled", True):
            return None
        transforms: list[TransformFn] = [
            RandomFlip3D(p=float(cfg.get("flip_p", 0.5))),
            RandomRot90Planes3D(p=float(cfg.get("rot90_p", 0.5))),
        ]
        logger.info("Built 3D train transforms")
        return Compose3D(transforms)

    # Eval: identity
    logger.info("Built identity eval transforms for split=%s", split)
    return None
