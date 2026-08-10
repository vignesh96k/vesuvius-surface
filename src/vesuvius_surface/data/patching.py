"""3D sliding-window patch extraction for Surface Detection volumes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Union

import numpy as np

from vesuvius_surface.data.schema import LABEL_IGNORE, LABEL_SURFACE

logger = logging.getLogger(__name__)

Size3D = Union[int, tuple[int, int, int]]


def as_dhw(size: Size3D) -> tuple[int, int, int]:
    """Normalize an int or ``(D, H, W)`` triple."""
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        return (size, size, size)
    if len(size) != 3:
        raise ValueError(f"size must be int or (D, H, W), got {size}")
    d, h, w = (int(size[0]), int(size[1]), int(size[2]))
    if min(d, h, w) <= 0:
        raise ValueError(f"size dimensions must be positive, got {(d, h, w)}")
    return (d, h, w)


@dataclass(frozen=True)
class PatchConfig3D:
    """Configuration for 3D sliding-window patches.

    Attributes:
        patch_size: ``(D, H, W)`` or scalar.
        stride: Step between origins; smaller than patch_size => overlap.
        pad_value: Fill for out-of-bounds image voxels.
        label_pad_value: Fill for out-of-bounds label voxels (default ignore).
        min_labeled_ratio: Minimum fraction of non-ignore voxels to keep a patch.
        min_foreground_ratio: Minimum fraction of surface (class 1) among
            labeled voxels (optional; ``0`` disables).
        ignore_index: Label value treated as unlabeled.
    """

    patch_size: Size3D = (128, 128, 128)
    stride: Size3D = (64, 64, 64)
    pad_value: float = 0.0
    label_pad_value: int = LABEL_IGNORE
    min_labeled_ratio: float = 0.1
    min_foreground_ratio: float = 0.0
    ignore_index: int = LABEL_IGNORE

    def resolved_size(self) -> tuple[int, int, int]:
        return as_dhw(self.patch_size)

    def resolved_stride(self) -> tuple[int, int, int]:
        return as_dhw(self.stride)


# Backward-compatible alias used in configs / docs.
PatchConfig = PatchConfig3D


@dataclass(frozen=True)
class PatchCoord3D:
    """Origin of a 3D patch inside a volume."""

    volume_index: int
    z: int
    y: int
    x: int


def _axis_origins(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    origins = list(range(0, length - patch + 1, stride))
    last = length - patch
    if origins[-1] != last:
        origins.append(last)
    return origins


def iter_patch_origins_3d(
    shape: tuple[int, int, int],
    patch_size: Size3D,
    stride: Size3D,
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(z, y, x)`` origins covering ``shape``."""
    d, h, w = shape
    pd, ph, pw = as_dhw(patch_size)
    sd, sh, sw = as_dhw(stride)
    for z in _axis_origins(d, pd, sd):
        for y in _axis_origins(h, ph, sh):
            for x in _axis_origins(w, pw, sw):
                yield z, y, x


def extract_patch_3d(
    volume: np.ndarray,
    z: int,
    y: int,
    x: int,
    patch_size: Size3D,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Extract a fixed-size 3D patch with constant padding."""
    if volume.ndim != 3:
        raise ValueError(f"Expected (D, H, W) volume, got {volume.shape}")
    pd, ph, pw = as_dhw(patch_size)
    d, h, w = volume.shape

    z1, y1, x1 = z, y, x
    z2, y2, x2 = z + pd, y + ph, x + pw

    src_z1, src_y1, src_x1 = max(z1, 0), max(y1, 0), max(x1, 0)
    src_z2, src_y2, src_x2 = min(z2, d), min(y2, h), min(x2, w)

    dst_z1 = src_z1 - z1
    dst_y1 = src_y1 - y1
    dst_x1 = src_x1 - x1
    dst_z2 = dst_z1 + (src_z2 - src_z1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    try:
        fill = volume.dtype.type(pad_value)
        dtype = volume.dtype
    except (TypeError, ValueError, OverflowError):
        fill = float(pad_value)
        dtype = np.float32

    out = np.full((pd, ph, pw), fill, dtype=dtype)
    if src_z2 > src_z1 and src_y2 > src_y1 and src_x2 > src_x1:
        out[dst_z1:dst_z2, dst_y1:dst_y2, dst_x1:dst_x2] = volume[
            src_z1:src_z2, src_y1:src_y2, src_x1:src_x2
        ]
    return out


def patch_validity_ratios(
    label_patch: np.ndarray,
    ignore_index: int = LABEL_IGNORE,
    min_labeled_ratio: float = 0.1,
    min_foreground_ratio: float = 0.0,
    surface_index: int = LABEL_SURFACE,
) -> tuple[bool, float, float]:
    """Return ``(keep, labeled_ratio, foreground_ratio_among_labeled)``."""
    total = label_patch.size
    if total == 0:
        return False, 0.0, 0.0
    labeled = label_patch != ignore_index
    labeled_ratio = float(labeled.mean())
    if labeled_ratio < min_labeled_ratio:
        return False, labeled_ratio, 0.0
    n_lab = int(labeled.sum())
    if n_lab == 0:
        return False, labeled_ratio, 0.0
    fg_ratio = float((label_patch[labeled] == surface_index).mean())
    if fg_ratio < min_foreground_ratio:
        return False, labeled_ratio, fg_ratio
    return True, labeled_ratio, fg_ratio


def build_patch_index_3d(
    label_volumes: Sequence[Optional[np.ndarray]],
    volume_shapes: Sequence[tuple[int, int, int]],
    config: PatchConfig3D,
) -> list[PatchCoord3D]:
    """Build a global list of valid 3D patch coordinates.

    When a label is ``None`` (test), all geometric windows are kept.
    """
    size = config.resolved_size()
    stride = config.resolved_stride()
    coords: list[PatchCoord3D] = []

    for vol_idx, shape in enumerate(volume_shapes):
        label = label_volumes[vol_idx]
        kept = total = 0
        for z, y, x in iter_patch_origins_3d(shape, size, stride):
            total += 1
            if label is None:
                coords.append(PatchCoord3D(vol_idx, z, y, x))
                kept += 1
                continue
            lab_p = extract_patch_3d(
                label, z, y, x, size, pad_value=config.label_pad_value
            )
            ok, _, _ = patch_validity_ratios(
                lab_p,
                ignore_index=config.ignore_index,
                min_labeled_ratio=config.min_labeled_ratio,
                min_foreground_ratio=config.min_foreground_ratio,
            )
            if ok:
                coords.append(PatchCoord3D(vol_idx, z, y, x))
                kept += 1
        logger.info(
            "Volume %d shape=%s: kept %d / %d patches",
            vol_idx,
            shape,
            kept,
            total,
        )

    if not coords:
        raise RuntimeError(
            "No valid 3D patches found. Lower min_labeled_ratio / "
            "min_foreground_ratio or check labels."
        )
    logger.info("Built 3D patch index with %d total patches", len(coords))
    return coords
