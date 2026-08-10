"""1st-place post-processing chain (baseline / control).

Faithful reimplementation of the pipeline described in:

    https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/
    writeups/1st-place-solution-for-the-vesuvius-challenge-su

Operational order (per writeup text; per-sheet ops on each connected component):

  1. Remove components smaller than ``min_component_size`` (writeup: 20k)
  2. Per sheet: binary_closing with spherical footprint (radius 3)
  3. Per sheet: height-map gap patching (discard if holes increase)
  4. Per sheet: 1-voxel hole plugging via 2x2x2 LUT
  5. Global ``binary_fill_holes``

Height-map / LUT mechanics follow the public reimplementation in
``bshepp/volumen`` (``src_v2/postprocess.py``), which matches the writeup
description. Defaults use the writeup's 20k component threshold rather than
volumen's 500.

This module is the **control**. Novelty (merge cuts, offline metric tuning)
belongs in separate modules that call into this baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    find_objects,
    generate_binary_structure,
    label as cc_label,
)

# Cumulative stage names for ablation tables (operational order).
FIRST_PLACE_STAGES: tuple[str, ...] = (
    "raw",
    "remove_small",
    "closing",
    "patch",
    "plug",
    "fill",
)


@dataclass
class PostprocessConfig:
    """Knobs for the 1st-place baseline. Defaults match the writeup."""

    min_component_size: int = 20_000
    closing_radius: int = 3
    connectivity: int = 26
    enable_closing: bool = True
    enable_patching: bool = True
    enable_hole_plugging: bool = True
    enable_fill_holes: bool = True
    # Surface class id when the prediction is a multi-class label map.
    surface_label: int = 1
    # Softmax / probability threshold when input is continuous.
    threshold: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


def binarize_prediction(
    volume: np.ndarray,
    *,
    threshold: float = 0.5,
    surface_label: int = 1,
) -> np.ndarray:
    """Convert nnU-Net output (label map, probs, or binary) to a uint8 mask."""
    arr = np.asarray(volume)
    if arr.ndim == 4:
        # (C, D, H, W) softmax — take surface channel if present.
        if arr.shape[0] > surface_label:
            arr = arr[surface_label]
        else:
            arr = arr.argmax(axis=0)
    if np.issubdtype(arr.dtype, np.floating):
        return (arr >= threshold).astype(np.uint8)
    # Integer label map: keep surface class only.
    uniq = set(np.unique(arr).tolist())
    if uniq <= {0, 1}:
        return (arr > 0).astype(np.uint8)
    return (arr == surface_label).astype(np.uint8)


def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return generate_binary_structure(3, 1)
    if connectivity == 18:
        return generate_binary_structure(3, 2)
    if connectivity == 26:
        return generate_binary_structure(3, 3)
    raise ValueError(f"connectivity must be 6, 18, or 26; got {connectivity}")


def make_ball_footprint(radius: int) -> np.ndarray:
    zz, yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1, -radius : radius + 1]
    return (zz**2 + yy**2 + xx**2) <= radius**2


def _pad_slices(sl, shape, pad: int):
    return tuple(
        slice(max(0, s.start - pad), min(dim, s.stop + pad)) for s, dim in zip(sl, shape)
    )


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 20_000,
    connectivity: int = 26,
) -> np.ndarray:
    struct = _structure(connectivity)
    labeled, n = cc_label(mask.astype(np.uint8), structure=struct)
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.zeros_like(mask, dtype=np.uint8)
    for i, size in enumerate(sizes, 1):
        if size >= min_size:
            keep[labeled == i] = 1
    return keep


def _count_internal_holes(mask: np.ndarray) -> int:
    inv = 1 - mask.astype(np.uint8)
    labeled, n = cc_label(inv, structure=generate_binary_structure(3, 1))
    if n == 0:
        return 0
    border = np.zeros(n + 1, dtype=bool)
    border[labeled[0]] = True
    border[labeled[-1]] = True
    border[labeled[:, 0]] = True
    border[labeled[:, -1]] = True
    border[labeled[:, :, 0]] = True
    border[labeled[:, :, -1]] = True
    border[0] = True
    return int(n - border[1:].sum())


def height_map_patch_crop(crop: np.ndarray) -> np.ndarray:
    """Fill projected gaps in a single-sheet crop; discard if holes increase."""
    if crop.sum() == 0:
        return crop

    best_axis, best_area = 0, 0
    for axis in range(3):
        area = int(crop.max(axis=axis).sum())
        if area > best_area:
            best_area = area
            best_axis = axis

    crop_t = np.moveaxis(crop, best_axis, 0)
    depth, height, width = crop_t.shape
    depth_coords = np.arange(depth, dtype=np.float32).reshape(depth, 1, 1)
    valid_3d = crop_t.astype(bool)
    count_map = valid_3d.sum(axis=0)
    has_voxels = count_map > 0

    height_map = np.full((height, width), np.nan, dtype=np.float32)
    thick_map = np.zeros((height, width), dtype=np.float32)
    depth_sum = (depth_coords * valid_3d).sum(axis=0)
    height_map[has_voxels] = depth_sum[has_voxels] / count_map[has_voxels]
    thick_map[has_voxels] = count_map[has_voxels]

    filled_proj = binary_fill_holes(has_voxels)
    gap_mask = filled_proj & ~has_voxels
    if not gap_mask.any():
        return crop

    holes_before = _count_internal_holes(crop)

    fill_row_h = np.full((height, width), np.nan, dtype=np.float32)
    fill_row_t = np.full((height, width), np.nan, dtype=np.float32)
    for r in range(height):
        valid_cols = np.where(has_voxels[r])[0]
        gap_cols = np.where(gap_mask[r])[0]
        if len(valid_cols) >= 2 and len(gap_cols) > 0:
            fill_row_h[r, gap_cols] = np.interp(
                gap_cols, valid_cols, height_map[r, valid_cols]
            )
            fill_row_t[r, gap_cols] = np.interp(
                gap_cols, valid_cols, thick_map[r, valid_cols]
            )

    fill_col_h = np.full((height, width), np.nan, dtype=np.float32)
    fill_col_t = np.full((height, width), np.nan, dtype=np.float32)
    for c in range(width):
        valid_rows = np.where(has_voxels[:, c])[0]
        gap_rows = np.where(gap_mask[:, c])[0]
        if len(valid_rows) >= 2 and len(gap_rows) > 0:
            fill_col_h[gap_rows, c] = np.interp(
                gap_rows, valid_rows, height_map[valid_rows, c]
            )
            fill_col_t[gap_rows, c] = np.interp(
                gap_rows, valid_rows, thick_map[valid_rows, c]
            )

    not_valid = ~has_voxels
    row_dist = distance_transform_edt(not_valid, sampling=[1e6, 1])
    col_dist = distance_transform_edt(not_valid, sampling=[1, 1e6])

    gap_r, gap_c = np.where(gap_mask)
    hr, hc = fill_row_h[gap_r, gap_c], fill_col_h[gap_r, gap_c]
    tr, tc = fill_row_t[gap_r, gap_c], fill_col_t[gap_r, gap_c]
    dr = np.maximum(row_dist[gap_r, gap_c], 1e-6)
    dc = np.maximum(col_dist[gap_r, gap_c], 1e-6)

    valid_r, valid_c = ~np.isnan(hr), ~np.isnan(hc)
    both = valid_r & valid_c
    only_r, only_c = valid_r & ~valid_c, valid_c & ~valid_r
    wr = np.where(both, 1.0 / dr, 0.0)
    wc = np.where(both, 1.0 / dc, 0.0)
    w_total = np.maximum(wr + wc, 1e-12)

    h_avg = np.where(
        both,
        (np.nan_to_num(hr) * wr + np.nan_to_num(hc) * wc) / w_total,
        np.where(only_r, hr, np.where(only_c, hc, np.nan)),
    )
    t_avg = np.where(
        both,
        (np.nan_to_num(tr) * wr + np.nan_to_num(tc) * wc) / w_total,
        np.where(only_r, tr, np.where(only_c, tc, 0.0)),
    )

    patched_t = crop_t.copy()
    for idx in range(len(gap_r)):
        h_val = h_avg[idx]
        if np.isnan(h_val):
            continue
        r, c = int(gap_r[idx]), int(gap_c[idx])
        center = int(round(float(h_val)))
        half = max(0, int(round(float(t_avg[idx]) / 2)))
        z0 = max(0, center - half)
        z1 = min(depth - 1, center + half)
        patched_t[z0 : z1 + 1, r, c] = 1

    patched_3d = np.moveaxis(patched_t, 0, best_axis)
    if _count_internal_holes(patched_3d) > holes_before:
        return crop
    return patched_3d.astype(np.uint8)


_HOLE_PLUG_LUT: Optional[np.ndarray] = None


def _build_hole_plug_lut() -> np.ndarray:
    """256-entry LUT: add bridges for face-diagonal gaps in a 2x2x2 cube."""
    face_diags = [
        (0, 3, 1, 2),
        (1, 2, 0, 3),
        (4, 7, 5, 6),
        (5, 6, 4, 7),
        (0, 5, 1, 4),
        (1, 4, 0, 5),
        (2, 7, 3, 6),
        (3, 6, 2, 7),
        (0, 6, 2, 4),
        (2, 4, 0, 6),
        (1, 7, 3, 5),
        (3, 5, 1, 7),
    ]
    lut = np.zeros(256, dtype=np.uint8)
    for pattern in range(256):
        add = 0
        for fa, fb, g1, g2 in face_diags:
            if (
                ((pattern >> fa) & 1)
                and ((pattern >> fb) & 1)
                and not ((pattern >> g1) & 1)
                and not ((pattern >> g2) & 1)
            ):
                add |= 1 << g1
        lut[pattern] = add
    return lut


def plug_holes_lut(mask: np.ndarray, max_iterations: int = 5) -> np.ndarray:
    global _HOLE_PLUG_LUT
    if _HOLE_PLUG_LUT is None:
        _HOLE_PLUG_LUT = _build_hole_plug_lut()
    lut = _HOLE_PLUG_LUT
    result = mask.astype(np.uint8).copy()
    depth, height, width = result.shape
    if depth < 2 or height < 2 or width < 2:
        return result

    offsets = [(dz, dy, dx) for dz in range(2) for dy in range(2) for dx in range(2)]
    for _ in range(max_iterations):
        pattern = np.zeros((depth - 1, height - 1, width - 1), dtype=np.uint8)
        for bit, (dz, dy, dx) in enumerate(offsets):
            pattern |= (
                result[dz : depth - 1 + dz, dy : height - 1 + dy, dx : width - 1 + dx] << bit
            )
        additions = lut[pattern]
        if not additions.any():
            break
        for bit, (dz, dy, dx) in enumerate(offsets):
            add_bit = ((additions >> bit) & 1).astype(np.uint8)
            result[dz : depth - 1 + dz, dy : height - 1 + dy, dx : width - 1 + dx] |= add_bit
    return result


def apply_first_place(
    prediction: np.ndarray,
    config: Optional[PostprocessConfig] = None,
    *,
    through_stage: str = "fill",
) -> np.ndarray:
    """Run the 1st-place chain, optionally stopping after ``through_stage``.

    ``through_stage`` is one of :data:`FIRST_PLACE_STAGES`. Use this for
    cumulative ablation tables that match the writeup's reporting style.
    """
    cfg = config or PostprocessConfig()
    if through_stage not in FIRST_PLACE_STAGES:
        raise ValueError(
            f"through_stage must be one of {FIRST_PLACE_STAGES}, got {through_stage!r}"
        )
    stop_at = FIRST_PLACE_STAGES.index(through_stage)

    mask = binarize_prediction(
        prediction, threshold=cfg.threshold, surface_label=cfg.surface_label
    )
    if stop_at == 0:
        return mask

    mask = remove_small_components(mask, cfg.min_component_size, cfg.connectivity)
    if stop_at == 1:
        return mask

    struct = _structure(cfg.connectivity)
    labeled, n = cc_label(mask, structure=struct)
    if n == 0:
        return mask

    slices = find_objects(labeled)
    footprint = make_ball_footprint(cfg.closing_radius) if cfg.closing_radius > 0 else None
    pad = cfg.closing_radius if cfg.enable_closing and cfg.closing_radius > 0 else 0
    result = np.zeros_like(mask, dtype=np.uint8)

    do_closing = cfg.enable_closing and stop_at >= 2
    do_patch = cfg.enable_patching and stop_at >= 3
    do_plug = cfg.enable_hole_plugging and stop_at >= 4

    for comp_id, sl in enumerate(slices, 1):
        if sl is None:
            continue
        padded_sl = _pad_slices(sl, mask.shape, pad)
        crop = (labeled[padded_sl] == comp_id).astype(np.uint8)

        if do_closing and footprint is not None:
            crop = binary_closing(crop, structure=footprint).astype(np.uint8)
        if do_patch:
            crop = height_map_patch_crop(crop)
        if do_plug:
            crop = plug_holes_lut(crop)

        result[padded_sl] |= crop

    if stop_at < 5 or not cfg.enable_fill_holes:
        return result
    return binary_fill_holes(result).astype(np.uint8)


__all__ = [
    "FIRST_PLACE_STAGES",
    "PostprocessConfig",
    "apply_first_place",
    "binarize_prediction",
    "height_map_patch_crop",
    "make_ball_footprint",
    "plug_holes_lut",
    "remove_small_components",
]
