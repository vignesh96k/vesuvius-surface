"""Modeling-critical analyses for Surface Detection (3D).

Answers that constrain architecture / sampling / validation / loss later:

- ignore-aware class imbalance and patch foreground hit-rate
- scroll-level domain shift (preferred validation unit)
- surface thickness via 3D distance transform on class-1
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from vesuvius_surface.eda.utils import subsample_for_hist, to_numpy
from vesuvius_surface.data.schema import LABEL_IGNORE, LABEL_SURFACE


@dataclass(frozen=True)
class ImbalanceReport3D:
    voxel_neg_pos_ratio: float
    recommended_bce_pos_weight: float
    surface_among_labeled: float
    frac_ignore: float
    patch_size: tuple[int, int, int]
    num_patches: int
    frac_patches_with_surface: float
    frac_patches_surface_ge_1pct: float
    median_patch_surface_frac: float

    def to_dict(self) -> dict[str, float | int | tuple[int, int, int]]:
        return asdict(self)


@dataclass(frozen=True)
class ThicknessStats:
    median_thickness_px: float
    mean_thickness_px: float
    p90_thickness_px: float
    recommended_patch_d: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_surface_thickness(label: np.ndarray) -> ThicknessStats:
    """Approximate local surface thickness with a 3D distance transform."""
    from scipy import ndimage

    fg = (to_numpy(label) == LABEL_SURFACE).astype(np.uint8)
    if fg.sum() == 0:
        return ThicknessStats(0.0, 0.0, 0.0, 64)
    dist = ndimage.distance_transform_edt(fg)
    widths = 2.0 * dist[fg.astype(bool)]
    widths = widths[widths > 0]
    med = float(np.median(widths)) if widths.size else 0.0
    raw = int(max(32, min(192, round(med * 8 / 16) * 16))) if med > 0 else 64
    candidates = np.array([32, 48, 64, 96, 128, 160, 192])
    rec = int(candidates[np.argmin(np.abs(candidates - raw))])
    return ThicknessStats(
        median_thickness_px=med,
        mean_thickness_px=float(widths.mean()) if widths.size else 0.0,
        p90_thickness_px=float(np.percentile(widths, 90)) if widths.size else 0.0,
        recommended_patch_d=rec,
    )


def compute_imbalance_report_3d(
    label: np.ndarray,
    patch_size: tuple[int, int, int] = (128, 128, 128),
    stride: tuple[int, int, int] = (64, 64, 64),
    min_labeled_ratio: float = 0.1,
) -> tuple[ImbalanceReport3D, np.ndarray]:
    """Voxel + patch surface-fraction statistics."""
    from vesuvius_surface.data.patching import PatchConfig3D, build_patch_index_3d, extract_patch_3d

    lab = to_numpy(label)
    labeled = lab != LABEL_IGNORE
    pos = int(((lab == LABEL_SURFACE) & labeled).sum())
    neg = int(((lab != LABEL_SURFACE) & labeled).sum())
    ratio = float(neg / max(pos, 1))

    coords = build_patch_index_3d(
        [lab],
        [tuple(lab.shape)],  # type: ignore[list-item]
        PatchConfig3D(
            patch_size=patch_size,
            stride=stride,
            min_labeled_ratio=min_labeled_ratio,
            min_foreground_ratio=0.0,
        ),
    )
    fracs = []
    for c in coords:
        p = extract_patch_3d(lab, c.z, c.y, c.x, patch_size, pad_value=LABEL_IGNORE)
        m = p != LABEL_IGNORE
        denom = max(int(m.sum()), 1)
        fracs.append(float((p[m] == LABEL_SURFACE).sum()) / denom)
    arr = np.asarray(fracs, dtype=np.float64) if fracs else np.zeros(0)
    report = ImbalanceReport3D(
        voxel_neg_pos_ratio=ratio,
        recommended_bce_pos_weight=ratio,
        surface_among_labeled=pos / max(pos + neg, 1),
        frac_ignore=float((lab == LABEL_IGNORE).mean()),
        patch_size=patch_size,
        num_patches=len(coords),
        frac_patches_with_surface=float((arr > 0).mean()) if arr.size else 0.0,
        frac_patches_surface_ge_1pct=float((arr >= 0.01).mean()) if arr.size else 0.0,
        median_patch_surface_frac=float(np.median(arr)) if arr.size else 0.0,
    )
    return report, arr


def compute_scroll_domain_shift(
    scroll_data: dict[str, dict[str, np.ndarray]],
    bins: int = 64,
) -> pd.DataFrame:
    """Pairwise scroll shift using labeled-voxel intensity histograms.

    ``scroll_data[scroll_id] = {"image": volume, "label": label}``
    """
    ids = sorted(scroll_data.keys())
    pooled = []
    for sid in ids:
        img = to_numpy(scroll_data[sid]["image"]).astype(np.float64)
        lab = to_numpy(scroll_data[sid]["label"])
        pooled.append(img[lab != LABEL_IGNORE])
    pool = np.concatenate(pooled) if pooled else np.array([0.0])
    pool = subsample_for_hist(pool)
    vrange = (float(np.percentile(pool, 1)), float(np.percentile(pool, 99)))

    stats: dict[str, dict[str, Any]] = {}
    for sid in ids:
        img = to_numpy(scroll_data[sid]["image"]).astype(np.float64)
        lab = to_numpy(scroll_data[sid]["label"])
        vals = img[lab != LABEL_IGNORE]
        hist, _ = np.histogram(
            subsample_for_hist(vals), bins=bins, range=vrange, density=True
        )
        hist = hist.astype(np.float64)
        hist /= max(hist.sum(), 1e-12)
        labeled = lab != LABEL_IGNORE
        surf = float(((lab == LABEL_SURFACE) & labeled).sum()) / max(int(labeled.sum()), 1)
        stats[sid] = {
            "mean": float(vals.mean()) if vals.size else 0.0,
            "std": float(vals.std()) if vals.size else 0.0,
            "hist": hist,
            "surface_density": surf,
        }

    rows = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            sa, sb = stats[a], stats[b]
            rows.append(
                {
                    "scroll_a": a,
                    "scroll_b": b,
                    "mean_diff": abs(sa["mean"] - sb["mean"]),
                    "std_ratio": float(
                        max(sa["std"], sb["std"]) / (min(sa["std"], sb["std"]) + 1e-6)
                    ),
                    "hist_l1": float(np.abs(sa["hist"] - sb["hist"]).sum()),
                    "surface_density_a": sa["surface_density"],
                    "surface_density_b": sb["surface_density"],
                    "surface_density_diff": abs(sa["surface_density"] - sb["surface_density"]),
                }
            )
    return pd.DataFrame(rows)


def validation_strategy_recommendation(shift_df: pd.DataFrame) -> str:
    if shift_df.empty:
        return (
            "Need ≥2 scrolls for shift estimates. Prefer spatial block validation "
            "inside a scroll — never random overlapping patches."
        )
    mean_hist = float(shift_df["hist_l1"].mean())
    mean_surf = float(shift_df["surface_density_diff"].mean())
    msg = (
        f"Mean pairwise hist L1={mean_hist:.3f}; mean surface-density gap={mean_surf:.3f}. "
    )
    if mean_hist > 0.25 or mean_surf > 0.05:
        msg += (
            "Scrolls look non-exchangeable → use leave-one-scroll-out. "
            "Random patch splits will overestimate generalization."
        )
    else:
        msg += (
            "Scrolls are relatively similar; still prefer scroll hold-out and report "
            "both OOF and pooled metrics."
        )
    return msg
