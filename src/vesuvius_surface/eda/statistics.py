"""Statistics for Surface Detection 3D volumes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from vesuvius_surface.eda.utils import subsample_for_hist, to_numpy
from vesuvius_surface.data.schema import LABEL_BG, LABEL_IGNORE, LABEL_SURFACE


@dataclass(frozen=True)
class IntensityStats:
    count: int
    min: float
    max: float
    mean: float
    std: float
    p01: float
    p50: float
    p99: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def to_series(self, name: str = "intensity") -> pd.Series:
        return pd.Series(self.to_dict(), name=name)


@dataclass(frozen=True)
class ClassStats:
    n_voxels: int
    frac_bg: float
    frac_surface: float
    frac_ignore: float
    frac_labeled: float
    surface_among_labeled: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def to_series(self, name: str = "classes") -> pd.Series:
        return pd.Series(self.to_dict(), name=name)


def compute_intensity_stats(
    volume: np.ndarray,
    label: Optional[np.ndarray] = None,
    labeled_only: bool = True,
    max_samples: int = 500_000,
) -> IntensityStats:
    """Intensity stats, optionally restricted to non-ignore voxels."""
    values = to_numpy(volume).astype(np.float64)
    if label is not None and labeled_only:
        lab = to_numpy(label)
        values = values[lab != LABEL_IGNORE]
    values = values.ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return IntensityStats(0, 0, 0, 0, 0, 0, 0, 0)
    sample = subsample_for_hist(values, max_samples=max_samples)
    p01, p50, p99 = np.percentile(sample, [1, 50, 99])
    return IntensityStats(
        count=int(values.size),
        min=float(values.min()),
        max=float(values.max()),
        mean=float(values.mean()),
        std=float(values.std()),
        p01=float(p01),
        p50=float(p50),
        p99=float(p99),
    )


def compute_class_stats(label: np.ndarray) -> ClassStats:
    lab = to_numpy(label).ravel()
    n = lab.size
    n_bg = int((lab == LABEL_BG).sum())
    n_fg = int((lab == LABEL_SURFACE).sum())
    n_ig = int((lab == LABEL_IGNORE).sum())
    n_lab = n_bg + n_fg
    return ClassStats(
        n_voxels=n,
        frac_bg=n_bg / max(n, 1),
        frac_surface=n_fg / max(n, 1),
        frac_ignore=n_ig / max(n, 1),
        frac_labeled=n_lab / max(n, 1),
        surface_among_labeled=n_fg / max(n_lab, 1),
    )


def compute_histogram(
    volume: np.ndarray,
    label: Optional[np.ndarray] = None,
    bins: int = 64,
    labeled_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    values = to_numpy(volume).astype(np.float64).ravel()
    if label is not None and labeled_only:
        lab = to_numpy(label).ravel()
        values = values[lab != LABEL_IGNORE]
    values = subsample_for_hist(values)
    if values.size == 0:
        edges = np.linspace(0, 1, bins + 1)
        return edges, np.zeros(bins)
    hist, edges = np.histogram(values, bins=bins, density=True)
    return edges, hist
