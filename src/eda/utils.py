"""Shared helpers for Surface Detection EDA."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

EDA_DIR = Path(__file__).resolve().parent
SRC_ROOT = EDA_DIR.parent
REPO_ROOT = SRC_ROOT.parent
FIGURES_DIR = EDA_DIR / "figures"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"

# Prefer src/ on path so `data` / `eda` / `utils` resolve (not a top-level data/ extract).
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

ArrayLike = Union[np.ndarray, Any]


def ensure_figures_dir(subdir: Optional[str] = None) -> Path:
    path = FIGURES_DIR if subdir is None else FIGURES_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_data_root(data_root: Optional[str | Path] = None) -> Path:
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    return root.resolve()


def save_figure(
    fig: Any,
    name: str,
    subdir: Optional[str] = None,
    dpi: int = 300,
    close: bool = True,
) -> Path:
    out_dir = ensure_figures_dir(subdir)
    filename = name if name.lower().endswith(".png") else f"{name}.png"
    path = out_dir / filename
    if hasattr(fig, "write_image"):
        try:
            fig.write_image(str(path), scale=max(dpi / 100, 1.0))
        except Exception as exc:  # pragma: no cover
            logger.warning("Plotly export failed (%s); writing HTML.", exc)
            html_path = path.with_suffix(".html")
            fig.write_html(str(html_path))
            return html_path
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        if close:
            import matplotlib.pyplot as plt

            plt.close(fig)
    logger.info("Saved figure -> %s", path)
    return path


def to_numpy(array: ArrayLike) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(array))


def stretch_for_display(
    image: np.ndarray,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    image = to_numpy(image).astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D plane, got {image.shape}")
    lo, hi = np.percentile(image, percentiles)
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image - lo) / (hi - lo)
    return (scaled * 255.0).clip(0, 255).astype(np.uint8)


def format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes:.0f} B"


def file_size_bytes(path: Path) -> int:
    path = Path(path)
    return path.stat().st_size if path.exists() else 0


def subsample_for_hist(
    array: np.ndarray,
    max_samples: int = 500_000,
    seed: int = 0,
) -> np.ndarray:
    flat = to_numpy(array).ravel()
    if flat.size <= max_samples:
        return flat
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.size, size=max_samples, replace=False)
    return flat[idx]


def set_publication_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "image.cmap": "gray",
        }
    )


def mid_planes(volume: np.ndarray) -> dict[str, np.ndarray]:
    """Return axial / coronal / sagittal mid-slices from ``(D,H,W)``."""
    vol = to_numpy(volume)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.shape}")
    d, h, w = vol.shape
    return {
        "axial": vol[d // 2],
        "coronal": vol[:, h // 2, :],
        "sagittal": vol[:, :, w // 2],
    }
