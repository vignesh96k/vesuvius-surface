"""Publication-quality visualizations for Surface Detection EDA."""

from __future__ import annotations

import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from eda.statistics import compute_histogram
from eda.utils import mid_planes, save_figure, set_publication_style, stretch_for_display, to_numpy
from data.schema import LABEL_IGNORE, LABEL_SURFACE

logger = logging.getLogger(__name__)
set_publication_style()


def _overlay_label_rgb(
    plane: np.ndarray,
    label_plane: np.ndarray,
    alpha_surface: float = 0.45,
    alpha_ignore: float = 0.25,
) -> np.ndarray:
    import cv2

    base = stretch_for_display(plane)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    lab = to_numpy(label_plane)
    surf = lab == LABEL_SURFACE
    ign = lab == LABEL_IGNORE
    out = rgb.copy()
    if surf.any():
        color = np.zeros_like(rgb)
        color[surf] = (228, 87, 46)
        out = cv2.addWeighted(color, alpha_surface, out, 1.0 - alpha_surface, 0)
    if ign.any():
        color = np.zeros_like(rgb)
        color[ign] = (100, 149, 237)
        out = cv2.addWeighted(color, alpha_ignore, out, 1.0 - alpha_ignore, 0)
    return out


def plot_volume_inventory(
    summary_df: Any,
    save_name: Optional[str] = "01_volume_inventory",
) -> plt.Figure:
    import pandas as pd

    df = pd.DataFrame(summary_df) if not isinstance(summary_df, pd.DataFrame) else summary_df
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].bar(df["volume_id"], df["shape_d"] * df["shape_h"] * df["shape_w"] / 1e6, color="#3B6B7A")
    axes[0].set_title("Volume size (Mvoxels)")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(df["volume_id"], df["frac_surface"] * 100, color="#E4572E")
    axes[1].set_title("Surface % of volume")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(df["volume_id"], df["frac_ignore"] * 100, color="#5B8FA8")
    axes[2].set_title("Ignore % of volume")
    axes[2].tick_params(axis="x", rotation=45)

    fig.suptitle("Volume inventory", y=1.02)
    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name, close=False)
    return fig


def plot_orthogonal_views(
    image: np.ndarray,
    label: Optional[np.ndarray] = None,
    volume_id: str = "",
    save_name: Optional[str] = "02_orthogonal_views",
) -> plt.Figure:
    img_planes = mid_planes(image)
    lab_planes = mid_planes(label) if label is not None else None
    names = ["axial", "coronal", "sagittal"]
    cols = 2 if label is not None else 1
    fig, axes = plt.subplots(3, cols, figsize=(4.2 * cols, 10.5))
    if cols == 1:
        axes = np.expand_dims(axes, 1)
    for r, name in enumerate(names):
        axes[r, 0].imshow(stretch_for_display(img_planes[name]), cmap="gray")
        axes[r, 0].set_title(f"{name} CT")
        axes[r, 0].axis("off")
        if lab_planes is not None:
            axes[r, 1].imshow(_overlay_label_rgb(img_planes[name], lab_planes[name]))
            axes[r, 1].set_title(f"{name} overlay")
            axes[r, 1].axis("off")
    fig.suptitle(f"Orthogonal mid-planes — {volume_id}".strip(" —"))
    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name, close=False)
    return fig


def plot_class_histogram(
    label: np.ndarray,
    volume_id: str = "",
    save_name: Optional[str] = "03_class_histogram",
) -> plt.Figure:
    lab = to_numpy(label).ravel()
    counts = [int((lab == v).sum()) for v in (0, 1, 2)]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.bar(["bg (0)", "surface (1)", "ignore (2)"], counts, color=["#888888", "#E4572E", "#5B8FA8"])
    ax.set_ylabel("voxels")
    ax.set_title(f"Label distribution — {volume_id}".strip(" —"))
    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name, close=False)
    return fig


def plot_intensity_hist(
    image: np.ndarray,
    label: Optional[np.ndarray] = None,
    volume_id: str = "",
    save_name: Optional[str] = "04_intensity_hist",
) -> plt.Figure:
    edges, hist = compute_histogram(image, label=label, labeled_only=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(centers, hist, color="#3B6B7A", lw=1.8)
    ax.fill_between(centers, hist, color="#3B6B7A", alpha=0.2)
    ax.set_xlabel("intensity")
    ax.set_ylabel("density")
    ax.set_title(f"Intensity (labeled voxels) — {volume_id}".strip(" —"))
    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name, close=False)
    return fig


def plot_scroll_summary(
    scroll_df: Any,
    save_name: Optional[str] = "05_scroll_summary",
) -> plt.Figure:
    import pandas as pd

    df = pd.DataFrame(scroll_df) if not isinstance(scroll_df, pd.DataFrame) else scroll_df
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(df["scroll_id"], df["n_volumes"], color="#5B4B8A")
    ax.set_ylabel("# volumes")
    ax.set_title("Volumes per scroll_id")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name, close=False)
    return fig


def plot_patch_montage(
    patches: list[dict[str, Any]],
    save_name: Optional[str] = "06_patch_montage",
) -> plt.Figure:
    """Show mid-axial plane of several 3D patches."""
    n = len(patches)
    fig, axes = plt.subplots(n, 2, figsize=(7.0, 2.6 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for i, sample in enumerate(patches):
        img = to_numpy(sample["image"])
        if img.ndim == 4:
            img = img[0]
        mid = img[img.shape[0] // 2]
        axes[i, 0].imshow(stretch_for_display(mid), cmap="gray")
        axes[i, 0].set_ylabel(
            f"{sample.get('volume_id', i)}\nz={sample.get('z','?')}",
            fontsize=8,
        )
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        if "label" in sample:
            lab = to_numpy(sample["label"])
            axes[i, 1].imshow(_overlay_label_rgb(mid, lab[lab.shape[0] // 2]))
        else:
            axes[i, 1].imshow(stretch_for_display(mid), cmap="gray")
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 0].set_title("patch CT")
            axes[i, 1].set_title("overlay")
    fig.suptitle("Sampled 3D patches (axial mid-plane)")
    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name, close=False)
    return fig


def plot_imbalance_hist(
    fracs: np.ndarray,
    report: Any,
    volume_id: str = "",
    save_name: Optional[str] = "07_patch_imbalance",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    if fracs.size:
        axes[0].hist(fracs, bins=40, color="#E4572E", alpha=0.85)
    axes[0].set_xlabel("surface fraction (labeled voxels)")
    axes[0].set_title("Per-patch surface fraction")
    axes[1].axis("off")
    text = (
        f"neg:pos            {report.voxel_neg_pos_ratio:.1f}:1\n"
        f"BCE pos_weight     {report.recommended_bce_pos_weight:.1f}\n"
        f"ignore frac        {report.frac_ignore:.3f}\n"
        f"patches            {report.num_patches}\n"
        f"with surface       {report.frac_patches_with_surface:.3f}\n"
        f">=1% surface       {report.frac_patches_surface_ge_1pct:.3f}\n"
        f"median surf frac   {report.median_patch_surface_frac:.4f}"
    )
    axes[1].text(
        0.05,
        0.95,
        text,
        va="top",
        family="monospace",
        fontsize=11,
        transform=axes[1].transAxes,
        bbox={"facecolor": "#F4F4F4", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.4"},
    )
    fig.suptitle(f"Imbalance — {volume_id}".strip(" —"))
    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name, close=False)
    return fig


def plot_domain_shift(
    shift_df: Any,
    save_name: Optional[str] = "08_scroll_domain_shift",
) -> plt.Figure:
    import pandas as pd

    df = pd.DataFrame(shift_df) if not isinstance(shift_df, pd.DataFrame) else shift_df
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "Need ≥2 scrolls for domain-shift analysis.", ha="center")
        if save_name:
            save_figure(fig, save_name, close=False)
        return fig
    ids = sorted(set(df["scroll_a"]).union(set(df["scroll_b"])))
    n = len(ids)
    idx = {s: i for i, s in enumerate(ids)}
    mat = np.zeros((n, n))
    for _, row in df.iterrows():
        i, j = idx[row["scroll_a"]], idx[row["scroll_b"]]
        mat[i, j] = mat[j, i] = row["hist_l1"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat, cmap="magma")
    ax.set_xticks(list(range(n)))
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels(ids)
    ax.set_title("Scroll pairwise hist L1")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save_name:
        save_figure(fig, save_name, close=False)
    return fig
