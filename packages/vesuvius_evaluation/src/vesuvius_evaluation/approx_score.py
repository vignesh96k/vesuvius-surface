"""Fast, dependency-light approximation of the Vesuvius Surface Detection competition metric.

Ported verbatim from jirkaborovec/replicate-lb-score-topology-aware-3d-surface-seg
(https://www.kaggle.com/code/jirkaborovec/replicate-lb-score-topology-aware-3d-surface-seg),
the same author as the training notebook this repo's sibling `baselinerun/` reproduces.

IMPORTANT -- this is an approximation, not the official scorer. It implements the same
formula (Score = 0.30*TopoScore + 0.35*SurfaceDice@tau + 0.35*VOI_score) and the same general
concepts, but its own Betti-number computation (Euler-characteristic + connected-component
counting, see `compute_betti_numbers`) is algorithmically different from the official
topometrics package's compiled Betti-Matching-3D persistent-homology matching -- so TopoScore
values (and therefore the combined score) will NOT exactly match `official_score.py`. Use this
for fast iteration during development; use `official_score.py` (evaluation.official_score,
requires the vesuvius_eval conda env) for numbers you actually trust.

Only two changes from the source notebook: functions are unchanged, but the module-level
demo/test-case code (notebook cells 16-17) is not included here since it's a script entry
point, not reusable library code -- see scripts/evaluate.py for the equivalent CLI driver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc3d_label
from skimage.measure import euler_number

TAU = 2.0  # Surface Dice tolerance
ALPHA_VOI = 0.3  # VOI conversion parameter
BETTI_WEIGHTS = {0: 0.34, 1: 0.33, 2: 0.33}  # TopoScore weights
W_TOPO = 0.30  # Final score weight for TopoScore
W_SURFACE_DICE = 0.35  # Final score weight for SurfaceDice
W_VOI = 0.35  # Final score weight for VOI

LABEL_BACKGROUND = 0
LABEL_SURFACE = 1
LABEL_UNLABELED = 2


def load_volume(path: Union[str, Path]) -> np.ndarray:
    """Load 3D volume from TIF file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")
    volume = tifffile.imread(str(path))
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    return volume.astype(np.uint8)


def preprocess(
    pred: np.ndarray, gt: np.ndarray, ignore_label: int = LABEL_UNLABELED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binarize volumes and create valid mask (excluding ignore_label)."""
    valid_mask = gt != ignore_label
    pred_bin = ((pred == LABEL_SURFACE) & valid_mask).astype(np.uint8)
    gt_bin = ((gt == LABEL_SURFACE) & valid_mask).astype(np.uint8)
    return pred_bin, gt_bin, valid_mask


def extract_surface(volume: np.ndarray) -> np.ndarray:
    """Extract boundary voxels using 6-connectivity erosion."""
    if np.sum(volume) == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    struct_6 = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(volume.astype(bool), structure=struct_6)
    return volume.astype(np.uint8) - eroded.astype(np.uint8)


def compute_surface_dice(pred: np.ndarray, gt: np.ndarray, tau: float = TAU) -> float:
    """Surface Dice @ tau: fraction of surface points within tolerance.

    Edge cases: both empty -> 1.0, one empty -> 0.0
    """
    pred_empty = np.sum(pred) == 0
    gt_empty = np.sum(gt) == 0

    if pred_empty and gt_empty:
        return 1.0
    if pred_empty or gt_empty:
        return 0.0

    pred_surface = extract_surface(pred)
    gt_surface = extract_surface(gt)

    n_pred = np.sum(pred_surface)
    n_gt = np.sum(gt_surface)

    if n_pred == 0 or n_gt == 0:
        return 0.0

    # Distance from pred surface to GT SURFACE (not volume!)
    dist_to_gt_surface = distance_transform_edt(~gt_surface.astype(bool))
    pred_matched = np.sum(dist_to_gt_surface[pred_surface > 0] <= tau)

    # Distance from GT surface to pred SURFACE (not volume!)
    dist_to_pred_surface = distance_transform_edt(~pred_surface.astype(bool))
    gt_matched = np.sum(dist_to_pred_surface[gt_surface > 0] <= tau)

    return float((pred_matched + gt_matched) / (n_pred + n_gt))


def compute_voi(pred_labels: np.ndarray, gt_labels: np.ndarray) -> Tuple[float, float, float]:
    """Compute Variation of Information: (VOI_split, VOI_merge, VOI_total)."""
    n = pred_labels.size
    if n == 0:
        return 0.0, 0.0, 0.0

    max_pred = int(np.max(pred_labels)) + 1
    max_gt = int(np.max(gt_labels)) + 1

    contingency = np.zeros((max_pred, max_gt), dtype=np.float64)
    for p, g in zip(pred_labels.ravel(), gt_labels.ravel()):
        contingency[p, g] += 1

    p_ij = contingency / n
    p_i = np.sum(p_ij, axis=1)
    p_j = np.sum(p_ij, axis=0)

    voi_split = 0.0
    voi_merge = 0.0

    for i in range(max_pred):
        for j in range(max_gt):
            if p_ij[i, j] > 0:
                if p_i[i] > 0:
                    voi_split -= p_ij[i, j] * np.log2(p_ij[i, j] / p_i[i])
                if p_j[j] > 0:
                    voi_merge -= p_ij[i, j] * np.log2(p_ij[i, j] / p_j[j])

    return float(voi_split), float(voi_merge), float(voi_split + voi_merge)


def compute_voi_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """VOI Score = 1 / (1 + alpha * VOI_total). Uses 26-connectivity."""
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 1.0
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return 0.0

    struct_26 = ndimage.generate_binary_structure(3, 3)
    pred_labels, _ = cc3d_label(pred.astype(bool), structure=struct_26)
    gt_labels, _ = cc3d_label(gt.astype(bool), structure=struct_26)

    union_fg = pred.astype(bool) | gt.astype(bool)
    _, _, voi_total = compute_voi(pred_labels[union_fg], gt_labels[union_fg])

    return float(1.0 / (1.0 + ALPHA_VOI * voi_total))


def compute_betti_numbers(volume: np.ndarray) -> Tuple[int, int, int]:
    """Compute Betti numbers (b0, b1, b2) for 3D binary volume."""
    if np.sum(volume) == 0:
        return (0, 0, 0)

    vol_bool = volume.astype(bool)
    struct_26 = ndimage.generate_binary_structure(3, 3)
    _, beta0 = cc3d_label(vol_bool, structure=struct_26)

    # Cavities (enclosed background components)
    struct_6 = ndimage.generate_binary_structure(3, 1)
    bg_labels, n_bg = cc3d_label(~vol_bool, structure=struct_6)

    boundary_labels = set()
    for face in [
        bg_labels[0, :, :],
        bg_labels[-1, :, :],
        bg_labels[:, 0, :],
        bg_labels[:, -1, :],
        bg_labels[:, :, 0],
        bg_labels[:, :, -1],
    ]:
        boundary_labels.update(face.ravel())
    boundary_labels.discard(0)
    beta2 = n_bg - len(boundary_labels)

    # beta1 from Euler characteristic
    chi = euler_number(vol_bool, connectivity=3)
    beta1 = max(0, beta0 - chi + beta2)

    return (int(beta0), int(beta1), int(beta2))


def compute_betti_f1(pred_betti: int, gt_betti: int) -> float:
    """F1 score for Betti number matching."""
    if pred_betti == 0 and gt_betti == 0:
        return 1.0
    if pred_betti == 0 or gt_betti == 0:
        return 0.0
    matched = min(pred_betti, gt_betti)
    precision = matched / pred_betti
    recall = matched / gt_betti
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_topo_score(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, Tuple, Tuple]:
    """TopoScore: weighted Betti F1. Returns (score, pred_betti, gt_betti)."""
    pred_betti = compute_betti_numbers(pred)
    gt_betti = compute_betti_numbers(gt)

    f1_scores = {}
    active_weights = {}

    for k in range(3):
        pb, gb = pred_betti[k], gt_betti[k]
        if pb > 0 or gb > 0:
            f1_scores[k] = compute_betti_f1(pb, gb)
            active_weights[k] = BETTI_WEIGHTS[k]

    if not active_weights:
        return 1.0, pred_betti, gt_betti

    total_weight = sum(active_weights.values())
    score = sum(f1_scores[k] * active_weights[k] for k in active_weights) / total_weight

    return float(score), pred_betti, gt_betti


def compute_score(
    pred: np.ndarray,
    gt: np.ndarray,
    tau: float = TAU,
    ignore_label: int = LABEL_UNLABELED,
    verbose: bool = False,
) -> Dict:
    """Compute approximate Vesuvius Surface Detection competition score.

    Score = 0.30 x TopoScore + 0.35 x SurfaceDice@tau + 0.35 x VOI_score
    """
    pred_bin, gt_bin, _ = preprocess(pred, gt, ignore_label)

    surface_dice = compute_surface_dice(pred_bin, gt_bin, tau)
    voi_score = compute_voi_score(pred_bin, gt_bin)
    topo_score, pred_betti, gt_betti = compute_topo_score(pred_bin, gt_bin)

    score = W_TOPO * topo_score + W_SURFACE_DICE * surface_dice + W_VOI * voi_score

    result = {
        "score": round(score, 6),
        "surface_dice": round(surface_dice, 6),
        "voi_score": round(voi_score, 6),
        "topo_score": round(topo_score, 6),
        "pred_betti": pred_betti,
        "gt_betti": gt_betti,
    }

    if verbose:
        print(f"SurfaceDice@{tau}: {surface_dice:.6f}")
        print(f"VOI_score:        {voi_score:.6f}")
        print(f"TopoScore:        {topo_score:.6f}")
        print(f"  Pred beta: {pred_betti}, GT beta: {gt_betti}")
        print(f">>> SCORE:        {score:.6f}")

    return result


def score_from_files(
    pred_path: Union[str, Path],
    gt_path: Union[str, Path],
    tau: float = TAU,
    verbose: bool = True,
) -> Dict:
    """Compute score directly from TIF file paths."""
    pred = load_volume(pred_path)
    gt = load_volume(gt_path)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    return compute_score(pred, gt, tau=tau, verbose=verbose)
