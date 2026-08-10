"""Novelty post-processing layer: metric-guided unmerge.

Starts FROM the 1st-place control output (:func:`postprocess.first_place.apply_first_place`),
detects candidate merge bridges (thin necks joining two thicker sheet-like masses inside one
connected component), proposes cuts that sever them, and accepts or rejects each volume's
cut-set only if the OFFICIAL competition metric improves on that volume. This module never
reimplements the metric -- it calls straight into `evaluation.metric_adapter.score_pair`, the
same function the evaluation harness uses.

Motivation (see research_log.md sections 8 and 14): the 1st-place control chain repairs holes
*inside* components (closing / height-map patch / hole plug / fill) but never severs a bridge
fused between two sheets -- it is merge-blind by construction, and the recorded ablation on
scroll 35360's holdout cases confirms it: voi_merge sits at 1.1230 (raw) through 1.1251 (fill)
across every control stage, essentially unchanged, while surface_dice and topo_score move.
This layer targets exactly that residual.

Bridge-detection method (v1, deliberately simple, per the task's explicit "morphological
thickness heuristics are fine for v1"):

  1. Erode each connected component by a ball of radius `erosion_radius`. A thin neck vanishes
     under erosion while the two thicker masses it joins survive as separate seed blobs.
  2. If erosion splits one component into >=2 seeds each >= `min_seed_size` voxels, the
     original component is a merge candidate.
  3. Partition the ORIGINAL (uneroded) component by nearest surviving seed -- a Euclidean
     nearest-seed assignment via `distance_transform_edt(..., return_indices=True)`. This is a
     Voronoi tessellation seeded by the eroded pieces: the simplest valid stand-in for a full
     watershed, and it needs nothing beyond scipy (already a hard dependency here).
  4. Remove the `cut_width`-voxel-wide boundary between differently-labeled partitions -- this
     is the actual cut. The partition boundary is by construction a full separating surface, so
     removing it disconnects the two sheets.
  5. Drop resulting pieces smaller than `min_piece_size` (cutting can shear off small debris at
     the neck; treat it the same way `remove_small_components` treats fragments).

Accept/reject is PER VOLUME, not per cut: every candidate cut in a volume is applied at once
(candidates are spatially disjoint components, so they don't interact), the volume is scored
once, and the cut version is kept only if `score` improves by at least `min_score_delta`.
Scoring one volume already costs ~60-90s (Betti-matching dominates), so per-cut scoring inside
a volume is not affordable -- per-volume is the coarsest granularity that is still a direct,
honest reading of "accept a cut only if the official metric improves on that volume."
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    find_objects,
    label as cc_label,
)

from vesuvius_surface.postprocess.first_place import _pad_slices, _structure, make_ball_footprint

logger = logging.getLogger(__name__)

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

# Cumulative stage names for the unmerge ablation (control -> proposed -> gated).
UNMERGE_STAGES: tuple[str, ...] = ("control", "unmerge_proposed", "unmerge_accepted")


@dataclass
class UnmergeConfig:
    """Knobs for bridge detection and cut acceptance.

    ``erosion_radius`` and ``min_seed_size`` are calibrated against measured sheet
    geometry, not guessed: a distance-transform check on real m7_holdout control
    masks shows even the largest healthy component's half-thickness maxes out
    around 2.0 voxels (median 1.0) -- these are inherently thin sheets, not
    blobs. ``erosion_radius=2`` erodes a healthy sheet away entirely (found zero
    candidates across 5 real cases); ``erosion_radius=1`` is the largest radius
    that still leaves real sheet material as seeds while still stripping
    sub-voxel-scale bridges, and finds real candidates -- 9 in a case from
    scroll 35360, the exact scroll research_log.md flags for merge skew.
    ``min_score_delta=0.0`` means a cut is accepted on any non-negative
    improvement; raise it to require a real margin before keeping a cut."""

    erosion_radius: int = 1
    min_seed_size: int = 100
    min_piece_size: int = 100
    cut_width: int = 1
    connectivity: int = 26
    min_score_delta: float = 0.0
    max_candidates_per_volume: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ComponentCutInfo:
    component_id: int
    voxel_count: int
    cut: bool
    n_seeds: int = 0
    reason: str = ""


@dataclass
class VolumeUnmergeResult:
    case_id: str
    n_components: int
    n_candidates: int
    n_cut: int
    accepted: bool
    control_score: Optional[float]
    proposed_score: Optional[float]
    score_delta: Optional[float]
    control_voi_merge: Optional[float]
    proposed_voi_merge: Optional[float]
    voi_merge_delta: Optional[float]
    seconds: float


def _seed_labels(crop: np.ndarray, cfg: UnmergeConfig) -> tuple[np.ndarray, int]:
    """Erode ``crop`` and label what survives. Each surviving blob >= min_seed_size
    is a seed; smaller survivors are dropped (treated as erosion noise, not a
    genuine second mass)."""
    footprint = make_ball_footprint(cfg.erosion_radius)
    eroded = binary_erosion(crop, structure=footprint).astype(np.uint8)
    struct = _structure(cfg.connectivity)
    labeled, n = cc_label(eroded, structure=struct)
    if n == 0:
        return labeled, 0

    keep_id = 0
    out = np.zeros_like(labeled)
    for i in range(1, n + 1):
        blob = labeled == i
        if int(blob.sum()) >= cfg.min_seed_size:
            keep_id += 1
            out[blob] = keep_id
    return out, keep_id


def _partition_by_nearest_seed(crop: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Assign every foreground voxel in ``crop`` to its nearest seed label
    (Euclidean nearest-seed / Voronoi partition, restricted to the mask)."""
    not_seed = seeds == 0
    _, indices = distance_transform_edt(not_seed, return_indices=True)
    nearest_label = seeds[tuple(indices)]
    return np.where(crop.astype(bool), nearest_label, 0)


def _cut_partition_boundary(partition: np.ndarray, cut_width: int) -> np.ndarray:
    """Remove a ``cut_width``-voxel boundary between differently-labeled regions
    of ``partition``. Returns a binary mask (0/1), same shape, with the cut applied.

    Neighbor comparison uses explicit padding + slicing (not ``np.roll``), which
    would wrap the far edge of the array back around as a "neighbor" and could
    fabricate a boundary between two unrelated faces of the crop."""
    fg = partition > 0
    result = fg.astype(np.uint8)
    padded = np.pad(partition, 1, mode="constant", constant_values=0)
    boundary = np.zeros_like(result, dtype=bool)

    shifts = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for dz, dy, dx in shifts:
        d, h, w = partition.shape
        shifted = padded[1 + dz : 1 + dz + d, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        differs = fg & (shifted > 0) & (partition != shifted)
        boundary |= differs

    if cut_width > 1:
        footprint = make_ball_footprint(cut_width - 1)
        boundary = binary_dilation(boundary, structure=footprint)

    result[boundary] = 0
    return result


def propose_cuts(
    mask: np.ndarray,
    config: Optional[UnmergeConfig] = None,
) -> tuple[np.ndarray, list[ComponentCutInfo]]:
    """Run bridge detection + cutting over every connected component of ``mask``.

    ``mask`` should already be the 1st-place CONTROL output (post remove_small /
    closing / patch / plug / fill), not a raw prediction -- this layer only
    targets residual merges the control leaves behind.

    Returns the proposed (uncut components untouched, candidates cut) mask plus
    a per-component diagnostic list.
    """
    cfg = config or UnmergeConfig()
    mask = (mask > 0).astype(np.uint8)
    struct = _structure(cfg.connectivity)
    labeled, n = cc_label(mask, structure=struct)
    if n == 0:
        return mask, []

    slices = find_objects(labeled)
    pad = cfg.erosion_radius + 1
    result = np.zeros_like(mask, dtype=np.uint8)
    infos: list[ComponentCutInfo] = []
    n_candidates_used = 0

    for comp_id, sl in enumerate(slices, 1):
        if sl is None:
            continue
        padded_sl = _pad_slices(sl, mask.shape, pad)
        crop = (labeled[padded_sl] == comp_id).astype(np.uint8)
        voxel_count = int(crop.sum())

        seeds, n_seeds = _seed_labels(crop, cfg)
        is_candidate = n_seeds >= 2
        budget_ok = (
            cfg.max_candidates_per_volume is None
            or n_candidates_used < cfg.max_candidates_per_volume
        )

        if not is_candidate or not budget_ok:
            result[padded_sl] |= crop
            infos.append(
                ComponentCutInfo(
                    component_id=comp_id,
                    voxel_count=voxel_count,
                    cut=False,
                    n_seeds=n_seeds,
                    reason="not_candidate" if not is_candidate else "budget_exhausted",
                )
            )
            continue

        n_candidates_used += 1
        partition = _partition_by_nearest_seed(crop, seeds)
        cut_crop = _cut_partition_boundary(partition, cfg.cut_width)

        # Drop pieces smaller than min_piece_size (cut debris).
        piece_labels, n_pieces = cc_label(cut_crop, structure=struct)
        cleaned = np.zeros_like(cut_crop)
        for i in range(1, n_pieces + 1):
            piece = piece_labels == i
            if int(piece.sum()) >= cfg.min_piece_size:
                cleaned[piece] = 1

        result[padded_sl] |= cleaned
        infos.append(
            ComponentCutInfo(
                component_id=comp_id,
                voxel_count=voxel_count,
                cut=True,
                n_seeds=n_seeds,
                reason=f"cut_into_{n_pieces}_pieces",
            )
        )

    return result, infos


def apply_unmerge(
    control_mask: np.ndarray,
    label: np.ndarray,
    config: Optional[UnmergeConfig] = None,
    *,
    case_id: str = "",
) -> tuple[np.ndarray, VolumeUnmergeResult]:
    """Propose cuts on ``control_mask``, score proposed-vs-control against
    ``label`` with the official metric, and accept only if score improves by at
    least ``cfg.min_score_delta``. Returns ``(final_mask, result)`` where
    ``final_mask`` is ``proposed`` if accepted else ``control_mask`` unchanged.
    """
    from vesuvius_surface.evaluation.metric_adapter import score_pair  # local import: keeps this

    # module importable even where the metric package isn't installed, e.g. unit
    # tests that only exercise propose_cuts().

    cfg = config or UnmergeConfig()
    started = time.perf_counter()

    proposed, infos = propose_cuts(control_mask, cfg)
    n_candidates = sum(1 for i in infos if i.n_seeds >= 2)
    n_cut = sum(1 for i in infos if i.cut)

    if n_cut == 0:
        elapsed = time.perf_counter() - started
        return control_mask, VolumeUnmergeResult(
            case_id=case_id,
            n_components=len(infos),
            n_candidates=n_candidates,
            n_cut=0,
            accepted=False,
            control_score=None,
            proposed_score=None,
            score_delta=None,
            control_voi_merge=None,
            proposed_voi_merge=None,
            voi_merge_delta=None,
            seconds=elapsed,
        )

    control_parts = score_pair(control_mask, label)
    proposed_parts = score_pair(proposed, label)

    control_score = control_parts.get("score")
    proposed_score = proposed_parts.get("score")
    delta = (
        proposed_score - control_score
        if control_score is not None and proposed_score is not None
        else None
    )
    accepted = delta is not None and delta >= cfg.min_score_delta
    final_mask = proposed if accepted else control_mask

    cv = control_parts.get("voi_merge")
    pv = proposed_parts.get("voi_merge")
    voi_merge_delta = (pv - cv) if cv is not None and pv is not None else None

    elapsed = time.perf_counter() - started
    result = VolumeUnmergeResult(
        case_id=case_id,
        n_components=len(infos),
        n_candidates=n_candidates,
        n_cut=n_cut,
        accepted=accepted,
        control_score=control_score,
        proposed_score=proposed_score,
        score_delta=delta,
        control_voi_merge=cv,
        proposed_voi_merge=pv,
        voi_merge_delta=voi_merge_delta,
        seconds=elapsed,
    )
    return final_mask, result


def _write_tif(path: Path, array: np.ndarray) -> None:
    if tifffile is None:
        raise ImportError("tifffile is required to write post-processed volumes")
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), array.astype(np.uint8))


def run_directory(
    control_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    *,
    config: Optional[UnmergeConfig] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
    resume: bool = True,
) -> dict:
    """Run metric-guided unmerge over every control mask in ``control_dir``.

    Writes ``output_dir/unmerge_proposed/<case>.tif`` (cut applied unconditionally,
    for inspection) and ``output_dir/unmerge_accepted/<case>.tif`` (cut only where
    it won the accept/reject gate -- this is the layer's actual output). Appends
    one JSON line per case to ``output_dir/unmerge_results.jsonl`` (resumable, same
    convention as ``evaluation.harness.evaluate_directory``) and writes a summary
    with per-scroll and overall deltas to ``output_dir/unmerge_summary.json``.
    """
    from vesuvius_surface.data.io import load_volume

    control_dir = Path(control_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    cfg = config or UnmergeConfig()

    case_ids = sorted(
        p.stem for p in control_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")
    )
    if limit is not None:
        case_ids = case_ids[:limit]

    results_path = output_dir / "unmerge_results.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "unmerge_meta.json").write_text(
        json.dumps({"config": cfg.to_dict(), "n_inputs": len(case_ids)}, indent=2) + "\n",
        encoding="utf-8",
    )

    done: dict[str, dict] = {}
    if resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                done[str(payload["case_id"])] = payload
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping malformed result line in %s", results_path)

    results: list[VolumeUnmergeResult] = [VolumeUnmergeResult(**v) for v in done.values()]
    pending = [c for c in case_ids if c not in done]
    logger.info("Unmerging %d case(s) (%d already done)", len(pending), len(done))

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(pending, desc="unmerge", unit="vol")
    except ImportError:
        iterator = pending

    with results_path.open("a", encoding="utf-8") as handle:
        for case_id in iterator:
            control_path = control_dir / f"{case_id}.tif"
            label_path = labels_dir / f"{case_id}.tif"
            if not control_path.exists() or not label_path.exists():
                logger.warning("Missing control or label for %s, skipping", case_id)
                continue

            control_mask = load_volume(control_path)
            label = load_volume(label_path)

            proposed, _infos = propose_cuts(control_mask, cfg)
            proposed_out = output_dir / "unmerge_proposed" / f"{case_id}.tif"
            if overwrite or not proposed_out.exists():
                _write_tif(proposed_out, proposed)

            final_mask, result = apply_unmerge(control_mask, label, cfg, case_id=case_id)
            accepted_out = output_dir / "unmerge_accepted" / f"{case_id}.tif"
            if overwrite or not accepted_out.exists():
                _write_tif(accepted_out, final_mask)

            handle.write(json.dumps(asdict(result)) + "\n")
            handle.flush()
            results.append(result)

    n_cut_volumes = sum(1 for r in results if r.n_cut > 0)
    n_accepted = sum(1 for r in results if r.accepted)
    deltas = [r.score_delta for r in results if r.score_delta is not None]
    voi_deltas = [r.voi_merge_delta for r in results if r.voi_merge_delta is not None]

    summary = {
        "n_volumes": len(results),
        "n_volumes_with_candidate_cuts": n_cut_volumes,
        "n_volumes_accepted": n_accepted,
        "mean_score_delta_where_cut": float(np.mean(deltas)) if deltas else None,
        "mean_voi_merge_delta_where_cut": float(np.mean(voi_deltas)) if voi_deltas else None,
        "config": cfg.to_dict(),
    }
    (output_dir / "unmerge_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "UNMERGE_STAGES",
    "ComponentCutInfo",
    "UnmergeConfig",
    "VolumeUnmergeResult",
    "apply_unmerge",
    "propose_cuts",
    "run_directory",
]
