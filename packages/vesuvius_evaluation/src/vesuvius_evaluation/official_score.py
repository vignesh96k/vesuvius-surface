"""Official Vesuvius Surface Detection competition metric.

`load_volume` and `score_single_tif` are ported verbatim (same params, same defaults, same
call into `topometrics.leaderboard.compute_leaderboard_score`) from the organizers' own demo
notebook, sohier/vesuvius-2025-metric-demo:
https://www.kaggle.com/code/sohier/vesuvius-2025-metric-demo

Two deliberate deviations from that source, both purely about *how the dependency gets
installed*, not about the scoring computation itself:

1. `install_dependencies()` is dropped. The source calls it lazily inside `score_single_tif`
   because Kaggle notebooks run with internet disabled and must install from an offline wheel
   cache mounted at /kaggle/input. We have real internet access, so installation is a one-time
   step via scripts/install_topometrics.sh instead (see that file for why: same pinned
   versions, installed from PyPI directly rather than offline wheels).
2. `load_volume` here uses `tifffile` instead of `PIL.Image` + `ImageSequence` for reading
   multi-page TIFFs. Both read the exact same on-disk multi-page TIFF format; tifffile is
   already a pinned dependency of this env (and of baselinerun) and is the more common library
   for reading exactly this kind of scientific TIFF in this codebase, so it's used here to
   avoid needing a third TIFF-reading path. This does not change any scored values -- it's an
   I/O substitution only.

`score()` (a Kaggle-competition-harness-shaped `score(solution_df, submission_df,
row_id_column_name)` function matching Kaggle's own metric-function calling convention) is
NOT ported as-is, since we're not running inside Kaggle's grading harness. `score_directory`
below performs the identical per-case scoring + mean aggregation `score()` does, just driven
by two local directories of matching-named .tif files instead of Kaggle's solution/submission
dataframes.
"""

from __future__ import annotations

import glob
import os

# Must happen before numpy (or anything that imports numpy) loads anywhere in this process --
# BLAS/OpenMP libraries read these once, at library init, not per-call. Set too late (e.g.
# inside a ProcessPoolExecutor `initializer`, which runs after fork) and a forked worker
# inherits the parent's already-initialized, already-unlimited thread pool regardless. This
# is the actual fix for the thread-oversubscription stall found scoring fold 3: workers were
# discovered to have 22 threads each with n_workers=12 on a 22-core machine (up to ~264
# threads for 22 cores) because nothing here had ever set these vars. This is genuinely the
# first numpy import in the whole evaluate.py -> official_score.py call chain (evaluate.py
# itself only imports argparse/json at module level and imports this module lazily inside
# main()), so this is the correct, earliest place to set it.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile


class ParticipantVisibleError(Exception):
    pass


class HostVisibleError(Exception):
    pass


def load_volume(path: Union[str, Path]) -> np.ndarray:
    return tifffile.imread(str(path))


def score_single_tif(
    gt_path: Union[str, Path],
    pred_path: Union[str, Path],
    surface_tolerance: float,
    voi_connectivity: int = 26,
    voi_transform: str = "one_over_one_plus",
    voi_alpha: float = 0.3,
    topo_weight: float = 0.3,
    surface_dice_weight: float = 0.35,
    voi_weight: float = 0.35,
):
    import topometrics.leaderboard

    gt: np.ndarray = load_volume(gt_path)
    pr: np.ndarray = load_volume(pred_path)

    score_report = topometrics.leaderboard.compute_leaderboard_score(
        predictions=pr,
        labels=gt,
        dims=(0, 1, 2),
        spacing=(1.0, 1.0, 1.0),  # (z, y, x)
        surface_tolerance=surface_tolerance,  # in spacing units
        voi_connectivity=voi_connectivity,
        voi_transform=voi_transform,
        voi_alpha=voi_alpha,
        combine_weights=(topo_weight, surface_dice_weight, voi_weight),  # (Topo, SurfaceDice, VOI)
        fg_threshold=None,  # None => legacy "!= 0"; else uses "x > threshold"
        ignore_label=2,  # voxels with this GT label are ignored
        ignore_mask=None,
    )
    return float(np.clip(score_report.score, a_min=0.0, a_max=1.0)), score_report


def _limit_worker_threads() -> None:
    """ProcessPoolExecutor initializer, run once per worker before any scoring starts.

    Defensive backup, not the primary fix -- the real fix is the module-level env vars set
    above, before numpy is imported anywhere in this process. Fork-based multiprocessing
    (Linux default) means a worker inherits the parent's already-imported numpy and its
    already-initialized BLAS thread pool; setting env vars here, after fork, would be too
    late on its own if the parent had ever imported numpy without the limit already set. This
    stays as a second layer in case some future code path imports numpy in a worker before
    the module-level guard has run (e.g. a different entrypoint that imports this module
    differently) -- cheap insurance, not redundant effort.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _score_one_case(
    cid: str,
    pred_path: Path,
    gt_path: Path,
    surface_tolerance: float,
    voi_connectivity: int,
    voi_transform: str,
    voi_alpha: float,
    topo_weight: float,
    surface_dice_weight: float,
    voi_weight: float,
) -> Tuple[str, Dict]:
    """Score a single case. Module-level (not nested) so it's picklable for
    ProcessPoolExecutor -- each worker process does its own lazy `import topometrics`."""
    s, report = score_single_tif(
        gt_path,
        pred_path,
        surface_tolerance,
        voi_connectivity=voi_connectivity,
        voi_transform=voi_transform,
        voi_alpha=voi_alpha,
        topo_weight=topo_weight,
        surface_dice_weight=surface_dice_weight,
        voi_weight=voi_weight,
    )
    return cid, {
        "score": s,
        "surface_dice": report.surface_dice,
        "topo_score": report.topo.toposcore,
        "voi_score": report.voi.voi_score,
    }


def score_directory(
    pred_dir: Union[str, Path],
    gt_dir: Union[str, Path],
    surface_tolerance: float = 2.0,
    voi_connectivity: int = 26,
    voi_transform: str = "one_over_one_plus",
    voi_alpha: float = 0.3,
    topo_weight: float = 0.3,
    surface_dice_weight: float = 0.35,
    voi_weight: float = 0.35,
    case_ids: Optional[List[str]] = None,
    verbose: bool = True,
    n_workers: int = 1,
) -> Dict:
    """Score every matching (by filename stem) .tif pair in pred_dir / gt_dir.

    Mirrors the official score() function's per-volume-then-mean aggregation, adapted to
    operate on two local directories instead of Kaggle's solution/submission dataframes.

    Each case's score is independent (no shared state), so this is embarrassingly parallel --
    `n_workers > 1` scores cases concurrently via a process pool (the scorer is CPU-bound:
    topology/Betti-number computation, not I/O), rather than one case at a time.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    if case_ids is None:
        pred_files = sorted(glob.glob(str(pred_dir / "*.tif")))
        case_ids = [Path(p).stem for p in pred_files]

    if len(case_ids) == 0:
        raise ParticipantVisibleError(f"No .tif predictions found in {pred_dir}")

    # Validate all paths up front (fail fast, before spinning up worker processes)
    jobs = []
    for cid in case_ids:
        pred_path = pred_dir / f"{cid}.tif"
        gt_path = gt_dir / f"{cid}.tif"
        if not pred_path.exists():
            raise ParticipantVisibleError(f"Missing prediction for case {cid}: {pred_path}")
        if not gt_path.exists():
            raise HostVisibleError(f"Missing ground truth for case {cid}: {gt_path}")
        jobs.append((cid, pred_path, gt_path))

    per_case = {}

    if n_workers <= 1:
        for cid, pred_path, gt_path in jobs:
            cid, result = _score_one_case(
                cid, pred_path, gt_path, surface_tolerance, voi_connectivity, voi_transform,
                voi_alpha, topo_weight, surface_dice_weight, voi_weight,
            )
            per_case[cid] = result
            if verbose:
                print(
                    f"{cid}: score={result['score']:.4f} "
                    f"(surface_dice={result['surface_dice']:.4f}, "
                    f"topo={result['topo_score']:.4f}, "
                    f"voi={result['voi_score']:.4f})"
                )
    else:
        if verbose:
            print(f"Scoring {len(jobs)} cases with {n_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_limit_worker_threads) as executor:
            futures = {
                executor.submit(
                    _score_one_case, cid, pred_path, gt_path, surface_tolerance,
                    voi_connectivity, voi_transform, voi_alpha, topo_weight,
                    surface_dice_weight, voi_weight,
                ): cid
                for cid, pred_path, gt_path in jobs
            }
            done = 0
            for future in as_completed(futures):
                cid, result = future.result()
                per_case[cid] = result
                done += 1
                if verbose:
                    print(
                        f"[{done}/{len(jobs)}] {cid}: score={result['score']:.4f} "
                        f"(surface_dice={result['surface_dice']:.4f}, "
                        f"topo={result['topo_score']:.4f}, "
                        f"voi={result['voi_score']:.4f})"
                    )

    mean_score = float(np.mean([per_case[cid]["score"] for cid in case_ids]))
    return {"mean_score": mean_score, "per_case": per_case}
