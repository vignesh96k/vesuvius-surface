#!/usr/bin/env python3
"""Measure how often patch-local instances disagree with volume-level ones.

Stage 2a derives sheet instances by connected components **on the augmented
patch**. Two fragments of the same sheet that only reconnect outside the patch
look like separate instances inside it, so a long-range affinity across them is
labelled 0 when the volume-level truth is 1.

This script reports, over random crops of every training volume, how often that
happens. Run it once before trusting Stage 2a numbers:

    export PYTHONPATH=/mnt/workspace/code/vesuvius-surface/src:$PYTHONPATH
    python scripts/audit_instance_locality.py \\
        --dataset-dir /mnt/workspace/code/nnUNet_raw/Dataset100_VesuviusSurface \\
        --patch-size 128 --n-crops 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vesuvius_surface.data.io import load_volume
from vesuvius_surface.data.nnunet_splits import list_case_ids
from vesuvius_surface.data.schema import LABEL_SURFACE
from vesuvius_surface.training.affinity import (
    DEFAULT_AFFINITY_OFFSETS,
    LONG_RANGE_OFFSETS,
    NEAREST_NEIGHBOUR_OFFSETS,
    affinity_targets,
    instance_labels,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/mnt/workspace/code/nnUNet_raw/Dataset100_VesuviusSurface"),
    )
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--n-crops", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-volumes", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def random_crop(volume: np.ndarray, patch: int, rng: np.random.Generator):
    starts = []
    for dim in volume.shape:
        if dim < patch:
            raise ValueError(f"volume shape {volume.shape} smaller than patch {patch}")
        starts.append(int(rng.integers(0, dim - patch + 1)))
    slices = tuple(slice(s, s + patch) for s in starts)
    return volume[slices], starts


def disagreement_rate(
    volume_instances: np.ndarray,
    patch_seg: np.ndarray,
    starts: list[int],
    offsets,
) -> tuple[int, int]:
    """Count valid pairs where patch-local affinity != volume-level affinity."""
    patch_size = patch_seg.shape
    vol_crop = volume_instances[
        tuple(slice(s, s + p) for s, p in zip(starts, patch_size))
    ]
    # Build both target stacks. For the volume-level one we pass the cropped
    # volume instances directly so connectivity outside the patch is preserved.
    local = affinity_targets(patch_seg, offsets=offsets)
    # Fake a "segmentation" that is surface wherever vol_crop > 0, then pass
    # the cropped volume instances in.
    fake_seg = np.where(vol_crop > 0, LABEL_SURFACE, 0).astype(patch_seg.dtype)
    # Preserve ignore from the real patch so the validity masks match.
    from vesuvius_surface.data.schema import LABEL_IGNORE

    fake_seg[patch_seg == LABEL_IGNORE] = LABEL_IGNORE
    volume_level = affinity_targets(fake_seg, offsets=offsets, instances=vol_crop)

    valid = (local >= 0) & (volume_level >= 0)
    disagree = valid & (local != volume_level)
    return int(disagree.sum()), int(valid.sum())


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    labels_dir = args.dataset_dir / "labelsTr"
    if not labels_dir.is_dir():
        print(f"ERROR: labels dir not found: {labels_dir}", file=sys.stderr)
        return 1

    case_ids = list_case_ids(args.dataset_dir)
    if args.max_volumes is not None:
        case_ids = case_ids[: args.max_volumes]

    rng = np.random.default_rng(args.seed)
    patch = args.patch_size

    buckets = {
        "short": NEAREST_NEIGHBOUR_OFFSETS,
        "long": LONG_RANGE_OFFSETS,
        "all": DEFAULT_AFFINITY_OFFSETS,
    }
    totals = {k: [0, 0] for k in buckets}

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(case_ids, desc="audit", unit="vol")
    except ImportError:
        iterator = case_ids

    for case_id in iterator:
        label = load_volume(labels_dir / f"{case_id}.tif")
        if any(d < patch for d in label.shape):
            logger.warning("Skipping %s: shape %s < patch %d", case_id, label.shape, patch)
            continue
        volume_instances = instance_labels(label)
        for _ in range(args.n_crops):
            crop, starts = random_crop(label, patch, rng)
            for name, offsets in buckets.items():
                bad, total = disagreement_rate(volume_instances, crop, starts, offsets)
                totals[name][0] += bad
                totals[name][1] += total

    print()
    print(f"{'bucket':>10}  {'disagree':>12}  {'valid':>12}  {'rate':>8}")
    print("-" * 48)
    for name in ("short", "long", "all"):
        bad, total = totals[name]
        rate = bad / total if total else float("nan")
        print(f"{name:>10}  {bad:>12}  {total:>12}  {rate:>8.4f}")

    print()
    print(
        "Interpretation: short-range rate near 0 is expected and healthy.\n"
        "Long-range rate is the cost of patch-local instances; if it is large\n"
        "(say >0.05), long-range affinities are teaching a noisy target and the\n"
        "shortrange ablation is the fairer Stage 2a test."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
