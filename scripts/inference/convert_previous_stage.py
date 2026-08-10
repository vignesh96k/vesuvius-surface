#!/usr/bin/env python3
"""Convert predictions into nnU-Net's cascade "previous stage prediction" format
(`predicted_next_stage/3d_cascade_fullres/<case_id>.b2nd`), so a `3d_cascade_fullres` trainer
can consume them as its coarse-hint input channel during training.

Consolidates 3 near-identical ad-hoc scripts written over the course of this project's real
cascade fine-tuning experiments (convert_zeroshot_ensemble_to_prevstage_d100.py,
convert_zeroshot_129_to_prevstage_d100.py, convert_combined_to_prevstage.py -- their original
forms are preserved in git history) into one parametrized tool.

Two real, non-obvious things this gets right, both hard-won during the actual experiments:

1. **bbox cropping.** Predictions must be cropped to each case's own bbox exactly matching the
   preprocessed GT segmentation's shape (read from the preprocessed `.pkl` sidecar's
   `bbox_used_for_cropping`), not saved at raw/original resolution.
2. **no explicit channel dimension.** `nnUNetDatasetBlosc2.save_seg` expects a raw 3D array.
   nnU-Net's own data loader adds its own channel dim at load time -- pre-adding one here
   causes a 4-vs-5-dims RuntimeError (a real bug hit and fixed during this project).

Two input modes:

    # Single already-combined discrete segmentation directory (.tif, integer labels):
    python scripts/inference/convert_previous_stage.py \\
        --input segmentation=path/to/combined_predictions \\
        --preprocessed-dir data/preprocessed/.../nnUNetPlans_3d_fullres \\
        --output-dir outputs/.../predicted_next_stage/3d_cascade_fullres

    # Two probability-map directories (.npz, argmax-combined with the given weight):
    python scripts/inference/convert_previous_stage.py \\
        --input probabilities=path/to/ensemble_A:0.65 \\
        --input probabilities=path/to/ensemble_B:0.35 \\
        --preprocessed-dir data/preprocessed/.../nnUNetPlans_3d_fullres \\
        --output-dir outputs/.../predicted_next_stage/3d_cascade_fullres

Idempotent: cases whose `.b2nd` already exists in `--output-dir` are skipped, so it's safe to
re-run repeatedly as more predictions land.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import tifffile


def _bbox_crop(volume: np.ndarray, preprocessed_dir: Path, case_id: str) -> np.ndarray:
    props = pickle.load(open(preprocessed_dir / f"{case_id}.pkl", "rb"))
    bbox = props["bbox_used_for_cropping"]
    return volume[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]


def _convert_one_segmentation(args: tuple) -> str:
    case_id, input_dir, preprocessed_dir, output_dir = args
    out_path = output_dir / f"{case_id}.b2nd"
    if out_path.exists():
        return case_id
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    seg = tifffile.imread(input_dir / f"{case_id}.tif")
    cropped = _bbox_crop(seg, preprocessed_dir, case_id).astype(np.int16)
    nnUNetDatasetBlosc2.save_seg(cropped, str(output_dir / case_id))
    return case_id


def _convert_one_probability(args: tuple) -> str:
    case_id, weighted_dirs, preprocessed_dir, output_dir = args
    out_path = output_dir / f"{case_id}.b2nd"
    if out_path.exists():
        return case_id
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    combined = None
    for weight, prob_dir in weighted_dirs:
        probs = np.load(prob_dir / f"{case_id}.npz")["probabilities"].astype(np.float32)
        combined = probs * weight if combined is None else combined + probs * weight
    seg = np.argmax(combined, axis=0).astype(np.uint8)
    cropped = _bbox_crop(seg, preprocessed_dir, case_id).astype(np.int16)
    nnUNetDatasetBlosc2.save_seg(cropped, str(output_dir / case_id))
    return case_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", action="append", required=True, metavar="MODE=PATH[:WEIGHT]",
        help="'segmentation=PATH' for one discrete-label .tif directory, or repeated "
             "'probabilities=PATH:WEIGHT' entries (weights should sum to 1.0) to argmax-combine "
             "multiple probability .npz directories",
    )
    parser.add_argument("--preprocessed-dir", type=Path, required=True, help="nnU-Net preprocessed dir containing each case's <id>.pkl (for bbox_used_for_cropping)")
    parser.add_argument("--output-dir", type=Path, required=True, help="predicted_next_stage/3d_cascade_fullres directory to write <id>.b2nd into")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seg_dirs, prob_dirs = [], []
    for spec in args.input:
        mode, _, rest = spec.partition("=")
        if mode == "segmentation":
            seg_dirs.append(Path(rest))
        elif mode == "probabilities":
            path_str, _, weight_str = rest.partition(":")
            prob_dirs.append((float(weight_str) if weight_str else 1.0, Path(path_str)))
        else:
            print(f"ERROR: --input mode must be 'segmentation' or 'probabilities', got {mode!r}", file=sys.stderr)
            return 1

    if seg_dirs and prob_dirs:
        print("ERROR: use either --input segmentation=... or --input probabilities=..., not both", file=sys.stderr)
        return 1

    if seg_dirs:
        input_dir = seg_dirs[0]
        case_ids = sorted(p.stem for p in input_dir.glob("*.tif"))
        jobs = [(cid, input_dir, args.preprocessed_dir, args.output_dir) for cid in case_ids]
        worker = _convert_one_segmentation
    else:
        common_ids = None
        for _, prob_dir in prob_dirs:
            ids = {p.stem for p in prob_dir.glob("*.npz")}
            common_ids = ids if common_ids is None else common_ids & ids
        case_ids = sorted(common_ids or [])
        jobs = [(cid, prob_dirs, args.preprocessed_dir, args.output_dir) for cid in case_ids]
        worker = _convert_one_probability

    print(f"{len(case_ids)} case(s) -> {args.output_dir}")
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, job): job[0] for job in jobs}
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"FAILED {cid}: {e}", file=sys.stderr)
                raise
            done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{len(case_ids)} done")
    print(f"done: {done}/{len(case_ids)} converted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
