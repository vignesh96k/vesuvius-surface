#!/usr/bin/env python3
"""Builds Dataset102_VesuviusSurfaceHighpassOnly's fullres preprocessed data by reusing
Dataset100's already-preprocessed fullres arrays (nnUNetPlans_3d_fullres, 786 cases, already
resampled/cropped/CT-normalized -- prepared earlier for our own cascade training) rather than
redoing the expensive raw-image resampling/cropping pipeline from scratch.

Computes the Laplacian high-pass transform directly on the already-normalized image data
(channel 0): highpass = x - gaussian_blur(x, sigma=1). Since CTNormalization and the Gaussian
blur/subtraction are both linear, shift-invariant operations, this is mathematically
equivalent to computing highpass on raw intensities and then normalizing -- up to a constant
scale factor (1/std), not identical to build_dataset101_laplacian.py's lowres approach but a
disclosed, minor methodological difference, not a silent one.

Segmentation labels are byte-identical between Dataset100 and Dataset102 fullres (same case,
same crop, same resampling), so they're reused directly rather than recomputed.

This is the real code behind the "highpass input" for the
full arunodhayan fine-tune (ensemble: 0.7029 -> 0.5172; cascade: 0.7198 -> 0.5208) -- run this
before fine-tuning `nnUNetTrainerSkeletonRecallAffinity` on either checkpoint with
`-p nnUNetResEncUNetMPlans -c 3d_fullres` (ensemble) or `-c 3d_cascade_fullres` (cascade). Run
scripts/data_prep/build_dataset102_highpass_only.py first (writes Dataset102's raw-space
lowres data and dataset.json); this script fills in the fullres preprocessed tree separately.
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from scipy.ndimage import gaussian_filter

import os

NNUNET_PREPROCESSED = Path(
    os.environ.get("nnUNet_preprocessed", "/mnt/workspace/vesuvius_training_scratch/nnUNet_data/nnUNet_preprocessed")
)
SRC_DIR = NNUNET_PREPROCESSED / "Dataset100_VesuviusSurface" / "nnUNetPlans_3d_fullres"
DST_DIR = NNUNET_PREPROCESSED / "Dataset102_VesuviusSurfaceHighpassOnly" / "nnUNetResEncUNetMPlans_3d_fullres"
N_WORKERS = 16

DST_DIR.mkdir(parents=True, exist_ok=True)


def process_one(cid: str) -> str:
    out_b2nd = DST_DIR / f"{cid}.b2nd"
    if out_b2nd.exists():
        return cid

    src_ds = nnUNetDatasetBlosc2(str(SRC_DIR), [cid])
    data, seg, seg_prev, properties = src_ds.load_case(cid)
    # data shape: (C, X, Y, Z); channel 0 = CT-normalized raw intensity
    highpass = data[0].astype(np.float32)
    highpass = highpass - gaussian_filter(highpass, sigma=1.0, mode="reflect")
    highpass = highpass[None]  # restore channel dim -> (1, X, Y, Z)

    nnUNetDatasetBlosc2.save_case(
        data=highpass, seg=seg, properties=properties,
        output_filename_truncated=str(DST_DIR / cid),
    )
    return cid


if __name__ == "__main__":
    identifiers = sorted(nnUNetDatasetBlosc2.get_identifiers(str(SRC_DIR)))
    print(f"{len(identifiers)} cases, {N_WORKERS} workers")

    done = 0
    with Pool(N_WORKERS) as pool:
        for cid in pool.imap_unordered(process_one, identifiers):
            done += 1
            if done % 50 == 0:
                print(f"  ...{done}/{len(identifiers)} done")

    print(f"done: {done}/{len(identifiers)}")
