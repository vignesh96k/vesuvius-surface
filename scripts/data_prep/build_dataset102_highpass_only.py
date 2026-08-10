#!/usr/bin/env python3
"""Builds Dataset102_VesuviusSurfaceHighpassOnly: single channel = the same Laplacian
high-pass data already computed for Dataset101 (channel 1 there), used here as the SOLE
input channel -- no raw CT channel at all. Tests whether the network needs absolute
intensity information, or whether edge/texture structure alone is sufficient. Labels
symlinked from the same original source as everything else.

This is experiment_summary.md Phase 3 item 11's "highpassonly" candidate in the 100-epoch
5-way loss/input comparison (real result: 0.5204, lost to skeleton-recall's 0.5307). Run
build_dataset101_laplacian.py first -- this script reuses its channel-1 output directly
rather than recomputing it. Preprocessing/training steps and the LOSO-split-copy step are
the same as build_dataset101_laplacian.py's docstring, substituting dataset id 102 and
`-tr nnUNetTrainer_100epochs` (same trainer, real, both are nnU-Net's own stock trainer --
no custom loss for either of these two candidates, only the input changes).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

NNUNET_RAW = Path(os.environ.get("nnUNet_raw", "/mnt/workspace/vesuvius_training_scratch/nnUNet_data/nnUNet_raw"))
SRC_HIGHPASS = NNUNET_RAW / "Dataset101_VesuviusSurfaceLaplacian" / "imagesTr"
ORIG_LABELS = Path(os.environ.get("VESUVIUS_DATA_ROOT", "/mnt/workspace/code/datasets/vesuvius-challenge-surface-detection")) / "train_labels"
DST_ROOT = NNUNET_RAW / "Dataset102_VesuviusSurfaceHighpassOnly"

(DST_ROOT / "imagesTr").mkdir(parents=True, exist_ok=True)
(DST_ROOT / "labelsTr").mkdir(parents=True, exist_ok=True)

case_ids = sorted(p.stem[:-5] for p in SRC_HIGHPASS.glob("*_0001.tif"))
print(f"{len(case_ids)} cases")

for cid in case_ids:
    ch0 = DST_ROOT / "imagesTr" / f"{cid}_0000.tif"
    ch0_json = DST_ROOT / "imagesTr" / f"{cid}_0000.json"
    label = DST_ROOT / "labelsTr" / f"{cid}.tif"

    if not ch0.exists():
        ch0.symlink_to(SRC_HIGHPASS / f"{cid}_0001.tif")
    if not ch0_json.exists():
        ch0_json.write_text(json.dumps({"spacing": [1.0, 1.0, 1.0]}))
    if not label.exists():
        label.symlink_to(ORIG_LABELS / f"{cid}.tif")

dataset_json = {
    "channel_names": {"0": "noNorm"},
    "labels": {"background": 0, "surface": 1, "ignore": 2},
    "numTraining": len(case_ids),
    "file_ending": ".tif",
    "overwrite_image_reader_writer": "SimpleTiffIO",
}
with open(DST_ROOT / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"done: {len(list((DST_ROOT/'imagesTr').glob('*_0000.tif')))} cases written to {DST_ROOT}")
