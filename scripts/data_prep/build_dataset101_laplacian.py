#!/usr/bin/env python3
"""Builds Dataset101_VesuviusSurfaceLaplacian: same cases/labels as Dataset100, but with a
second input channel = first (finest) high-pass sub-band of a 3D Laplacian pyramid
(volume - gaussian_blur(volume, sigma=1.0), see highpass.py). Channel 0 (raw CT) and labels
are symlinked from Dataset100 (no duplication); channel 1 is newly computed and written as
a float32 TIFF.

This is experiment_summary.md Phase 3 item 11's "laplacian" candidate in the 100-epoch 5-way
loss/input comparison (real result: 0.5122, lost to skeleton-recall's 0.5307). New dataset ID
(101) deliberately, so Dataset100 (in active use everywhere else) stays untouched.

After this script, run standard nnU-Net preprocessing, then copy this project's own LOSO
split into the new dataset's preprocessed folder (nnU-Net's own auto-generated split would
NOT match our authored scroll-holdout):

    nnUNetv2_extract_fingerprint -d 101
    nnUNetv2_plan_experiment -d 101 -pl nnUNetPlannerResEncM
    nnUNetv2_preprocess -d 101 -c 3d_lowres -plans_name nnUNetResEncUNetMPlans
    cp $nnUNet_preprocessed/Dataset100_VesuviusSurface/splits_final.json \\
       $nnUNet_preprocessed/Dataset101_VesuviusSurfaceLaplacian/splits_final.json
    nnUNetv2_train 101 3d_lowres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_100epochs

(Verified directly: Dataset101's own splits_final.json fold_0 validation set, from the real
run, is byte-identical to Dataset100's -- the copy step above is not a guess.)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from highpass import laplacian_highpass_3d  # noqa: E402

NNUNET_RAW = Path(os.environ.get("nnUNet_raw", "/mnt/workspace/vesuvius_training_scratch/nnUNet_data/nnUNet_raw"))
SRC_ROOT = NNUNET_RAW / "Dataset100_VesuviusSurface"
DST_ROOT = NNUNET_RAW / "Dataset101_VesuviusSurfaceLaplacian"
SIGMA = 1.0


def process_case(case_id: str) -> str:
    src_img = SRC_ROOT / "imagesTr" / f"{case_id}_0000.tif"
    src_json = SRC_ROOT / "imagesTr" / f"{case_id}_0000.json"
    dst_ch0 = DST_ROOT / "imagesTr" / f"{case_id}_0000.tif"
    dst_ch0_json = DST_ROOT / "imagesTr" / f"{case_id}_0000.json"
    dst_ch1 = DST_ROOT / "imagesTr" / f"{case_id}_0001.tif"
    dst_ch1_json = DST_ROOT / "imagesTr" / f"{case_id}_0001.json"
    dst_label = DST_ROOT / "labelsTr" / f"{case_id}.tif"
    src_label = SRC_ROOT / "labelsTr" / f"{case_id}.tif"

    if not dst_ch0.exists():
        dst_ch0.symlink_to(src_img)
    if not dst_ch0_json.exists():
        dst_ch0_json.symlink_to(src_json)
    if not dst_label.exists():
        dst_label.symlink_to(src_label)

    if not dst_ch1.exists():
        img = tifffile.imread(src_img)
        hp = laplacian_highpass_3d(img, sigma=SIGMA)
        tifffile.imwrite(dst_ch1, hp)
    if not dst_ch1_json.exists():
        shutil.copy2(src_json, dst_ch1_json)

    return case_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only first N cases (for testing)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    (DST_ROOT / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "labelsTr").mkdir(parents=True, exist_ok=True)

    case_ids = sorted(p.stem[:-5] for p in (SRC_ROOT / "imagesTr").glob("*_0000.tif"))
    if args.limit:
        case_ids = case_ids[: args.limit]
    print(f"processing {len(case_ids)} cases with {args.workers} workers")

    done = 0
    with Pool(args.workers) as pool:
        for cid in pool.imap_unordered(process_case, case_ids):
            done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{len(case_ids)}")

    src_dataset_json = json.load(open(SRC_ROOT / "dataset.json"))
    dataset_json = dict(src_dataset_json)
    dataset_json["channel_names"] = {"0": "CT", "1": "noNorm"}
    dataset_json["numTraining"] = len(case_ids)
    with open(DST_ROOT / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"done. {len(case_ids)} cases written to {DST_ROOT}")


if __name__ == "__main__":
    main()
