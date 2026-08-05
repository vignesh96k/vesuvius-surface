#!/usr/bin/env python3
"""Step 1 — export Kaggle Surface Detection data to nnU-Net v2 raw format."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.nnunet_export import export_nnunet_dataset, write_scroll_holdout_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/workspace/code/datasets/vesuvius-challenge-surface-detection"),
        help="Kaggle extract root (train_images/, train_labels/, …).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/workspace/code/nnUNet_raw"),
        help="nnU-Net raw parent (env nnUNet_raw).",
    )
    p.add_argument("--dataset-id", type=int, default=100)
    p.add_argument("--dataset-name", type=str, default="VesuviusSurface")
    p.add_argument(
        "--mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to place files under imagesTr/labelsTr (symlink avoids a 27GB copy).",
    )
    p.add_argument("--no-test", action="store_true", help="Skip imagesTs/")
    p.add_argument(
        "--max-train-volumes",
        type=int,
        default=None,
        help="Optional cap for a smoke export.",
    )
    p.add_argument(
        "--val-scroll-ids",
        nargs="*",
        default=None,
        help="If set, also write splits_final.json with these scrolls held out.",
    )
    p.add_argument(
        "--splits-out",
        type=Path,
        default=None,
        help="Path for splits_final.json (default: <dataset_dir>/splits_final.json).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if not (args.data_root / "train.csv").exists():
        print(f"ERROR: train.csv not found under {args.data_root}", file=sys.stderr)
        return 1

    result = export_nnunet_dataset(
        args.data_root,
        args.output_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        mode=args.mode,
        include_test=not args.no_test,
        max_train_volumes=args.max_train_volumes,
    )

    print(f"dataset_dir : {result.dataset_dir}")
    print(f"n_train     : {result.n_train}")
    print(f"n_test      : {result.n_test}")
    print(f"dataset.json: {result.dataset_json}")
    print(f"scrolls     : {result.scroll_groups_json}")

    if args.val_scroll_ids:
        splits_out = args.splits_out or (result.dataset_dir / "splits_final.json")
        write_scroll_holdout_split(
            result.scroll_groups_json,
            splits_out,
            val_scroll_ids=args.val_scroll_ids,
        )
        print(f"splits      : {splits_out}")
        print(
            "NOTE: copy/symlink splits_final.json into "
            "$nnUNet_preprocessed/DatasetXXX_Name/ after planning, "
            "or pass it via your training workflow."
        )

    print(
        "\nNext (step 2):\n"
        "  export nnUNet_raw=/mnt/workspace/code/nnUNet_raw\n"
        "  export nnUNet_preprocessed=/mnt/workspace/code/nnUNet_preprocessed\n"
        "  export nnUNet_results=/mnt/workspace/code/nnUNet_results\n"
        f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
