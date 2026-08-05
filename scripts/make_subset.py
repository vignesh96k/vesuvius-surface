#!/usr/bin/env python3
"""Materialise an images(+labels) folder for a subset of cases.

Typical use: build the fold-0 holdout so a published checkpoint can be scored
on volumes it never trained on.

    python scripts/nnunet_folds.py --out reports/m7_holdout.json
    python scripts/make_subset.py --manifest reports/m7_holdout.json \\
        --output /mnt/workspace/code/subsets/m7_holdout
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.nnunet_export import materialize_case_subset
from data.nnunet_splits import describe_fold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/mnt/workspace/code/nnUNet_raw/Dataset100_VesuviusSurface"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Holdout manifest from scripts/nnunet_folds.py (uses its val_case_ids).",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Derive the case list from this fold instead of a manifest.",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    p.add_argument(
        "--no-labels",
        action="store_true",
        help="Images only (e.g. when the subset is for prediction alone).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def load_case_ids(args: argparse.Namespace) -> list[str]:
    if args.manifest is not None:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        case_ids = [str(c) for c in payload.get("val_case_ids", [])]
        if not case_ids:
            raise ValueError(f"No val_case_ids in {args.manifest}")
        return case_ids

    if args.fold is None:
        raise ValueError("Provide either --manifest or --fold")

    report = describe_fold(args.dataset_dir, fold=args.fold)
    return report.val_case_ids


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if not args.dataset_dir.is_dir():
        print(f"ERROR: dataset dir not found: {args.dataset_dir}", file=sys.stderr)
        return 1

    try:
        case_ids = load_case_ids(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    images_dir, labels_dir = materialize_case_subset(
        args.dataset_dir,
        case_ids,
        args.output,
        mode=args.mode,
        include_labels=not args.no_labels,
    )

    print(f"cases   : {len(case_ids)}")
    print(f"images  : {images_dir}")
    if labels_dir is not None:
        print(f"labels  : {labels_dir}")

    print(
        "\nNext — reference prediction:\n"
        f"  bash scripts/nnunet_predict.sh \\\n"
        f"      --input {images_dir} \\\n"
        f"      --output {args.output}/predictions \\\n"
        f"      --plans nnUNetResEncUNetLPlans --config 3d_fullres"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
