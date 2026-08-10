#!/usr/bin/env python3
"""Report nnU-Net fold membership and export a leakage-free holdout list.

The published ``scrollprize/surface_m7_nnunet`` checkpoint is ``fold_0`` of a
786-case dataset built with the same case ids we export. Reconstructing the
default split therefore identifies the cases that checkpoint never trained on,
which is the only subset it can be evaluated on honestly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vesuvius_surface.data.nnunet_splits import (
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    describe_fold,
    write_holdout_manifest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/mnt/workspace/code/nnUNet_raw/Dataset100_VesuviusSurface"),
        help="nnU-Net raw dataset directory (contains labelsTr/, scroll_groups.json).",
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument(
        "--splits-json",
        type=Path,
        default=None,
        help="Existing splits_final.json to trust instead of reconstructing.",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the holdout manifest JSON here.",
    )
    p.add_argument(
        "--list-cases",
        action="store_true",
        help="Print every validation case id.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if not args.dataset_dir.is_dir():
        print(f"ERROR: dataset dir not found: {args.dataset_dir}", file=sys.stderr)
        print("Run step 1 first: python scripts/export_nnunet.py --mode symlink", file=sys.stderr)
        return 1

    report = describe_fold(
        args.dataset_dir,
        fold=args.fold,
        splits_json=args.splits_json,
        seed=args.seed,
        n_splits=args.n_splits,
    )

    print(f"fold        : {report.fold}")
    print(f"source      : {report.source}")
    print(f"train cases : {report.n_train}")
    print(f"val cases   : {report.n_val}")
    print(f"{'scroll':>10}  {'val':>5} {'total':>6}  {'val share':>9}")
    for scroll, total in report.total_scroll_counts.items():
        in_val = report.scroll_counts.get(scroll, 0)
        share = in_val / total if total else 0.0
        print(f"  {scroll:>8}  {in_val:>5} {total:>6}  {share:>8.1%}")

    missing = report.missing_scrolls()
    if missing:
        print(
            f"\nWARNING: {len(missing)} scroll(s) absent from this validation set: "
            f"{', '.join(missing)}.\n"
            "         A score on this subset says nothing about those scrolls."
        )

    if report.source == "reconstructed":
        print(
            "\nNOTE: split was reconstructed from the default seed, not read from a\n"
            "      shipped file. Verify against the checkpoint's own splits_final.json\n"
            "      or fold_0/validation/ listing before relying on it."
        )

    if args.list_cases:
        print("\nval case ids:")
        for case_id in report.val_case_ids:
            print(f"  {case_id}")

    if args.out is not None:
        write_holdout_manifest(
            report,
            args.out,
            note="Cases absent from fold training set; safe for evaluating that fold's checkpoint.",
        )
        print(f"\nmanifest    : {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
