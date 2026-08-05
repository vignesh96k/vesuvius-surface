#!/usr/bin/env python3
"""Validate Vesuvius Surface Detection dataset layout and labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--max-volumes",
        type=int,
        default=None,
        help="Optional cap on volumes fully scanned (for quick checks).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional directory for CSV reports.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_dataset(
        args.data_root,
        split=args.split,
        max_volumes_to_scan=args.max_volumes,
    )
    print(report.summary())

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        report.to_frame().to_csv(args.out / f"issues_{args.split}.csv", index=False)
        if not report.inventory.empty:
            report.inventory.to_csv(args.out / f"inventory_{args.split}.csv", index=False)
        print(f"Wrote reports under {args.out}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
