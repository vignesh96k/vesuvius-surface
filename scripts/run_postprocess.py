#!/usr/bin/env python3
"""Apply the 1st-place post-processing baseline (control).

    # Full chain on a prediction folder:
    python scripts/run_postprocess.py \\
        --predictions /mnt/workspace/code/subsets/m7_holdout/predictions \\
        --output      /mnt/workspace/code/subsets/m7_holdout/pp_firstplace

    # Cumulative ablation (one subfolder per stage):
    python scripts/run_postprocess.py \\
        --predictions ... --output ... --ablate --limit 12

Then score each stage with scripts/evaluate.py against the matching labels.
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

from postprocess.first_place import FIRST_PLACE_STAGES, PostprocessConfig
from postprocess.pipeline import run_directory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--through-stage",
        choices=FIRST_PLACE_STAGES,
        default="fill",
        help="Stop after this cumulative stage (default: full chain).",
    )
    p.add_argument(
        "--ablate",
        action="store_true",
        help="Write every cumulative stage into output/<stage>/.",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-component-size", type=int, default=20_000)
    p.add_argument("--closing-radius", type=int, default=3)
    p.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26])
    p.add_argument("--no-closing", action="store_true")
    p.add_argument("--no-patching", action="store_true")
    p.add_argument("--no-plug", action="store_true")
    p.add_argument("--no-fill", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if not args.predictions.is_dir():
        print(f"ERROR: predictions dir not found: {args.predictions}", file=sys.stderr)
        return 1

    cfg = PostprocessConfig(
        min_component_size=args.min_component_size,
        closing_radius=args.closing_radius,
        connectivity=args.connectivity,
        enable_closing=not args.no_closing,
        enable_patching=not args.no_patching,
        enable_hole_plugging=not args.no_plug,
        enable_fill_holes=not args.no_fill,
        threshold=args.threshold,
    )

    stages = list(FIRST_PLACE_STAGES) if args.ablate else None
    written = run_directory(
        args.predictions,
        args.output,
        config=cfg,
        through_stage=args.through_stage,
        stages=stages,
        limit=args.limit,
        overwrite=args.overwrite,
    )

    print(f"wrote {len(written)} file(s) -> {args.output}")
    if args.ablate:
        print("stages:", ", ".join(FIRST_PLACE_STAGES))
        print(
            "\nScore each stage, e.g.:\n"
            f"  python scripts/evaluate.py \\\n"
            f"    --predictions {args.output}/fill \\\n"
            f"    --labels <labels-dir> \\\n"
            f"    --out reports/pp_fill.jsonl"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
