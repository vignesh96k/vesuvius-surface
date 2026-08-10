#!/usr/bin/env python
"""Score a directory of prediction .tif files against ground truth using the Vesuvius
Surface Detection competition metric (Score = 0.30*TopoScore + 0.35*SurfaceDice + 0.35*VOI).

Two backends:
  --backend official (default): the real organizer scorer (vesuvius_evaluation.official_score),
      wraps the topometrics-3d package. Requires `conda activate vesuvius_eval` and having
      run scripts/install_topometrics.sh once. Slow (~1min/volume at 320^3, matches the
      competition's own "expect hours" warning) but numerically matches the leaderboard.
  --backend approx: fast pure-Python approximation (vesuvius_evaluation.approx_score), same formula
      but a different (non-official) Betti-number algorithm -- use for quick iteration only.

Example:
    python scripts/evaluate.py \\
        --pred-dir /mnt/workspace/code/outputs/training_run/predictions_tiff \\
        --gt-dir /mnt/workspace/code/datasets/vesuvius-challenge-surface-detection/train_labels
"""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory of prediction .tif files")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory of ground-truth .tif files")
    parser.add_argument("--backend", type=str, choices=["official", "approx"], default="official")
    parser.add_argument("--surface-tolerance", "--tau", dest="tau", type=float, default=2.0)
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated case IDs (default: all .tif in pred-dir)")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write full per-case results as JSON")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes for --backend official (CPU-bound, embarrassingly parallel across cases). Default 1 (sequential).")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_ids = args.cases.split(",") if args.cases else None

    if args.backend == "official":
        from vesuvius_evaluation.official_score import score_directory

        result = score_directory(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            surface_tolerance=args.tau,
            case_ids=case_ids,
            verbose=not args.quiet,
            n_workers=args.workers,
        )
        mean_score = result["mean_score"]
    else:
        from pathlib import Path

        from vesuvius_evaluation.approx_score import score_from_files

        pred_dir = Path(args.pred_dir)
        gt_dir = Path(args.gt_dir)
        if case_ids is None:
            case_ids = sorted(p.stem for p in pred_dir.glob("*.tif"))

        per_case = {}
        for cid in case_ids:
            r = score_from_files(pred_dir / f"{cid}.tif", gt_dir / f"{cid}.tif", tau=args.tau, verbose=not args.quiet)
            per_case[cid] = r
        mean_score = sum(v["score"] for v in per_case.values()) / len(per_case)
        result = {"mean_score": mean_score, "per_case": per_case}

    print(f"\n{'=' * 60}")
    print(f"Backend: {args.backend}")
    print(f"Cases scored: {len(result['per_case'])}")
    print(f"Mean competition score: {mean_score:.6f}")
    print(f"{'=' * 60}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Full results written to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
