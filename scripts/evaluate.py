#!/usr/bin/env python3
"""Score predictions against the official competition metric.

    Score = 0.30*TopoScore + 0.35*SurfaceDice@2.0 + 0.35*VOI_score

Results append to a JSONL file as each case finishes, so a long run can be
interrupted and resumed.

    python scripts/evaluate.py \\
        --predictions /mnt/workspace/code/subsets/m7_holdout/predictions \\
        --labels      /mnt/workspace/code/subsets/m7_holdout/labels \\
        --out         reports/m7_holdout_scores.jsonl
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

from evaluation.harness import aggregate_by_scroll, evaluate_directory
from evaluation.metric_adapter import MetricUnavailable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("reports/scores.jsonl"))
    p.add_argument(
        "--scroll-groups",
        type=Path,
        default=Path("/mnt/workspace/code/nnUNet_raw/Dataset100_VesuviusSurface/scroll_groups.json"),
        help="Maps case id -> scroll id, for the per-scroll breakdown.",
    )
    p.add_argument(
        "--ignore-mode",
        choices=["neutralize", "background"],
        default="neutralize",
        help="How to handle label 2 (see src/evaluation/harness.py).",
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def load_scroll_map(path: Path) -> dict[str, str]:
    if not path.exists():
        logging.warning("No scroll groups at %s; per-scroll breakdown disabled", path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Accept either {case: scroll} or {scroll: [cases]}.
    if payload and isinstance(next(iter(payload.values())), list):
        return {str(case): str(scroll) for scroll, cases in payload.items() for case in cases}
    return {str(case): str(scroll) for case, scroll in payload.items()}


def print_table(summary: dict[str, dict[str, float]]) -> None:
    header = f"{'scroll':>10}  {'n':>4}  {'topo':>7}  {'sdice':>7}  {'voi':>7}  {'score':>7}"
    print()
    print(header)
    print("-" * len(header))
    for scroll, row in summary.items():
        print(
            f"{scroll:>10}  {int(row['n']):>4}  "
            f"{row['topo_score']:>7.4f}  {row['surface_dice']:>7.4f}  "
            f"{row['voi_score']:>7.4f}  {row['composite']:>7.4f}"
        )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    for label, path in (("predictions", args.predictions), ("labels", args.labels)):
        if not path.is_dir():
            print(f"ERROR: {label} dir not found: {path}", file=sys.stderr)
            return 1

    try:
        scores = evaluate_directory(
            args.predictions,
            args.labels,
            args.out,
            scroll_map=load_scroll_map(args.scroll_groups),
            ignore_mode=args.ignore_mode,
            resume=not args.no_resume,
        )
    except MetricUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not scores:
        print("No cases scored.", file=sys.stderr)
        return 1

    summary = aggregate_by_scroll(scores)
    print_table(summary)
    print(f"\nignore mode : {args.ignore_mode}")
    print(f"per-case    : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
