#!/usr/bin/env python3
"""Apply the 1st-place post-processing baseline and optionally score it.

One run writes the final cleaned masks. Pass ``--labels`` to score automatically
and print the metric table — no separate evaluate.py pass required.

    python scripts/run_postprocess.py \\
        --predictions /path/to/predictions \\
        --output      /path/to/pp_out \\
        --labels      /path/to/labels

Ablation (all cumulative stages + one comparison table):

    python scripts/run_postprocess.py \\
        --predictions ... --output ... --labels ... --ablate --limit 12
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

from vesuvius_surface.evaluation.harness import aggregate_by_scroll, discover_cases, evaluate_directory
from vesuvius_surface.evaluation.metric_adapter import MetricUnavailable
from vesuvius_surface.postprocess.first_place import FIRST_PLACE_STAGES, PostprocessConfig
from vesuvius_surface.postprocess.pipeline import run_directory
from vesuvius_surface.postprocess.unmerge import UnmergeConfig
from vesuvius_surface.postprocess import unmerge as unmerge_module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="If set, score the written masks with the official metric and "
        "print the table in this same run. Required when --method unmerge, "
        "since accept/reject needs the metric to gate each cut.",
    )
    p.add_argument(
        "--method",
        choices=["first_place", "unmerge"],
        default="first_place",
        help="first_place: control chain only (default, unchanged behavior). "
        "unmerge: run control, then the novelty metric-guided-unmerge layer on "
        "top of it (src/postprocess/unmerge.py) -- detects thin merge bridges "
        "the control leaves behind and cuts them only where the official metric "
        "improves on that volume. Writes control/, unmerge_proposed/ (cut applied "
        "unconditionally, for inspection), and unmerge_accepted/ (the gated "
        "output) under --output.",
    )
    p.add_argument(
        "--through-stage",
        choices=FIRST_PLACE_STAGES,
        default="fill",
        help="Stop after this cumulative stage (default: full chain).",
    )
    p.add_argument(
        "--ablate",
        action="store_true",
        help="Write every cumulative stage under output/<stage>/ and, with "
        "--labels, print one ablation table across stages.",
    )
    p.add_argument(
        "--scroll-groups",
        type=Path,
        default=Path(
            "/mnt/workspace/code/nnUNet_raw/Dataset100_VesuviusSurface/scroll_groups.json"
        ),
    )
    p.add_argument("--scores-out", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-component-size", type=int, default=20_000)
    p.add_argument("--closing-radius", type=int, default=3)
    p.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26])
    p.add_argument("--no-closing", action="store_true")
    p.add_argument("--no-patching", action="store_true")
    p.add_argument("--no-plug", action="store_true")
    p.add_argument("--no-fill", action="store_true")
    p.add_argument(
        "--erosion-radius",
        type=int,
        default=1,
        help="[unmerge] Ball radius used to find seed masses via erosion. "
        "Calibrated against measured sheet thickness (see UnmergeConfig docstring) "
        "-- 1 is the default; 2 erodes real sheets away entirely on this dataset.",
    )
    p.add_argument("--min-seed-size", type=int, default=100, help="[unmerge]")
    p.add_argument("--min-piece-size", type=int, default=100, help="[unmerge]")
    p.add_argument("--cut-width", type=int, default=1, help="[unmerge]")
    p.add_argument(
        "--min-score-delta",
        type=float,
        default=0.0,
        help="[unmerge] Accept a volume's cut-set only if score improves by at "
        "least this much (default: any non-negative improvement).",
    )
    p.add_argument(
        "--max-candidates-per-volume",
        type=int,
        default=None,
        help="[unmerge] Optional safety cap on cuts proposed per volume.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def load_scroll_map(path: Path) -> dict[str, str]:
    if not path.exists():
        logging.warning("No scroll groups at %s; per-scroll breakdown disabled", path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload and isinstance(next(iter(payload.values())), list):
        return {str(case): str(scroll) for scroll, cases in payload.items() for case in cases}
    return {str(case): str(scroll) for case, scroll in payload.items()}


def score_dir(
    predictions: Path,
    labels: Path,
    scores_path: Path,
    *,
    scroll_map: dict[str, str],
    limit: int | None,
    resume: bool,
) -> dict[str, dict[str, float]]:
    case_ids = None
    if limit is not None:
        case_ids = discover_cases(predictions)[:limit]
    scores = evaluate_directory(
        predictions,
        labels,
        scores_path,
        case_ids=case_ids,
        scroll_map=scroll_map,
        resume=resume,
    )
    return aggregate_by_scroll(scores)


def print_scroll_table(summary: dict[str, dict[str, float]], title: str) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'scroll':>10}  {'n':>4}  {'topo':>7}  {'sdice':>7}  {'voi':>7}  "
        f"{'split':>7}  {'merge':>7}  {'SCORE':>7}"
    )
    print(header)
    print("-" * len(header))
    for scroll, row in summary.items():
        print(
            f"{scroll:>10}  {int(row['n']):>4}  "
            f"{row['topo_score']:>7.4f}  {row['surface_dice']:>7.4f}  "
            f"{row['voi_score']:>7.4f}  {row['voi_split']:>7.4f}  "
            f"{row['voi_merge']:>7.4f}  {row['score']:>7.4f}"
        )


def print_ablation_table(by_stage: dict[str, dict[str, dict[str, float]]]) -> None:
    """One row per stage using the ALL aggregate."""
    print("\n=== ablation (ALL scrolls) ===")
    header = (
        f"{'stage':>14}  {'n':>4}  {'topo':>7}  {'sdice':>7}  {'voi':>7}  "
        f"{'split':>7}  {'merge':>7}  {'SCORE':>7}  {'dSCORE':>7}"
    )
    print(header)
    print("-" * len(header))
    baseline = None
    for stage in FIRST_PLACE_STAGES:
        if stage not in by_stage or "ALL" not in by_stage[stage]:
            continue
        row = by_stage[stage]["ALL"]
        if baseline is None:
            baseline = row["score"]
        delta = row["score"] - baseline
        print(
            f"{stage:>14}  {int(row['n']):>4}  "
            f"{row['topo_score']:>7.4f}  {row['surface_dice']:>7.4f}  "
            f"{row['voi_score']:>7.4f}  {row['voi_split']:>7.4f}  "
            f"{row['voi_merge']:>7.4f}  {row['score']:>7.4f}  "
            f"{delta:>+7.4f}"
        )


def print_unmerge_delta_table(
    control_summary: dict[str, dict[str, float]],
    accepted_summary: dict[str, dict[str, float]],
) -> None:
    """One row per scroll: control vs unmerge_accepted, official metric only
    (this is the novelty layer's own comparison, separate from the first-place
    ablation table above)."""
    print("\n=== unmerge vs control (novelty layer only) ===")
    header = (
        f"{'scroll':>10}  {'n':>4}  {'ctrl_SCORE':>10}  {'unm_SCORE':>10}  {'dSCORE':>8}  "
        f"{'ctrl_merge':>10}  {'unm_merge':>10}  {'dMERGE':>8}"
    )
    print(header)
    print("-" * len(header))
    for scroll in control_summary:
        if scroll not in accepted_summary:
            continue
        c, a = control_summary[scroll], accepted_summary[scroll]
        print(
            f"{scroll:>10}  {int(c['n']):>4}  "
            f"{c['score']:>10.4f}  {a['score']:>10.4f}  {a['score'] - c['score']:>+8.4f}  "
            f"{c['voi_merge']:>10.4f}  {a['voi_merge']:>10.4f}  {a['voi_merge'] - c['voi_merge']:>+8.4f}"
        )


def run_unmerge_method(args: argparse.Namespace) -> int:
    """--method unmerge: control, then the metric-guided-unmerge novelty layer
    on top of it. Separate function so this path stays clearly distinct from
    the first_place-only path above, matching "novelty on top of control, not
    a reimplementation of it"."""
    if args.labels is None:
        print("ERROR: --method unmerge requires --labels (accept/reject needs the metric)",
              file=sys.stderr)
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
    control_dir = args.output / "control"
    written = run_directory(
        args.predictions,
        control_dir,
        config=cfg,
        through_stage="fill",
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"[control] wrote {len(written)} file(s) -> {control_dir}")

    unmerge_cfg = UnmergeConfig(
        erosion_radius=args.erosion_radius,
        min_seed_size=args.min_seed_size,
        min_piece_size=args.min_piece_size,
        cut_width=args.cut_width,
        connectivity=args.connectivity,
        min_score_delta=args.min_score_delta,
        max_candidates_per_volume=args.max_candidates_per_volume,
    )
    unmerge_summary = unmerge_module.run_directory(
        control_dir,
        args.labels,
        args.output,
        config=unmerge_cfg,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(
        f"[unmerge] {unmerge_summary['n_volumes']} volume(s), "
        f"{unmerge_summary['n_volumes_with_candidate_cuts']} with candidate cuts, "
        f"{unmerge_summary['n_volumes_accepted']} accepted"
    )

    scroll_map = load_scroll_map(args.scroll_groups)
    scores_root = args.scores_out or (args.output / "scores")
    scores_root.mkdir(parents=True, exist_ok=True)

    try:
        control_summary = score_dir(
            control_dir, args.labels, scores_root / "control.jsonl",
            scroll_map=scroll_map, limit=args.limit, resume=not args.no_resume,
        )
        accepted_summary = score_dir(
            args.output / "unmerge_accepted", args.labels, scores_root / "unmerge_accepted.jsonl",
            scroll_map=scroll_map, limit=args.limit, resume=not args.no_resume,
        )
    except MetricUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print_scroll_table(control_summary, "control")
    print_scroll_table(accepted_summary, "unmerge_accepted")
    print_unmerge_delta_table(control_summary, accepted_summary)

    table_path = scores_root / "unmerge_vs_control_summary.json"
    table_path.write_text(
        json.dumps(
            {
                "control": control_summary.get("ALL", {}),
                "unmerge_accepted": accepted_summary.get("ALL", {}),
                "unmerge_run_summary": unmerge_summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nsummary json: {table_path}")
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if not args.predictions.is_dir():
        print(f"ERROR: predictions dir not found: {args.predictions}", file=sys.stderr)
        return 1
    if args.labels is not None and not args.labels.is_dir():
        print(f"ERROR: labels dir not found: {args.labels}", file=sys.stderr)
        return 1

    if args.method == "unmerge":
        return run_unmerge_method(args)

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

    if args.labels is None:
        final = args.output if not args.ablate else args.output / args.through_stage
        print(f"\nFinal masks: {final}")
        print("(pass --labels DIR to score in this same run)")
        return 0

    scroll_map = load_scroll_map(args.scroll_groups)
    scores_root = args.scores_out or (args.output / "scores")
    scores_root.mkdir(parents=True, exist_ok=True)

    try:
        if args.ablate:
            by_stage: dict[str, dict[str, dict[str, float]]] = {}
            for stage in FIRST_PLACE_STAGES:
                pred_dir = args.output / stage
                if not pred_dir.is_dir():
                    continue
                summary = score_dir(
                    pred_dir,
                    args.labels,
                    scores_root / f"{stage}.jsonl",
                    scroll_map=scroll_map,
                    limit=args.limit,
                    resume=not args.no_resume,
                )
                by_stage[stage] = summary
                print_scroll_table(summary, f"stage={stage}")
            print_ablation_table(by_stage)
            table_path = scores_root / "ablation_summary.json"
            table_path.write_text(
                json.dumps(
                    {s: by_stage[s].get("ALL", {}) for s in by_stage},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"\nablation json: {table_path}")
        else:
            summary = score_dir(
                args.output,
                args.labels,
                scores_root / f"{args.through_stage}.jsonl",
                scroll_map=scroll_map,
                limit=args.limit,
                resume=not args.no_resume,
            )
            print_scroll_table(summary, f"stage={args.through_stage}")
            print(f"\nFinal masks : {args.output}")
            print(f"scores      : {scores_root / (args.through_stage + '.jsonl')}")
    except MetricUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
