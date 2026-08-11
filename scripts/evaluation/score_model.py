#!/usr/bin/env python3
"""Score one or more prediction directories against ground truth, official leaderboard
metric, parallelized. Consolidates 5 near-identical ad-hoc scripts written over the course of
this project's real experiments (score_1000ep_loso.py, score_a2_700ep_skelrecall.py,
score_a3_skelrecall_pp.py, score_b3_lastlayers_pp.py, score_single_model.py -- their original
forms are preserved in git history) into one parametrized tool.

Every real comparison table in this project was produced by a script matching this
one's shape. Usage:

    # Score a single condition (accepts .tif segmentation or .npz probability files):
    python scripts/evaluation/score_model.py --gt-dir data/train_labels \\
        --pred-dir baseline=outputs/some_model/validation

    # Score multiple conditions against the same case set in one pass (e.g. before/after
    # postprocessing), each on its own line of the printed comparison table:
    python scripts/evaluation/score_model.py --gt-dir data/train_labels \\
        --pred-dir zero_shot=outputs/zero_shot/validation \\
        --pred-dir finetuned=outputs/finetuned/validation --postprocess finetuned

    # Restrict to a specific LOSO fold's held-out case ids instead of every file present:
    python scripts/evaluation/score_model.py --gt-dir data/train_labels \\
        --pred-dir model=outputs/model/validation \\
        --splits-file data/preprocessed/splits_final.json --fold 0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

# Must happen before numpy is imported anywhere in this process -- see
# packages/vesuvius_evaluation/src/vesuvius_evaluation/official_score.py's own docstring for
# why (thread-oversubscription stall, a real bug hit and fixed earlier in this project).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vesuvius_surface.postprocess.first_place import apply_first_place, PostprocessConfig  # noqa: E402

try:
    import topometrics.leaderboard as leaderboard
except ImportError:
    leaderboard = None


def _load_prediction(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        probs = np.load(path)["probabilities"].astype(np.float32)
        return np.argmax(probs, axis=0).astype(np.uint8)
    return tifffile.imread(path)


def _score_unit(args: tuple) -> tuple:
    cid, cond_name, pred_dir, postprocess, gt_dir = args
    gt = tifffile.imread(gt_dir / f"{cid}.tif")

    pred_path = pred_dir / f"{cid}.npz"
    if not pred_path.exists():
        pred_path = pred_dir / f"{cid}.tif"
    pred = _load_prediction(pred_path)
    if postprocess:
        pred = apply_first_place(pred, PostprocessConfig())

    if leaderboard is None:
        raise RuntimeError(
            "topometrics not importable -- activate the vesuvius_eval env and install "
            "packages/vesuvius_evaluation first (see environment-eval.yml)"
        )
    r = leaderboard.compute_leaderboard_score(
        predictions=pred, labels=gt,
        dims=(0, 1, 2), spacing=(1.0, 1.0, 1.0), surface_tolerance=2.0,
        voi_connectivity=26, voi_transform="one_over_one_plus", voi_alpha=0.3,
        combine_weights=(0.3, 0.35, 0.35), fg_threshold=None, ignore_label=2, ignore_mask=None,
    )
    return cid, cond_name, (r.score, r.surface_dice, r.topo.toposcore, r.voi.voi_score)


def discover_case_ids(pred_dir: Path) -> list[str]:
    ids = {p.stem for p in pred_dir.glob("*.tif")} | {p.stem for p in pred_dir.glob("*.npz")}
    return sorted(ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-dir", type=Path, required=True, help="ground truth .tif directory")
    parser.add_argument(
        "--pred-dir", action="append", required=True, metavar="NAME=PATH",
        help="a named condition's prediction directory; repeat for multiple conditions",
    )
    parser.add_argument(
        "--postprocess", action="append", default=[], metavar="NAME",
        help="apply the 1st-place postprocessing chain to this condition before scoring "
             "(repeatable; condition name must match one given to --pred-dir)",
    )
    parser.add_argument("--splits-file", type=Path, default=None, help="nnU-Net splits_final.json, to restrict to one fold's held-out cases")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8, help="8 is the proven-safe ceiling on a 117GB box for this metric's memory use")
    parser.add_argument("--out", type=Path, default=None, help="pickle path for raw per-case results (default: alongside the first --pred-dir)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    conditions: dict[str, Path] = {}
    for spec in args.pred_dir:
        name, _, path = spec.partition("=")
        if not path:
            print(f"ERROR: --pred-dir must be NAME=PATH, got {spec!r}", file=sys.stderr)
            return 1
        conditions[name] = Path(path)
    postprocess_set = set(args.postprocess)
    unknown = postprocess_set - set(conditions)
    if unknown:
        print(f"ERROR: --postprocess names not in --pred-dir: {unknown}", file=sys.stderr)
        return 1

    if args.splits_file:
        splits = json.loads(args.splits_file.read_text())
        case_ids = sorted(splits[args.fold]["val"])
    else:
        case_ids = discover_case_ids(next(iter(conditions.values())))

    units = [
        (cid, name, pred_dir, name in postprocess_set, args.gt_dir)
        for cid in case_ids
        for name, pred_dir in conditions.items()
    ]
    print(f"scoring {len(units)} units ({len(case_ids)} cases x {len(conditions)} condition(s)), {args.workers} workers")

    results: dict[str, dict[str, tuple]] = {cid: {} for cid in case_ids}
    with Pool(args.workers) as pool:
        for i, (cid, name, values) in enumerate(pool.imap_unordered(_score_unit, units), 1):
            results[cid][name] = values
            if i % 20 == 0:
                print(f"  ...{i}/{len(units)} done", flush=True)

    out_path = args.out or next(iter(conditions.values())).parent / "score_model_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"wrote {out_path}")

    metrics = ["score", "surface_dice", "toposcore", "voi_score"]
    agg = {name: {m: [] for m in metrics} for name in conditions}
    for cid, by_cond in results.items():
        for name, values in by_cond.items():
            for i, m in enumerate(metrics):
                agg[name][m].append(values[i])

    print(f"\n=== n={len(case_ids)} ===")
    header = f"{'metric':<15}" + "".join(f"{name:<32}" for name in conditions)
    print(header)
    for m in metrics:
        row = f"{m:<15}"
        for name in conditions:
            vals = agg[name][m]
            row += f"{np.mean(vals):.4f} (+/-{np.std(vals):.4f})        "
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
