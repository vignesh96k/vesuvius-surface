#!/usr/bin/env python3
"""Write a known, scroll-aware ``splits_final.json``.

Motivation: the published m7 checkpoint records which *fold* it trained on but
not which *split* defined that fold, so no volume is provably outside its
training set and no measurement on it is a clean generalisation estimate.
Training our own model against a split we author removes that ambiguity.

nnU-Net's own default is a plain KFold over sorted case ids with no notion of
scroll, which is how our reconstructed fold 0 ended up with zero cases from
scroll 44430 — the hardest scroll in the dataset, and therefore the one a
validation set most needs to contain.

Three modes:

``stratified`` (default)
    Every fold's validation set receives a proportional share of every scroll.
    Guarantees coverage. Train and validation share scrolls, so scores measure
    "another region of a familiar scroll", which is optimistic but stable.

``scroll-holdout``
    One entire scroll is held out per fold (leave-one-scroll-out over all
    scrolls). Small scrolls (44430 has 16, 53997 has 13) make some folds tiny.

``holdout-scroll``
    Hold out one (or more) named scroll(s) as a single validation fold; train
    on everything else. This is the clean generalisation split — the held-out
    scroll is never seen during training.

        # Hold out scroll 26010 (129 volumes) as the validation set:
        python scripts/make_scroll_split.py --mode holdout-scroll --val-scroll 26010
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.nnunet_splits import list_case_ids, load_scroll_groups

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/mnt/workspace/code/nnUNet_raw/Dataset100_VesuviusSurface"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to <preprocessed>/Dataset100_VesuviusSurface/splits_final.json",
    )
    p.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("/mnt/workspace/code/nnUNet_preprocessed/Dataset100_VesuviusSurface"),
    )
    p.add_argument(
        "--mode",
        choices=["stratified", "scroll-holdout", "holdout-scroll"],
        default="stratified",
    )
    p.add_argument(
        "--val-scroll",
        action="append",
        default=None,
        metavar="SCROLL_ID",
        help="Scroll id(s) to hold out as validation (required for "
        "--mode holdout-scroll). Repeatable. Example: --val-scroll 26010",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the distribution without writing the file.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def stratified_folds(
    case_ids: list[str],
    scroll_map: dict[str, str],
    n_splits: int,
    seed: int,
) -> list[list[str]]:
    """Deal each scroll's cases round-robin so every fold gets a share."""
    by_scroll: dict[str, list[str]] = defaultdict(list)
    for case_id in case_ids:
        by_scroll[scroll_map.get(case_id, "unknown")].append(case_id)

    folds: list[list[str]] = [[] for _ in range(n_splits)]
    rng = random.Random(seed)
    # Offset each scroll's starting fold so small scrolls do not all pile into
    # fold 0 while still distributing evenly.
    for offset, scroll in enumerate(sorted(by_scroll)):
        cases = sorted(by_scroll[scroll])
        rng.shuffle(cases)
        for i, case_id in enumerate(cases):
            folds[(i + offset) % n_splits].append(case_id)
    return [sorted(f) for f in folds]


def scroll_holdout_folds(
    case_ids: list[str],
    scroll_map: dict[str, str],
) -> tuple[list[list[str]], list[str]]:
    by_scroll: dict[str, list[str]] = defaultdict(list)
    for case_id in case_ids:
        by_scroll[scroll_map.get(case_id, "unknown")].append(case_id)
    scrolls = sorted(by_scroll)
    return [sorted(by_scroll[s]) for s in scrolls], scrolls


def named_scroll_holdout(
    case_ids: list[str],
    scroll_map: dict[str, str],
    val_scrolls: list[str],
) -> list[str]:
    """Validation = every case from the named scroll(s); train = the rest."""
    wanted = {str(s) for s in val_scrolls}
    present = {scroll_map.get(c, "unknown") for c in case_ids}
    missing = sorted(wanted - present)
    if missing:
        raise ValueError(
            f"Scroll id(s) not in the dataset: {missing}. "
            f"Available: {sorted(present)}"
        )
    val = sorted(c for c in case_ids if scroll_map.get(c) in wanted)
    if not val:
        raise ValueError(f"No cases found for val scrolls {sorted(wanted)}")
    return val


def print_distribution(
    folds: list[list[str]],
    scroll_map: dict[str, str],
    totals: dict[str, int],
) -> None:
    scrolls = sorted(totals)
    header = "     scroll  " + "  ".join(f"f{i:<4}" for i in range(len(folds))) + "   total"
    print()
    print(header)
    print("-" * len(header))
    for scroll in scrolls:
        counts = [
            sum(1 for c in fold if scroll_map.get(c) == scroll) for fold in folds
        ]
        row = "  ".join(f"{c:<5}" for c in counts)
        print(f"{scroll:>11}  {row}   {totals[scroll]:>5}")
    sizes = "  ".join(f"{len(f):<5}" for f in folds)
    print(f"{'val size':>11}  {sizes}   {sum(len(f) for f in folds):>5}")

    empty = [
        (i, s)
        for i, fold in enumerate(folds)
        for s in scrolls
        if not any(scroll_map.get(c) == s for c in fold)
    ]
    if empty:
        print(
            "\nNOTE: scrolls absent from some validation folds "
            "(expected for holdout-scroll / leave-one-scroll-out):"
        )
        for i, scroll in empty:
            print(f"  fold {i}: {scroll}")
    else:
        print("\nEvery scroll appears in every validation fold.")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if not args.dataset_dir.is_dir():
        print(f"ERROR: dataset dir not found: {args.dataset_dir}", file=sys.stderr)
        return 1

    case_ids = list_case_ids(args.dataset_dir)
    groups_path = args.dataset_dir / "scroll_groups.json"
    scroll_map = load_scroll_groups(groups_path)
    if not scroll_map:
        print(
            f"ERROR: no scroll_groups.json at {groups_path}. "
            "Re-run scripts/export_nnunet.py.",
            file=sys.stderr,
        )
        return 1

    totals: dict[str, int] = defaultdict(int)
    for case_id in case_ids:
        totals[scroll_map.get(case_id, "unknown")] += 1

    if args.mode == "stratified":
        val_folds = stratified_folds(case_ids, scroll_map, args.n_splits, args.seed)
        note = f"scroll-stratified, {args.n_splits} folds, seed={args.seed}"
    elif args.mode == "scroll-holdout":
        val_folds, scrolls = scroll_holdout_folds(case_ids, scroll_map)
        note = f"leave-one-scroll-out over {len(scrolls)} scrolls: {', '.join(scrolls)}"
    else:
        if not args.val_scroll:
            print(
                "ERROR: --mode holdout-scroll requires --val-scroll SCROLL_ID\n"
                "Available scrolls and sizes: "
                + ", ".join(f"{s}={n}" for s, n in sorted(totals.items())),
                file=sys.stderr,
            )
            return 1
        try:
            val = named_scroll_holdout(case_ids, scroll_map, args.val_scroll)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        val_folds = [val]
        held = ", ".join(args.val_scroll)
        note = (
            f"hold out scroll(s) [{held}] as fold 0 "
            f"({len(val)} val / {len(case_ids) - len(val)} train)"
        )

    print(f"cases      : {len(case_ids)}")
    print(f"mode       : {args.mode}")
    print(f"note       : {note}")
    print_distribution(val_folds, scroll_map, dict(totals))

    all_ids = set(case_ids)
    splits = [
        {"train": sorted(all_ids - set(val)), "val": sorted(val)} for val in val_folds
    ]

    for i, split in enumerate(splits):
        overlap = set(split["train"]) & set(split["val"])
        if overlap:
            print(f"ERROR: fold {i} has {len(overlap)} overlapping cases", file=sys.stderr)
            return 1
        if set(split["train"]) | set(split["val"]) != all_ids:
            print(f"ERROR: fold {i} does not cover all cases", file=sys.stderr)
            return 1

    out = args.out or (args.preprocessed_dir / "splits_final.json")
    if args.dry_run:
        print(f"\n(dry run; would write {out})")
        print(f"fold 0     : {len(splits[0]['train'])} train / {len(splits[0]['val'])} val")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        backup = out.with_suffix(".json.bak")
        backup.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("Backed up existing split -> %s", backup)
    out.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    print(f"\nwrote      : {out}")
    print(f"fold 0     : {len(splits[0]['train'])} train / {len(splits[0]['val'])} val")
    print(
        "\nnnU-Net reads this file instead of generating its own. Train with\n"
        "  -f 0\n"
        "so the held-out scroll is the validation set."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
