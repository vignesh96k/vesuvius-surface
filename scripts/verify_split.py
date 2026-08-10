#!/usr/bin/env python3
"""Check whether our reconstructed fold split can be trusted.

The m7 snapshot ships no ``splits_final.json``, so `scripts/nnunet_folds.py`
regenerates the split from nnU-Net's default seed over *our* sorted case ids.
That is only valid if their case list and sort order match ours.

``dataset_fingerprint.json`` records per-case geometry in sorted-case-id order,
so comparing the two sequences tests exactly that assumption.

Our own `plan_and_preprocess` run wrote a fingerprint with the identical
nnU-Net code path, so preferring that over raw TIFF shapes makes this a
like-for-like comparison (nnU-Net records shapes *after* cropping to the
nonzero region, which raw shapes do not account for).

    python scripts/verify_split.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vesuvius_surface.data.io import probe_volume
from vesuvius_surface.data.nnunet_splits import list_case_ids

logger = logging.getLogger(__name__)

Shape = tuple[int, ...]
SHAPE_KEYS = ("shapes_after_crop", "shapes_after_cropping", "shapes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fingerprint",
        type=Path,
        default=Path("/mnt/workspace/code/pretrained/surface_m7_nnunet/dataset_fingerprint.json"),
        help="Their fingerprint, from the downloaded snapshot.",
    )
    p.add_argument(
        "--our-fingerprint",
        type=Path,
        default=Path(
            "/mnt/workspace/code/nnUNet_preprocessed/Dataset100_VesuviusSurface/dataset_fingerprint.json"
        ),
        help="Ours, written by nnUNetv2_plan_and_preprocess. Falls back to "
        "probing raw TIFFs when absent (weaker: raw shapes are uncropped).",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/mnt/workspace/code/nnUNet_raw/Dataset100_VesuviusSurface"),
    )
    p.add_argument("--max-examples", type=int, default=5)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def extract_shapes(payload: dict) -> tuple[list[Shape], str]:
    """Pull the per-case shape list out of a dataset fingerprint."""
    for key in SHAPE_KEYS:
        value = payload.get(key)
        if isinstance(value, list) and value:
            return [tuple(int(v) for v in s) for s in value], key
    raise KeyError(
        "No shape list found in fingerprint. Top-level keys: " + ", ".join(sorted(payload))
    )


def load_our_shapes(args: argparse.Namespace, case_ids: list[str]) -> tuple[list[Shape], bool]:
    """Return (shapes, comparable), where comparable marks a like-for-like source."""
    if args.our_fingerprint.is_file():
        payload = json.loads(args.our_fingerprint.read_text(encoding="utf-8"))
        shapes, key = extract_shapes(payload)
        print(f"our source      : {args.our_fingerprint.name} [{key}] (post-crop)")
        return shapes, True

    print(f"our source      : raw TIFF headers (no fingerprint at {args.our_fingerprint})")
    images_dir = args.dataset_dir / "imagesTr"
    iterator: object = case_ids
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(case_ids, desc="probe", unit="vol")
    except ImportError:
        pass
    shapes = [
        tuple(int(v) for v in probe_volume(images_dir / f"{cid}_0000.tif")["shape"])
        for cid in iterator  # type: ignore[union-attr]
    ]
    return shapes, False


def report_mismatch(
    our_shapes: list[Shape],
    their_shapes: list[Shape],
    case_ids: list[str],
    *,
    comparable: bool,
    max_examples: int,
) -> int:
    same_multiset = Counter(our_shapes) == Counter(their_shapes)

    if same_multiset:
        print(
            "\nORDER MISMATCH: the two runs contain the same multiset of shapes\n"
            "but in a different sequence. Same data, different sort order — the\n"
            "K-fold assignment does NOT transfer. Do not trust the holdout."
        )
    elif comparable:
        print(
            "\nMISMATCH: post-crop geometry differs between the two fingerprints,\n"
            "computed by the same nnU-Net code path. Their dataset is not ours,\n"
            "so the reconstructed split cannot be trusted."
        )
    else:
        print(
            "\nINCONCLUSIVE: compared raw (uncropped) shapes against their\n"
            "post-crop shapes, so a uniform difference is expected. Run\n"
            "nnUNetv2_plan_and_preprocess to generate our own fingerprint,\n"
            "then re-run for a like-for-like comparison."
        )

    print("\nExamples (index: ours vs theirs):")
    shown = 0
    for i, (a, b) in enumerate(zip(our_shapes, their_shapes)):
        if a == b:
            continue
        print(f"  [{i}] {case_ids[i]}: {a} vs {b}")
        shown += 1
        if shown >= max_examples:
            break
    return 3 if not comparable else 2


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if not args.fingerprint.is_file():
        print(f"ERROR: fingerprint not found: {args.fingerprint}", file=sys.stderr)
        print("Re-run scripts/fetch_pretrained_m7.py to download the snapshot.", file=sys.stderr)
        return 1

    payload = json.loads(args.fingerprint.read_text(encoding="utf-8"))
    try:
        their_shapes, key = extract_shapes(payload)
    except KeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    case_ids = list_case_ids(args.dataset_dir)
    print(f"their source    : {args.fingerprint.name} [{key}]")

    try:
        our_shapes, comparable = load_our_shapes(args, case_ids)
    except KeyError as exc:
        print(f"ERROR reading our fingerprint: {exc}", file=sys.stderr)
        return 1

    print(f"their cases     : {len(their_shapes)}")
    print(f"our cases       : {len(our_shapes)}")

    if len(their_shapes) != len(our_shapes):
        print(
            "\nMISMATCH: different number of cases. The reconstructed split "
            "cannot be trusted — their dataset is not ours."
        )
        return 2

    n = len(our_shapes)
    n_match = sum(1 for a, b in zip(our_shapes, their_shapes) if a == b)
    print(f"\nexact matches   : {n_match}/{n} ({100 * n_match / n:.1f}%)")

    if n_match == n:
        verdict = "VERIFIED" if comparable else "CONSISTENT"
        print(
            f"\n{verdict} ordering: our sorted case list produces the same per-case\n"
            "geometry sequence as theirs, so the K-fold assignment is identical.\n"
            "Caveat: this does not prove they used the *default* split rather\n"
            "than a custom splits_final.json."
        )
        return 0

    return report_mismatch(
        our_shapes,
        their_shapes,
        case_ids,
        comparable=comparable,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    raise SystemExit(main())
