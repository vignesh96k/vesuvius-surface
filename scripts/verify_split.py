#!/usr/bin/env python3
"""Check whether our reconstructed fold split can be trusted.

The m7 snapshot ships no ``splits_final.json``, so `scripts/nnunet_folds.py`
regenerates the split from nnU-Net's default seed over *our* sorted case ids.
That is only valid if their case list and sort order match ours.

``dataset_fingerprint.json`` records per-case geometry in sorted-case-id order.
Comparing their entries against ours tests exactly that assumption.

    python scripts/verify_split.py

Match  -> our ordering equals theirs; the reconstruction is sound provided they
          also used the default split.
Differ -> inconclusive rather than disproving: nnU-Net records shapes *after*
          cropping to the nonzero region, so a mismatch can be cropping rather
          than a different case list. Check the reported examples.
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

from data.io import probe_volume
from data.nnunet_splits import list_case_ids

logger = logging.getLogger(__name__)

SHAPE_KEYS = ("shapes_after_crop", "shapes", "shapes_after_cropping")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fingerprint",
        type=Path,
        default=Path("/mnt/workspace/code/pretrained/surface_m7_nnunet/dataset_fingerprint.json"),
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/mnt/workspace/code/nnUNet_raw/Dataset100_VesuviusSurface"),
    )
    p.add_argument("--max-examples", type=int, default=5)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def extract_shapes(payload: dict) -> tuple[list[tuple[int, ...]], str]:
    """Pull the per-case shape list out of a dataset fingerprint."""
    for key in SHAPE_KEYS:
        if key in payload and isinstance(payload[key], list):
            return [tuple(int(v) for v in s) for s in payload[key]], key

    # Some versions nest per-case dicts keyed by identifier.
    for key, value in payload.items():
        if isinstance(value, dict) and value:
            first = next(iter(value.values()))
            if isinstance(first, dict) and any(k in first for k in SHAPE_KEYS):
                shapes = [
                    tuple(int(v) for v in first_val[k])
                    for _, first_val in sorted(value.items())
                    for k in SHAPE_KEYS
                    if k in first_val
                ]
                return shapes, f"{key}[*]"

    raise KeyError(
        "No shape list found in fingerprint. Top-level keys: " + ", ".join(sorted(payload))
    )


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
    print(f"fingerprint key : {key}")
    print(f"their cases     : {len(their_shapes)}")
    print(f"our cases       : {len(case_ids)}")

    if len(their_shapes) != len(case_ids):
        print(
            "\nMISMATCH: different number of cases. The reconstructed split "
            "cannot be trusted — their dataset is not ours."
        )
        return 2

    images_dir = args.dataset_dir / "imagesTr"
    our_shapes: list[tuple[int, ...]] = []
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(case_ids, desc="probe", unit="vol")
    except ImportError:
        iterator = case_ids

    for case_id in iterator:
        info = probe_volume(images_dir / f"{case_id}_0000.tif")
        our_shapes.append(tuple(int(v) for v in info["shape"]))

    matches = [i for i, (a, b) in enumerate(zip(our_shapes, their_shapes)) if a == b]
    n_match = len(matches)
    print(f"\nexact matches   : {n_match}/{len(case_ids)} ({100 * n_match / len(case_ids):.1f}%)")

    if n_match == len(case_ids):
        print(
            "\nVERIFIED ordering: our sorted case list produces the same per-case\n"
            "geometry sequence as theirs, so the K-fold assignment is identical.\n"
            "Caveat: this does not prove they used the *default* split rather\n"
            "than a custom splits_final.json."
        )
        return 0

    print("\nINCONCLUSIVE: shapes differ. nnU-Net stores shapes after cropping to")
    print("the nonzero region, so this may be cropping rather than a different")
    print("case list. Examples (index: ours vs theirs):")
    shown = 0
    for i, (a, b) in enumerate(zip(our_shapes, their_shapes)):
        if a == b:
            continue
        print(f"  [{i}] {case_ids[i]}: {a} vs {b}")
        shown += 1
        if shown >= args.max_examples:
            break
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
