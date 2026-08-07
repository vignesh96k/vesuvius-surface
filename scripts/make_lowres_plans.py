#!/usr/bin/env python3
"""Author a ``3d_lowres`` configuration for Track A topology training.

Track A (Skeleton Recall + affinity auxiliary) trains **from scratch** on
``3d_lowres``. Coarser spacing means each 128^3 patch covers more of the
scroll, which is what long-range affinities need. nnU-Net often skips
generating ``3d_lowres`` when volumes are ~320^3, so this script adds it by
hand.

    python scripts/make_lowres_plans.py --dry-run
    python scripts/make_lowres_plans.py
    nnUNetv2_preprocess -d 100 -c 3d_lowres -plans_name nnUNetPlans

Then train with ``scripts/nnunet_train_topology.sh`` (defaults to 3d_lowres).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("/mnt/workspace/code/nnUNet_preprocessed/Dataset100_VesuviusSurface"),
    )
    p.add_argument(
        "--plans-name",
        default="nnUNetPlans",
        help="Plans file stem inside the preprocessed dataset dir.",
    )
    p.add_argument("--base-config", default="3d_fullres")
    p.add_argument("--config-name", default="3d_lowres")
    p.add_argument(
        "--spacing-factor",
        type=float,
        default=2.0,
        help="Multiply fullres spacing by this (isotropic). Default 2 → "
        "resampled volumes ~160^3 when fullres median is ~320^3.",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        nargs="+",
        default=None,
        help="Defaults to the base config's patch size. 1 or 3 ints.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to the base config's batch size.",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def normalize_patch(values: list[int]) -> list[int]:
    if len(values) == 1:
        return [values[0]] * 3
    if len(values) == 3:
        return list(values)
    raise ValueError("--patch-size takes 1 or 3 integers")


def stride_product(arch_kwargs: dict[str, Any]) -> list[int]:
    strides = arch_kwargs.get("strides") or []
    if not strides:
        return [1, 1, 1]
    factors = [1, 1, 1]
    for stage in strides:
        stage_list = [stage] * 3 if isinstance(stage, int) else list(stage)
        for axis in range(3):
            factors[axis] *= int(stage_list[axis])
    return factors


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    plans_path = args.preprocessed_dir / f"{args.plans_name}.json"
    if not plans_path.is_file():
        print(f"ERROR: plans not found: {plans_path}", file=sys.stderr)
        print(
            "Run scripts/nnunet_setup_and_preprocess.sh first (3d_fullres).",
            file=sys.stderr,
        )
        return 1

    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    configs = plans.setdefault("configurations", {})
    base = configs.get(args.base_config)
    if base is None:
        print(
            f"ERROR: config {args.base_config!r} not in plans. Available: "
            + ", ".join(sorted(configs)),
            file=sys.stderr,
        )
        return 1

    base_spacing = [float(x) for x in base.get("spacing", [1.0, 1.0, 1.0])]
    lowres_spacing = [s * args.spacing_factor for s in base_spacing]

    base_patch = [int(v) for v in base.get("patch_size", [128, 128, 128])]
    try:
        patch = normalize_patch(args.patch_size) if args.patch_size else list(base_patch)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    arch_kwargs = (base.get("architecture") or {}).get("arch_kwargs", {})
    factors = stride_product(arch_kwargs)
    bad = [
        (axis, patch[axis], factors[axis])
        for axis in range(3)
        if factors[axis] and patch[axis] % factors[axis]
    ]
    if bad:
        print("ERROR: patch size must be divisible by the total downsampling factor.")
        for axis, value, factor in bad:
            print(f"  axis {axis}: {value} % {factor} != 0")
        return 1

    batch_size = (
        args.batch_size if args.batch_size is not None else int(base.get("batch_size", 2))
    )

    # Own data_identifier so preprocess writes a separate folder; fullres data
    # is not reused (different target spacing).
    data_identifier = f"{args.plans_name}_{args.config_name}"

    base_median = base.get("median_image_size_in_voxels")
    if base_median is not None and len(base_median) == 3:
        # Rough estimate after isotropic downsampling by spacing_factor.
        median = [max(1.0, float(m) / args.spacing_factor) for m in base_median]
    else:
        median = None

    new_config: dict[str, Any] = {
        "inherits_from": args.base_config,
        "data_identifier": data_identifier,
        "spacing": lowres_spacing,
        "patch_size": patch,
        "batch_size": batch_size,
    }
    if median is not None:
        new_config["median_image_size_in_voxels"] = median

    print(f"plans        : {args.plans_name}")
    print(f"base         : {args.base_config}")
    print(f"new config   : {args.config_name}")
    print(f"spacing      : {base_spacing} -> {lowres_spacing} (x{args.spacing_factor:g})")
    print(f"patch size   : {patch}")
    print(f"batch size   : {batch_size}")
    print(f"data id      : {data_identifier}")
    if median is not None:
        print(f"median est.  : {[round(m, 1) for m in median]} voxels (from fullres / factor)")

    if args.config_name in configs and not args.dry_run:
        print(f"\nNOTE: replacing existing config {args.config_name!r}")

    if args.dry_run:
        print("\n(dry run; plans not written)")
        print(
            "\nNext after a real run:\n"
            f"  nnUNetv2_preprocess -d 100 -c {args.config_name} "
            f"-plans_name {args.plans_name}"
        )
        return 0

    configs[args.config_name] = new_config
    plans["configurations"] = configs

    backup = plans_path.with_suffix(".json.bak")
    shutil.copy2(plans_path, backup)
    logger.info("Backed up plans -> %s", backup)
    plans_path.write_text(json.dumps(plans, indent=2, sort_keys=False), encoding="utf-8")
    print(f"\nwrote        : {plans_path}")
    print(
        "\nNext — preprocess the new spacing (required; fullres data is not reused):\n"
        f"  export nnUNet_raw=/mnt/workspace/code/nnUNet_raw\n"
        f"  export nnUNet_preprocessed=/mnt/workspace/code/nnUNet_preprocessed\n"
        f"  export nnUNet_results=/mnt/workspace/code/nnUNet_results_topology\n"
        f"  nnUNetv2_preprocess -d 100 -c {args.config_name} "
        f"-plans_name {args.plans_name}\n"
        "\nThen author the split and train Track A from scratch:\n"
        "  python scripts/make_scroll_split.py --mode holdout-scroll --val-scroll 26010\n"
        "  bash scripts/nnunet_train_topology.sh --stage skelrecall --dry-run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
