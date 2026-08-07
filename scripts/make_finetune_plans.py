#!/usr/bin/env python3
"""Add a larger-patch configuration for fine-tuning the published checkpoint.

The 1st-place solution was built by fine-tuning a 128-patch base model at
progressively larger patch sizes (192, then 256). Larger patches give the
network more spatial context, which is what reduces broken and fused sheets —
the VOI and TopoScore terms that hold all of our headroom.

Patch size does not change the *architecture*: stage count, feature widths,
kernels and strides are what the weights map onto. So we derive a new
configuration that inherits everything from `3d_fullres` and overrides only
the patch size, which keeps `-pretrained_weights` loadable.

    python scripts/make_finetune_plans.py --patch-size 192

Preprocessed data is keyed by target spacing, not patch size, so the new
configuration reuses whatever `3d_fullres` already produced.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-plans",
        type=Path,
        default=Path("/mnt/workspace/code/pretrained/surface_m7_nnunet/plans.json"),
        help="Plans that the pretrained checkpoint was trained with.",
    )
    p.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("/mnt/workspace/code/nnUNet_preprocessed/Dataset100_VesuviusSurface"),
    )
    p.add_argument("--base-config", default="3d_fullres")
    p.add_argument("--patch-size", type=int, nargs="+", default=[192])
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to the base config's batch size. Lower it if VRAM is tight.",
    )
    p.add_argument("--config-name", default=None, help="Defaults to <base>_<patch>.")
    p.add_argument(
        "--data-identifier",
        default=None,
        help="Override to reuse preprocessed data from a different plans run.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def normalize_patch(values: list[int]) -> list[int]:
    if len(values) == 1:
        return [values[0]] * 3
    if len(values) == 3:
        return list(values)
    raise ValueError("--patch-size takes 1 or 3 integers")


def stride_product(arch_kwargs: dict[str, Any]) -> list[int]:
    """Total downsampling factor per axis, which patch size must divide."""
    strides = arch_kwargs.get("strides") or []
    if not strides:
        return [1, 1, 1]
    factors = [1, 1, 1]
    for stage in strides:
        stage_list = [stage] * 3 if isinstance(stage, int) else list(stage)
        for axis in range(3):
            factors[axis] *= int(stage_list[axis])
    return factors


def activation_ratio(new_patch: list[int], base_patch: list[int], new_bs: int, base_bs: int) -> float:
    base_voxels = math.prod(base_patch) * max(base_bs, 1)
    new_voxels = math.prod(new_patch) * max(new_bs, 1)
    return new_voxels / base_voxels if base_voxels else float("nan")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if not args.source_plans.is_file():
        print(f"ERROR: plans not found: {args.source_plans}", file=sys.stderr)
        return 1
    if not args.preprocessed_dir.is_dir():
        print(f"ERROR: preprocessed dir not found: {args.preprocessed_dir}", file=sys.stderr)
        print("Run scripts/nnunet_setup_and_preprocess.sh first.", file=sys.stderr)
        return 1

    plans = json.loads(args.source_plans.read_text(encoding="utf-8"))
    plans_name = plans.get("plans_name") or args.source_plans.stem
    configs = plans.get("configurations", {})

    base = configs.get(args.base_config)
    if base is None:
        print(
            f"ERROR: config {args.base_config!r} not in plans. Available: "
            + ", ".join(sorted(configs)),
            file=sys.stderr,
        )
        return 1

    try:
        patch = normalize_patch(args.patch_size)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    base_patch = [int(v) for v in base.get("patch_size", [])]
    base_bs = int(base.get("batch_size", 2))
    batch_size = args.batch_size if args.batch_size is not None else base_bs

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
        print(f"  factors: {factors}  (nearest valid multiples of {factors})")
        return 1

    config_name = args.config_name or f"{args.base_config}_{patch[0]}"
    new_config: dict[str, Any] = {
        "inherits_from": args.base_config,
        "patch_size": patch,
        "batch_size": batch_size,
    }
    if args.data_identifier:
        new_config["data_identifier"] = args.data_identifier

    configs[config_name] = new_config
    plans["configurations"] = configs

    target = args.preprocessed_dir / f"{plans_name}.json"
    if target.exists():
        backup = target.with_suffix(".json.bak")
        shutil.copy2(target, backup)
        logger.info("Backed up existing plans -> %s", backup)
    target.write_text(json.dumps(plans, indent=2, sort_keys=False), encoding="utf-8")

    ratio = activation_ratio(patch, base_patch, batch_size, base_bs)
    inherited_id = base.get("data_identifier", "?")

    print(f"plans        : {plans_name}")
    print(f"written to   : {target}")
    print(f"new config   : {config_name}")
    print(f"patch size   : {base_patch} -> {patch}")
    print(f"batch size   : {base_bs} -> {batch_size}")
    print(f"downsampling : {factors} (patch divides cleanly)")
    print(f"data id      : {args.data_identifier or inherited_id} (inherited; no re-preprocess)")
    print(f"activation   : ~{ratio:.2f}x the base config's per-step volume")

    if ratio > 4:
        print(
            "\nWARNING: that is a large jump in activation memory. If training "
            "OOMs, halve --batch-size and re-run."
        )

    print(
        "\nNext — fine-tune from the published checkpoint:\n"
        f"  bash scripts/nnunet_finetune.sh --config {config_name} --plans {plans_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
