#!/usr/bin/env python3
"""Fetch the public nnU-Net checkpoint from the 1st-place solution.

Downloads ``scrollprize/surface_m7_nnunet`` (the "m7" component of the winning
ensemble) and arranges it as an ``nnUNet_results`` tree so ``nnUNetv2_predict``
can use it directly.

Attribution: the weights are third-party work released under Apache-2.0. They
are used here as a reference baseline, not as our own trained model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ID = "scrollprize/surface_m7_nnunet"
CHECKPOINT_GLOBS = ("checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_*.pth")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/mnt/workspace/code/pretrained"),
        help="Where to download the snapshot.",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("/mnt/workspace/code/nnUNet_results"),
        help="nnUNet_results parent to install into.",
    )
    p.add_argument("--dataset-id", type=int, default=100)
    p.add_argument("--dataset-name", default="VesuviusSurface")
    p.add_argument(
        "--trainer",
        default="nnUNetTrainer",
        help="Used only when the snapshot has no trainer__plans__config folder.",
    )
    p.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    p.add_argument("--config", default="3d_fullres")
    p.add_argument(
        "--install",
        action="store_true",
        help="Create the nnUNet_results layout (default is inspect-only).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def download_snapshot(repo_id: str, cache_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is required.\n  pip install huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", repo_id, cache_dir)
    path = snapshot_download(repo_id=repo_id, local_dir=str(cache_dir / repo_id.split("/")[-1]))
    return Path(path)


def _is_internal(path: Path, root: Path) -> bool:
    """True for hugging_face bookkeeping paths such as ``.cache/huggingface/``."""
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def _walk(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.rglob(pattern) if not _is_internal(p, root))


def find_model_dirs(root: Path) -> list[Path]:
    """Directories named ``trainer__plans__configuration``."""
    return [p for p in _walk(root, "*") if p.is_dir() and p.name.count("__") == 2]


def find_files(root: Path, name: str) -> list[Path]:
    return [p for p in _walk(root, name) if p.is_file()]


def find_checkpoints(root: Path) -> list[Path]:
    found: set[Path] = set()
    for pattern in CHECKPOINT_GLOBS:
        found.update(p for p in _walk(root, pattern) if p.is_file())
    return sorted(found)


def find_fold_dirs(root: Path) -> list[Path]:
    return [p for p in _walk(root, "fold_*") if p.is_dir()]


def describe_snapshot(snapshot: Path) -> dict[str, list[Path]]:
    return {
        "model_dirs": find_model_dirs(snapshot),
        "plans": find_files(snapshot, "plans.json"),
        "dataset_json": find_files(snapshot, "dataset.json"),
        "splits": find_files(snapshot, "splits_final.json"),
        "validation_summaries": find_files(snapshot, "summary.json"),
        "checkpoints": find_checkpoints(snapshot),
        "fold_dirs": find_fold_dirs(snapshot),
    }


def read_plans(snapshot: Path) -> dict:
    """Plans identifier and available configurations, when plans.json exists."""
    matches = find_files(snapshot, "plans.json")
    if not matches:
        return {}
    try:
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse %s: %s", matches[0], exc)
        return {}
    return {
        "plans_name": payload.get("plans_name"),
        "configurations": sorted(payload.get("configurations", {}).keys()),
        "trainer_hint": payload.get("trainer_name"),
    }


def read_dataset_labels(snapshot: Path) -> dict:
    matches = find_files(snapshot, "dataset.json")
    if not matches:
        return {}
    try:
        payload = json.loads(matches[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse %s: %s", matches[0], exc)
        return {}
    return {
        "labels": payload.get("labels", {}),
        "file_ending": payload.get("file_ending"),
        "numTraining": payload.get("numTraining"),
    }


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            logger.warning("Refusing to replace existing directory: %s", dst)
            return
    os.symlink(src.resolve(), dst)
    logger.info("linked %s -> %s", dst, src)


def install_results_tree(
    snapshot: Path,
    results_root: Path,
    dataset_id: int,
    dataset_name: str,
    trainer: str,
    plans: str,
    configuration: str,
) -> Path:
    """Expose the snapshot through an nnUNet_results-compatible path."""
    dataset_folder = results_root / f"Dataset{dataset_id:03d}_{dataset_name}"
    model_dirs = find_model_dirs(snapshot)

    if model_dirs:
        source = model_dirs[0]
        target = dataset_folder / source.name
        if len(model_dirs) > 1:
            logger.warning("Multiple model dirs found, using %s", source.name)
        _link(source, target)
        return target

    target = dataset_folder / f"{trainer}__{plans}__{configuration}"
    target.mkdir(parents=True, exist_ok=True)

    for name in ("plans.json", "dataset.json"):
        matches = find_files(snapshot, name)
        if matches:
            _link(matches[0], target / name)
        else:
            logger.warning("%s not found in snapshot", name)

    for fold_dir in find_fold_dirs(snapshot):
        _link(fold_dir, target / fold_dir.name)

    return target


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    snapshot = download_snapshot(args.repo_id, args.cache_dir)
    print(f"snapshot    : {snapshot}")

    info = describe_snapshot(snapshot)
    for key, paths in info.items():
        print(f"{key:<21}: {len(paths)}")
        for path in paths[:6]:
            print(f"  {path.relative_to(snapshot)}")
        if len(paths) > 6:
            print(f"  ... {len(paths) - 6} more")

    if info["splits"]:
        print(
            "\nA splits_final.json ships with this checkpoint. Pass it to\n"
            "scripts/nnunet_folds.py --splits-json to replace our reconstruction\n"
            "with the authoritative fold membership."
        )
    else:
        print(
            "\nNo splits_final.json in the snapshot. Fold membership must be\n"
            "reconstructed (scripts/nnunet_folds.py); treat it as an assumption\n"
            "and cross-check against any fold_*/validation listing above."
        )

    if info["validation_summaries"]:
        summary = info["validation_summaries"][0]
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            per_case = payload.get("metric_per_case", [])
            print(f"\nvalidation summary lists {len(per_case)} cases: {summary.name}")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not parse %s: %s", summary, exc)

    plans_info = read_plans(snapshot)
    dataset_info = read_dataset_labels(snapshot)
    if plans_info:
        print(f"\nplans_name  : {plans_info.get('plans_name')}")
        print(f"configs     : {', '.join(plans_info.get('configurations', [])) or 'unknown'}")
    if dataset_info:
        print(f"labels      : {dataset_info.get('labels')}")
        print(f"file_ending : {dataset_info.get('file_ending')}")
        print(f"numTraining : {dataset_info.get('numTraining')}")

    plans_name = plans_info.get("plans_name") or args.plans
    configuration = args.config
    available = plans_info.get("configurations") or []
    if available and configuration not in available:
        logger.warning(
            "Config %r not in plans (%s); using %s",
            configuration,
            ", ".join(available),
            available[0],
        )
        configuration = available[0]

    if args.install:
        target = install_results_tree(
            snapshot,
            args.results_root,
            args.dataset_id,
            args.dataset_name,
            args.trainer,
            plans_name,
            configuration,
        )
        print(f"\ninstalled   : {target}")
        print(
            "\nNext — reference prediction:\n"
            "  bash scripts/nnunet_predict.sh --input <dir-of-tifs> --output <out-dir> \\\n"
            f"      --trainer {args.trainer} --plans {plans_name} --config {configuration}"
        )
    else:
        print(
            f"\n(inspect-only; re-run with --install to build "
            f"{args.trainer}__{plans_name}__{configuration})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
