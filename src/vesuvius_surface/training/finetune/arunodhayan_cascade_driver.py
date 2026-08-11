"""Config-driven CLI for fine-tuning arunodhayan's cascade checkpoint via nnU-Net's standard
-pretrained_weights transfer-learning mechanism.

This module is a generic, config-driven entrypoint written from how nnU-Net's own
-pretrained_weights transfer learning works, as a readable starting point for a future
fine-tune run driven by configs/finetune_cascade.yaml. It is a plain fine-tune (no loss/
architecture change), so it does NOT cover Phase 3, item 12's full arunodhayan fine-tune
(highpass input + skeleton-recall + affinity loss) -- that one reuses
`nnUNetTrainerSkeletonRecallAffinity` directly via nnU-Net's own `-tr` + `-pretrained_weights`
flags, no wrapper script needed. See README.md step 9 for the real command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml


def _run_command(cmd: str, name: str = "Command", timeout: Optional[int] = None) -> bool:
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"{name} TIMEOUT after {timeout}s!")
        return False
    if result.returncode != 0:
        print(f"{name} FAILED!\nSTDERR:\n{result.stderr[-3000:]}")
        return False
    print(f"{name} complete!")
    if result.stdout.strip():
        print("\n".join(result.stdout.strip().split("\n")[-30:]))
    return True


def run_finetune(
    dataset_id: int,
    configuration: str,
    fold: int,
    plans: str,
    trainer: str,
    pretrained_weights: Path,
    num_epochs: Optional[int] = None,
    timeout: Optional[int] = None,
) -> bool:
    """Fine-tune `pretrained_weights` on `dataset_id`/`configuration`/`fold` via nnU-Net's
    own -pretrained_weights flag (real transfer learning: nnU-Net loads matching-shape layers
    from the checkpoint and initializes the rest fresh, warns and skips on shape mismatch --
    this is why the real experiment 12 result used a purpose-built trainer subclass rather
    than this generic entrypoint whenever architecture-level changes, e.g. an added affinity
    head, were involved; this driver covers the plain-fine-tune case only)."""
    cmd = f"nnUNetv2_train {dataset_id:03d} {configuration} {fold} -p {plans} -tr {trainer}"
    cmd += f" -pretrained_weights {pretrained_weights}"
    return _run_command(cmd, f"Fine-tune ({trainer})", timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/finetune_cascade.yaml"))
    parser.add_argument("--trainer", required=True, help="nnU-Net trainer class name to fine-tune with")
    parser.add_argument("--dry-run", action="store_true", help="print the command without running it")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    training = cfg["training"]
    pretrained = cfg["pretrained"]

    cascade_ckpt = Path(pretrained["cascade_checkpoint"])
    if not cascade_ckpt.exists():
        print(f"ERROR: {cascade_ckpt} not found. Download arunodhayan/cascade-updated from Kaggle Models first.")
        return 1

    cmd_preview = (
        f"nnUNetv2_train {cfg['dataset']['dataset_id']:03d} {training['configuration']} "
        f"{training['fold']} -p {training['plans_name']} -tr {args.trainer} "
        f"-pretrained_weights {cascade_ckpt}"
    )
    if args.dry_run:
        print(cmd_preview)
        return 0

    ok = run_finetune(
        dataset_id=cfg["dataset"]["dataset_id"],
        configuration=training["configuration"],
        fold=training["fold"],
        plans=training["plans_name"],
        trainer=args.trainer,
        pretrained_weights=cascade_ckpt,
        num_epochs=training.get("epochs"),
        timeout=training.get("command_timeout"),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
