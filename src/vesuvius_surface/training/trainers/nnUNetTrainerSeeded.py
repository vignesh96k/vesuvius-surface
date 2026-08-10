"""Seeded nnU-Net trainer variant, for reproducible baseline runs.

Stock nnU-Net sets no random seed anywhere: `nnUNetTrainer.__init__` has no seed
parameter, and `nnUNetv2_train` has no `--seed` flag (confirmed by reading
nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py and nnunetv2/run/run_training.py
directly, nnunetv2==2.8.1). nnU-Net's run_training.py also explicitly sets
`cudnn.benchmark = True` / `cudnn.deterministic = False` for speed -- that's left
untouched here. Forcing cudnn determinism would cost meaningful training speed for a
baseline run that only needs run-to-run comparability (same init, same augmentation
sampling order), not bit-exact reproduction, so this seeds Python/NumPy/PyTorch RNG
state at trainer construction and otherwise leaves nnU-Net's own defaults alone.

Loaded via nnU-Net's external-trainer mechanism (env var `nnUNet_extTrainer`, set in
training.environment.setup_environment) rather than by editing the installed nnunetv2
package -- see nnunetv2.utilities.find_objects.recursive_find_trainer_class_by_name.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _SeededMixin:
    """Seeds torch/numpy/random (env var NNUNET_SEED, default 42) before nnUNetTrainer.__init__
    runs. Shared by every nnUNetTrainerSeeded* variant so the seeding logic lives in one place."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        self.seed = int(os.environ.get("NNUNET_SEED", 42))
        _seed_everything(self.seed)
        print(f"[nnUNetTrainerSeeded] seeded torch/numpy/random with seed={self.seed}")
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainerSeeded(_SeededMixin, nnUNetTrainer):
    """Seeded variant of stock nnUNetTrainer. num_epochs is configurable via env var
    NNUNET_NUM_EPOCHS (default 1000, matching nnU-Net's own stock default) rather than fixed,
    specifically so a training run can be resumed (`--c`) targeting a longer schedule later
    (e.g. train to 1000, then later continue to 2000) without needing a new trainer class per
    target length -- the output folder is keyed on trainer *name* only (`nnUNetTrainerSeeded`,
    unchanged regardless of target), so `--c` correctly finds the existing checkpoint.

    Real caveat, not hidden: nnU-Net's LR schedule (PolyLRScheduler) decays smoothly to ~0 by
    whatever num_epochs the trainer is configured for at the time. Resuming a checkpoint whose
    LR has already decayed under one target (e.g. 1000) into a NEW, longer target (e.g. 2000)
    recomputes LR for the *new* schedule at the current epoch -- e.g. resuming a fully-decayed
    1000-epoch run to a 2000-epoch target jumps LR from ~0.00002 back up to ~0.0054, a ~269x
    jump, verified by direct calculation of PolyLRScheduler's formula. This is not equivalent
    to training fresh to the longer length -- it's closer to a "warm restart" (a real,
    sometimes-beneficial technique in the broader literature, e.g. Loshchilov & Hutter's
    SGDR), but not something nnU-Net's own recipe was specifically validated for. Printed as
    an explicit warning at trainer construction so it's never a silent assumption."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        target_epochs = int(os.environ.get("NNUNET_NUM_EPOCHS", 1000))
        if target_epochs != self.num_epochs:
            print(
                f"[nnUNetTrainerSeeded] num_epochs target changed: {self.num_epochs} -> {target_epochs}. "
                f"If resuming (--c) a checkpoint trained under a shorter/different target, this "
                f"recomputes the LR schedule for the new target at the current epoch -- expect a real "
                f"LR discontinuity (can be a large jump), not equivalent to a fresh run at this length."
            )
        self.num_epochs = target_epochs


class nnUNetTrainerSeeded_100epochs(_SeededMixin, nnUNetTrainer):
    """nnUNetTrainer_100epochs + a fixed RNG seed."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
