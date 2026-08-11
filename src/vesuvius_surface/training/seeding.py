"""RNG seeding, deliberately dependency-free (no nnunetv2, no torch training-framework
imports) so it can be tested and reused without pulling in the full nnU-Net trainer stack.

Stock nnU-Net sets no random seed anywhere: `nnUNetTrainer.__init__` has no seed parameter,
and `nnUNetv2_train` has no `--seed` flag (confirmed by reading
nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py and nnunetv2/run/run_training.py directly,
nnunetv2==2.8.1). This is the only RNG-seeding fix anywhere in the project -- every
from-scratch baseline result in this project traces back to this function.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
