"""Combined clDice + RAdamScheduleFree trainer, 100-epoch ablation cell.

Replicates the specific loss+optimizer combination arunodhayan's real, evidenced-strong
checkpoint uses (nnUNetTrainer_RotFlip_ClDice_prob0.8_arun: DC+CE + 0.2*clDice,
RAdamScheduleFree lr=1e-3 wd=1e-4, no LR schedule) -- verified directly from their public
train.py and embedded trainer class, not guessed. Run on our own 657-case LOSO split (not
their fold='all'), so the comparison is leakage-free. Deliberately does NOT include their
boosted-rotation augmentation change -- isolating loss+optimizer as one combined factor,
augmentation held constant, to keep this a bounded, fast (100-epoch) signal check rather than
a full factorial ablation.

clDice ignore-label handling: MONAI's SoftclDiceLoss has no native ignore_label support (it
expects two same-shape multi-channel [background, foreground] probability tensors and performs
a global skeletonization, which can't be masked per-voxel without corrupting the skeleton
topology). Ignore-label voxels are forced to background in both prediction and target before
computing clDice -- the same "neutralize by forcing background" convention verified this
session in the official competition scorer's own ignore-label handling, not an ad-hoc choice.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast
from schedulefree import RAdamScheduleFree


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DC_CE_ClDice_loss(torch.nn.Module):
    """DC_and_CE_loss + cldice_weight * SoftclDiceLoss, ignore-label voxels forced to
    background for the clDice term (see module docstring)."""

    def __init__(self, dc_ce_kwargs: dict, ignore_label: int, cldice_weight: float = 0.2):
        super().__init__()
        from monai.losses import SoftclDiceLoss

        self.dc_ce = DC_and_CE_loss(
            dc_ce_kwargs, {}, weight_ce=1, weight_dice=1,
            ignore_label=ignore_label, dice_class=MemoryEfficientSoftDiceLoss,
        )
        self.cldice = SoftclDiceLoss(iter_=3, smooth=1.0)
        self.ignore_label = ignore_label
        self.cldice_weight = cldice_weight

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.dc_ce(net_output, target)

        probs = torch.softmax(net_output, dim=1)
        surface_pred = probs[:, 1:2]  # P(surface), keep channel dim
        surface_true = (target[:, 0:1] == 1).float()

        if self.ignore_label is not None:
            ignore_mask = (target[:, 0:1] == self.ignore_label)
            surface_pred = torch.where(ignore_mask, torch.zeros_like(surface_pred), surface_pred)
            surface_true = torch.where(ignore_mask, torch.zeros_like(surface_true), surface_true)

        y_pred = torch.cat([1 - surface_pred, surface_pred], dim=1)
        y_true = torch.cat([1 - surface_true, surface_true], dim=1)

        cldice_loss = self.cldice(y_true, y_pred)
        return base + self.cldice_weight * cldice_loss


class nnUNetTrainerSeeded_ClDice_ScheduleFree(nnUNetTrainer):
    """100-epoch ablation cell: DC+CE + 0.2*clDice loss, RAdamScheduleFree optimizer
    (lr=1e-3, wd=1e-4, no LR schedule), on our own LOSO split. Everything else (architecture,
    augmentation, seed, data) held identical to the 100-epoch baseline for direct comparison.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        self.seed = int(os.environ.get("NNUNET_SEED", 42))
        _seed_everything(self.seed)
        print(f"[nnUNetTrainerSeeded_ClDice_ScheduleFree] seeded with seed={self.seed}")
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100

    def _build_loss(self):
        loss = DC_CE_ClDice_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            ignore_label=self.label_manager.ignore_label,
            cldice_weight=0.2,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def configure_optimizers(self):
        optimizer = RAdamScheduleFree(self.network.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer, None

    def on_train_epoch_start(self):
        self.network.train()
        self.optimizer.train()  # ScheduleFree requires this mode toggle -- see module docstring
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)} "
            f"(RAdamScheduleFree, constant -- no schedule)")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def on_validation_epoch_start(self):
        self.network.eval()
        self.optimizer.eval()  # ScheduleFree requires this mode toggle for eval-mode weights
