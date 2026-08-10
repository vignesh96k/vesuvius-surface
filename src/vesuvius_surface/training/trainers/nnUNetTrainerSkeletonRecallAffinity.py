"""Combined Skeleton Recall (Stage 1) + affinity auxiliary head (Stage 2a).

Both mechanisms in this repo are additive auxiliary losses layered on top of the standard
DC+CE segmentation loss, and they attach to the target tensor at genuinely different points
in the augmentation pipeline, verified directly from training/transforms.py rather than
assumed:

- TubedSkeletonTransform runs BEFORE deep-supervision downsampling (insert_before_deep_supervision),
  so the skeleton channel (index 1) is present, correctly downsampled, at every DS scale.
- AffinityTargetTransform runs AFTER deep-supervision downsampling (append_transform, at the
  very end of the pipeline), and only touches the full-resolution entry (segmentation[0] once
  segmentation is already a per-scale list).

Composed in that order (skeleton insert, then affinity append), the full-resolution target
ends up as channels [label(0), skeleton(1), affinity_0(2), ..., affinity_{N-1}(N+1)]; every
downsampled DS scale stays [label(0), skeleton(1)] only (affinity is full-res-only by design,
per AffinityTargetTransform's own docstring -- a downsampled 15-channel affinity target would
be pure waste since the affinity head only ever reads full-resolution features).

This composition is safe for the *skeleton* side unmodified: SkeletonTargetAdapter reads only
target[:, :1] and target[:, 1:2] (verified from training/losses.py), so it ignores any trailing
affinity channels on scale 0 without needing changes.

It is NOT safe for the *affinity* side unmodified: the shared helper
`training.transforms.auxiliary_channels()` assumes channel 1+ is entirely affinity channels
(true for the standalone nnUNetTrainerAffinity, which has no skeleton channel occupying index
1). Reused here it would incorrectly include the skeleton channel as if it were affinity data --
caught by that helper's own shape check (it would raise, not silently corrupt), but still wrong.
This trainer does its own slicing instead: target[0][:, 2:] (skip label AND skeleton).

Nothing in either source trainer, nor this combination, has been executed before now.
"""

from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import torch
from torch import autocast

from vesuvius_surface.training import compat
from vesuvius_surface.training.affinity import DEFAULT_AFFINITY_OFFSETS, INSTANCE_CONNECTIVITY, Offset
from vesuvius_surface.training.losses import AffinityLoss, DC_SkelRec_and_CE_loss, SkeletonTargetAdapter
from vesuvius_surface.training.network import FullResolutionFeatureTap, attach_affinity_head, get_affinity_head
from vesuvius_surface.training.skeleton import DEFAULT_TUBE_DILATIONS
from vesuvius_surface.training.transforms import (
    AffinityTargetTransform,
    TubedSkeletonTransform,
    append_transform,
    insert_before_deep_supervision,
)

nnUNetTrainer = compat.nnUNetTrainer


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class nnUNetTrainerSkeletonRecallAffinity(nnUNetTrainer):  # noqa: N801
    """``L_Dice + L_CE + w_srec * L_SkelRecall`` (all resolutions) ``+ w_aff * L_Affinity``
    (full resolution only, via a bolted-on decoder head)."""

    weight_skeleton_recall: float = 1.0
    do_tube: bool = True
    tube_dilations: int = DEFAULT_TUBE_DILATIONS

    affinity_offsets: Tuple[Offset, ...] = DEFAULT_AFFINITY_OFFSETS
    weight_affinity: float = 1.0
    affinity_loss_mode: str = "bce_dice"
    affinity_batch_dice: bool = True
    instance_connectivity: int = INSTANCE_CONNECTIVITY
    mask_background_pairs: bool = True

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        # Explicit signature -- see nnUNetTrainerSkeletonRecall.py / nnUNetTrainerAffinity.py
        # for why (*args, **kwargs) breaks nnU-Net's own my_init_kwargs introspection.
        self.seed = int(os.environ.get("NNUNET_SEED", 42))
        _seed_everything(self.seed)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.weight_srec = float(self.weight_skeleton_recall)
        self.weight_aff = float(self.weight_affinity)
        self.affinity_tap: FullResolutionFeatureTap | None = None
        self.affinity_head = None
        self.affinity_loss = AffinityLoss(
            mode=self.affinity_loss_mode, batch_dice=self.affinity_batch_dice
        )
        print(
            f"[nnUNetTrainerSkeletonRecallAffinity] seeded with seed={self.seed}, "
            f"weight_srec={self.weight_srec}, weight_aff={self.weight_aff}, "
            f"{len(self.affinity_offsets)} affinity offsets"
        )
        if self.label_manager.has_regions:
            raise NotImplementedError(
                "region-based training not supported; dataset.json uses plain labels."
            )

    # -- network --------------------------------------------------------------

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager,
                                    num_input_channels, num_output_channels,
                                    enable_deep_supervision: bool = True):
        # Explicit signature required -- see nnUNetTrainerAffinity.py for why *args/**kwargs
        # breaks nnU-Net's own call-convention introspection in initialize().
        network = super().build_network_architecture(
            plans_manager, configuration_manager, num_input_channels,
            num_output_channels, enable_deep_supervision,
        )
        attach_affinity_head(network, len(cls.affinity_offsets))
        return network

    def _do_i_compile(self) -> bool:
        # Affinity logits are read via a forward pre-hook on the last segmentation layer --
        # see nnUNetTrainerAffinity for the same reasoning.
        return False

    def initialize(self):
        super().initialize()
        self.affinity_tap = FullResolutionFeatureTap(self.network)
        self.affinity_head = get_affinity_head(self.network)
        self.print_to_log_file(
            f"Affinity head: {len(self.affinity_offsets)} offsets, weight {self.weight_aff}, "
            f"loss '{self.affinity_loss_mode}'. Skeleton recall weight {self.weight_srec}, "
            f"tube={self.do_tube} (dilations={self.tube_dilations})."
        )

    # -- loss -------------------------------------------------------------------

    def _build_loss(self):
        loss = DC_SkelRec_and_CE_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp,
            },
            soft_skelrec_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp,
            },
            ce_kwargs={},
            weight_ce=1, weight_dice=1, weight_srec=self.weight_srec,
            ignore_label=self.label_manager.ignore_label,
            dice_class=compat.MemoryEfficientSoftDiceLoss,
        )
        loss = SkeletonTargetAdapter(loss)  # reads target[:, :1] and target[:, 1:2] only --
        # safe even with trailing affinity channels on scale 0, verified from source.

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = compat.DeepSupervisionWrapper(loss, weights)
        return loss

    def _affinity_target(self, target) -> torch.Tensor:
        """Channels [2:] of the full-resolution target -- NOT training.transforms.
        auxiliary_channels(), which assumes channel 1+ is entirely affinity (true only
        without a skeleton channel occupying index 1). See module docstring."""
        full_res = target[0] if isinstance(target, list) else target
        aff = full_res[:, 2:]
        expected = len(self.affinity_offsets)
        if aff.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} affinity channels at target[0][:, 2:], found "
                f"{aff.shape[1]}. Transform pipeline and trainer disagree on channel layout."
            )
        return aff

    # -- data augmentation --------------------------------------------------

    @classmethod
    def get_training_transforms(cls, *args, **kwargs):
        transforms = super().get_training_transforms(*args, **kwargs)
        transforms = insert_before_deep_supervision(transforms, cls._skeleton_transform())
        transforms = append_transform(transforms, cls._affinity_transform())
        return transforms

    @classmethod
    def get_validation_transforms(cls, *args, **kwargs):
        transforms = super().get_validation_transforms(*args, **kwargs)
        transforms = insert_before_deep_supervision(transforms, cls._skeleton_transform())
        transforms = append_transform(transforms, cls._affinity_transform())
        return transforms

    @classmethod
    def _skeleton_transform(cls) -> TubedSkeletonTransform:
        return TubedSkeletonTransform(do_tube=cls.do_tube, n_dilations=cls.tube_dilations)

    @classmethod
    def _affinity_transform(cls) -> AffinityTargetTransform:
        return AffinityTargetTransform(
            offsets=cls.affinity_offsets,
            connectivity=cls.instance_connectivity,
            mask_background_pairs=cls.mask_background_pairs,
        )

    # -- steps ----------------------------------------------------------------

    def _to_device(self, target):
        if isinstance(target, list):
            return [i.to(self.device, non_blocking=True) for i in target]
        return target.to(self.device, non_blocking=True)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device(batch["target"])

        self.optimizer.zero_grad(set_to_none=True)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else compat.dummy_context()
        ):
            with self.affinity_tap.capturing():
                output = self.network(data)
            del data
            affinity_logits = self.affinity_head(self.affinity_tap.take())

            segmentation_loss = self.loss(output, target)  # DC+CE+skelrec, DS-weighted
            affinity_loss = self.affinity_loss(affinity_logits, self._affinity_target(target))
            total = segmentation_loss + self.weight_aff * affinity_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            "loss": total.detach().cpu().numpy(),
            "loss_affinity": affinity_loss.detach().cpu().numpy(),
            "loss_segmentation": segmentation_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device(batch["target"])

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else compat.dummy_context()
        ):
            with self.affinity_tap.capturing():
                output = self.network(data)
            del data
            affinity_logits = self.affinity_head(self.affinity_tap.take())

            segmentation_loss = self.loss(output, target)
            affinity_loss = self.affinity_loss(affinity_logits, self._affinity_target(target))
            total = segmentation_loss + self.weight_aff * affinity_loss

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # Only the label channel for the online "fake Dice" -- target[:, :1] is correct
        # regardless of how many auxiliary channels (skeleton, affinity) follow it.
        target = target[:, :1]

        axes = [0] + list(range(2, output.ndim))

        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(
            output.shape, device=output.device, dtype=torch.float32
        )
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

        if self.label_manager.has_ignore_label:
            mask = (target != self.label_manager.ignore_label).float()
            target = torch.where(
                target == self.label_manager.ignore_label, torch.zeros_like(target), target
            )
        else:
            mask = None

        tp, fp, fn, _ = compat.get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        return {
            "loss": total.detach().cpu().numpy(),
            "loss_affinity": affinity_loss.detach().cpu().numpy(),
            "loss_segmentation": segmentation_loss.detach().cpu().numpy(),
            "tp_hard": tp.detach().cpu().numpy()[1:],
            "fp_hard": fp.detach().cpu().numpy()[1:],
            "fn_hard": fn.detach().cpu().numpy()[1:],
        }

    # -- logging ------------------------------------------------------------

    def on_train_epoch_end(self, train_outputs):
        self._log_affinity_component(train_outputs, "train")
        super().on_train_epoch_end(train_outputs)

    def on_validation_epoch_end(self, val_outputs):
        self._log_affinity_component(val_outputs, "val")
        super().on_validation_epoch_end(val_outputs)

    def _log_affinity_component(self, outputs, phase: str) -> None:
        try:
            affinity = float(np.mean([float(o["loss_affinity"]) for o in outputs]))
            segmentation = float(np.mean([float(o["loss_segmentation"]) for o in outputs]))
        except (KeyError, TypeError, ValueError):
            return
        self.print_to_log_file(
            f"{phase} loss split: segmentation {segmentation:.4f}, affinity {affinity:.4f} "
            f"(x{self.weight_aff:g} = {self.weight_aff * affinity:.4f})"
        )


__all__ = ["nnUNetTrainerSkeletonRecallAffinity"]
