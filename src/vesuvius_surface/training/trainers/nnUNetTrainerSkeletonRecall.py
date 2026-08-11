"""Stage 1 — Skeleton Recall.

Port of ``nnUNetTrainerSkeletonRecall`` from
https://github.com/MIC-DKFZ/Skeleton-Recall (Kirchhoff et al., ECCV 2024),
adapted to run against a stock ``nnunetv2`` install instead of their fork.

What changed relative to upstream, and why
------------------------------------------
*Plumbing.* Upstream carries the skeleton in a new ``skel`` batch key, which
requires forking ``nnUNetDataLoader{2,3}DSkel`` and ``get_dataloaders``. We
append it as an extra segmentation channel instead, so only the transform list
and the loss need to know it exists — see :mod:`training.transforms`.

*Transform list.* Upstream copies nnU-Net's whole 150-line
``get_training_transforms`` to insert one transform. We call ``super()`` and
splice ours in before the deep-supervision downsampling, so nnU-Net's own
augmentation stays authoritative and we do not silently pin it to the release
this was written against.

*Ignore label.* Upstream warns that ignore-label support is experimental. The
concrete problem is that it binarises with ``seg > 0``, which for us folds the
58%-of-the-volume ignore class into the structure being skeletonised.
:func:`training.skeleton.tubed_skeleton` binarises on ``LABEL_SURFACE`` only.

Why this loss for this competition
----------------------------------
Measured directly on the 100-epoch baseline: SurfaceDice is at 0.985 and contributes 0.3448 of a possible
0.35, while VOI sits at 0.323 and TopoScore at 0.437. Essentially all remaining
headroom is topological, and a sheet riddled with pinholes still scores near
perfectly on position. Skeleton recall penalises exactly the breaks that
volumetric Dice is indifferent to.

Nothing here has been executed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import autocast

from vesuvius_surface.training import compat
from vesuvius_surface.training.losses import DC_SkelRec_and_CE_loss, SkeletonTargetAdapter
from vesuvius_surface.training.skeleton import DEFAULT_TUBE_DILATIONS
from vesuvius_surface.training.transforms import (
    TubedSkeletonTransform,
    insert_before_deep_supervision,
)

nnUNetTrainer = compat.nnUNetTrainer


class nnUNetTrainerSkeletonRecall(nnUNetTrainer):  # noqa: N801 - nnU-Net naming
    """``L_Dice + L_CE + w * L_SkelRecall``.

    ``weight_skeleton_recall`` is a class attribute so that a sweep is a set of
    three-line subclasses (see the bottom of this file) rather than an edit to
    the training script. nnU-Net records the trainer name in the results folder
    name, so each point of the sweep lands in its own directory and the
    checkpoint always knows which weight produced it.

    ``self.weight_srec`` is set from it in ``__init__`` and is what
    ``_build_loss`` reads, so it can also be poked from a debugger or a
    subclass' ``__init__`` without touching the class attribute.
    """

    weight_skeleton_recall: float = 1.0
    do_tube: bool = True
    tube_dilations: int = DEFAULT_TUBE_DILATIONS

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        # Explicit signature, matching nnUNetTrainer's own -- required for nnU-Net's own
        # self.my_init_kwargs bookkeeping, which introspects inspect.signature(self.__init__)
        # and indexes the *caller* frame's locals() by those parameter names. A generic
        # (*args, **kwargs) signature here breaks that (KeyError: 'args'), confirmed by
        # actually running this trainer -- not a hypothetical.
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight_srec = float(self.weight_skeleton_recall)
        if self.label_manager.has_regions:
            raise NotImplementedError(
                "nnUNetTrainerSkeletonRecall does not support region-based "
                "training. Our dataset.json uses plain labels "
                "(background/surface/ignore), so this should not trigger; if it "
                "does, the plans being used are not the ones this repository "
                "exports."
            )

    def _build_loss(self):
        loss = DC_SkelRec_and_CE_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            soft_skelrec_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            ce_kwargs={},
            weight_ce=1,
            weight_dice=1,
            weight_srec=self.weight_srec,
            ignore_label=self.label_manager.ignore_label,
            dice_class=compat.MemoryEfficientSoftDiceLoss,
        )

        # Unpacks [label, skeleton] from the target's channel axis. Sitting
        # inside the deep-supervision wrapper means it runs once per resolution
        # and train_step needs no override at all.
        loss = SkeletonTargetAdapter(loss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = compat.DeepSupervisionWrapper(loss, weights)
        return loss

    @classmethod
    def get_training_transforms(cls, *args, **kwargs):
        transforms = super().get_training_transforms(*args, **kwargs)
        return insert_before_deep_supervision(transforms, cls._skeleton_transform())

    @classmethod
    def get_validation_transforms(cls, *args, **kwargs):
        transforms = super().get_validation_transforms(*args, **kwargs)
        return insert_before_deep_supervision(transforms, cls._skeleton_transform())

    @classmethod
    def _skeleton_transform(cls) -> TubedSkeletonTransform:
        return TubedSkeletonTransform(
            do_tube=cls.do_tube, n_dilations=cls.tube_dilations
        )

    def validation_step(self, batch: dict) -> dict:
        # Adapted from nnUNetTrainer.validation_step. The only change is
        # dropping the auxiliary channel before the online "fake Dice" is
        # computed, since that code treats the target as a label map.
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else compat.dummy_context()
        ):
            output = self.network(data)
            del data
            loss_value = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

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
                target == self.label_manager.ignore_label,
                torch.zeros_like(target),
                target,
            )
        else:
            mask = None

        tp, fp, fn, _ = compat.get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        # [1:] drops background; we are not interested in background Dice.
        tp_hard = tp.detach().cpu().numpy()[1:]
        fp_hard = fp.detach().cpu().numpy()[1:]
        fn_hard = fn.detach().cpu().numpy()[1:]

        return {
            "loss": loss_value.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }


class nnUNetTrainerSkeletonRecall_w05(nnUNetTrainerSkeletonRecall):  # noqa: N801
    weight_skeleton_recall = 0.5


class nnUNetTrainerSkeletonRecall_w2(nnUNetTrainerSkeletonRecall):  # noqa: N801
    weight_skeleton_recall = 2.0


class nnUNetTrainerSkeletonRecall_w4(nnUNetTrainerSkeletonRecall):  # noqa: N801
    weight_skeleton_recall = 4.0


class nnUNetTrainerSkeletonRecall_notube(nnUNetTrainerSkeletonRecall):  # noqa: N801
    """Ablation: skeleton without the 2-voxel tube.

    Worth running once. Median sheet thickness here is 2 voxels
    (measured directly via 3D distance transform), so the tube is roughly as wide as the structure it is
    supposed to be a thin proxy for, and the tubed target may be close to the
    plain foreground mask.
    """

    do_tube = False


__all__ = [
    "nnUNetTrainerSkeletonRecall",
    "nnUNetTrainerSkeletonRecall_notube",
    "nnUNetTrainerSkeletonRecall_w05",
    "nnUNetTrainerSkeletonRecall_w2",
    "nnUNetTrainerSkeletonRecall_w4",
]
