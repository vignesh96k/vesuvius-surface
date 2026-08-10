"""Auxiliary losses: skeleton recall (Stage 1) and affinities (Stage 2a).

``SoftSkeletonRecallLoss`` and ``DC_SkelRec_and_CE_loss`` are ports of
``nnunetv2/training/loss/dice.py`` and ``compound_losses.py`` from
https://github.com/MIC-DKFZ/Skeleton-Recall (Kirchhoff et al., ECCV 2024). They
are reproduced here rather than imported so that this repository does not
require a forked nnU-Net checkout on the training box.

``AffinityLoss`` is new. It is a masked binary loss over the affinity target
stack built by :mod:`training.affinity`.

The two ``*TargetAdapter`` modules exist because our auxiliary targets ride
through nnU-Net's augmentation pipeline as extra channels of the segmentation
tensor (see :mod:`training.transforms` for why). They unpack those channels
immediately before the real loss sees them, which keeps ``train_step`` free of
target surgery and lets nnU-Net's ``DeepSupervisionWrapper`` stay in charge of
the per-resolution weighting.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from vesuvius_surface.training import compat


class SoftSkeletonRecallLoss(nn.Module):
    """Soft recall of the predicted foreground over the tubed skeleton.

    Port of ``SoftSkeletonRecallLoss`` from MIC-DKFZ/Skeleton-Recall. Recall
    rather than Dice is the point: a skeleton has vanishing volume compared to
    the structure, so a Dice term over it would be dominated by the denominator
    and the gradient would mostly push the prediction *down*. Recall only ever
    rewards covering the skeleton, which is what preserves connectivity.
    """

    def __init__(
        self,
        apply_nonlin: Optional[Callable] = None,
        batch_dice: bool = False,
        do_bg: bool = False,
        smooth: float = 1.0,
        ddp: bool = True,
    ) -> None:
        super().__init__()
        if do_bg:
            raise RuntimeError(
                "SoftSkeletonRecallLoss is undefined for the background class: "
                "recall over a background skeleton has no meaning. Pass do_bg=False."
            )
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, loss_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Drop the background channel; the skeleton target is foreground-only.
        x = x[:, 1:]

        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all(i == j for i, j in zip(shp_x, shp_y)):
                y_onehot = y[:, 1:]
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=x.dtype)
                y_onehot.scatter_(1, gt, 1)
                y_onehot = y_onehot[:, 1:]

            sum_gt = (
                y_onehot.sum(axes)
                if loss_mask is None
                else (y_onehot * loss_mask).sum(axes)
            )

        inter_rec = (
            (x * y_onehot).sum(axes)
            if loss_mask is None
            else (x * y_onehot * loss_mask).sum(axes)
        )

        if self.ddp and self.batch_dice:
            all_gather = compat.AllGatherGrad
            inter_rec = all_gather.apply(inter_rec).sum(0)
            sum_gt = all_gather.apply(sum_gt).sum(0)

        if self.batch_dice:
            inter_rec = inter_rec.sum(0)
            sum_gt = sum_gt.sum(0)

        recall = (inter_rec + self.smooth) / (torch.clip(sum_gt + self.smooth, 1e-8))
        return -recall.mean()


class DC_SkelRec_and_CE_loss(nn.Module):  # noqa: N801 - mirrors upstream naming
    """``L_Dice + L_CE + w * L_SkelRecall``.

    Port of ``DC_SkelREC_and_CE_loss`` from MIC-DKFZ/Skeleton-Recall, with the
    ignore-label handling left intact: ``mask`` suppresses the Dice and skeleton
    terms on ignore voxels, and ``ignore_index`` does the same for cross
    entropy. That matters much more for us than it does upstream — ignore is
    58% of a typical volume here.
    """

    def __init__(
        self,
        soft_dice_kwargs: dict,
        soft_skelrec_kwargs: dict,
        ce_kwargs: dict,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        weight_srec: float = 1.0,
        ignore_label: Optional[int] = None,
        dice_class: Optional[type] = None,
    ) -> None:
        super().__init__()
        ce_kwargs = dict(ce_kwargs)
        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_srec = weight_srec
        self.ignore_label = ignore_label

        dice_class = dice_class if dice_class is not None else compat.MemoryEfficientSoftDiceLoss
        softmax_helper_dim1 = compat.softmax_helper_dim1

        self.ce = compat.RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.srec = SoftSkeletonRecallLoss(
            apply_nonlin=softmax_helper_dim1, **soft_skelrec_kwargs
        )

    def forward(
        self, net_output: torch.Tensor, target: torch.Tensor, skel: torch.Tensor
    ) -> torch.Tensor:
        if self.ignore_label is not None:
            if target.shape[1] != 1:
                raise ValueError(
                    "DC_SkelRec_and_CE_loss with an ignore label needs a single-channel "
                    f"label map, got {target.shape[1]} channels. If the auxiliary "
                    "channels are still attached, SkeletonTargetAdapter did not run."
                )
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            target_skel = torch.where(mask, skel, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            target_skel = skel
            mask = None
            num_fg = None

        dc_loss = (
            self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        )
        srec_loss = (
            self.srec(net_output, target_skel, loss_mask=mask) if self.weight_srec != 0 else 0
        )
        ce_loss = (
            self.ce(net_output, target[:, 0])
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0)
            else 0
        )

        return (
            self.weight_ce * ce_loss
            + self.weight_dice * dc_loss
            + self.weight_srec * srec_loss
        )


class AffinityLoss(nn.Module):
    """Masked binary loss over an affinity target stack.

    ``target`` carries its own validity mask: entries below zero
    (:data:`training.affinity.AFFINITY_INVALID`) are dropped. That covers pairs
    reaching outside the patch, pairs touching ``LABEL_IGNORE``, and — when the
    target builder was asked to — background-to-background pairs.

    ``mode`` is one of ``"bce"``, ``"dice"`` or ``"bce_dice"`` (the mean of the
    two, mirroring nnU-Net's own Dice+CE compound). Soft Dice earns its place
    because affinity positives are rare and get rarer with offset length, and
    Dice is invariant to that imbalance in a way BCE is not.
    """

    def __init__(
        self,
        mode: str = "bce_dice",
        smooth: float = 1e-5,
        batch_dice: bool = True,
        channel_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if mode not in ("bce", "dice", "bce_dice"):
            raise ValueError(f"mode must be 'bce', 'dice' or 'bce_dice', got {mode!r}")
        self.mode = mode
        self.smooth = smooth
        self.batch_dice = batch_dice
        if channel_weights is None:
            self.register_buffer("channel_weights", None)
        else:
            self.register_buffer(
                "channel_weights", torch.as_tensor(list(channel_weights), dtype=torch.float32)
            )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"affinity logits {tuple(logits.shape)} and target "
                f"{tuple(target.shape)} disagree. The head's output channel count "
                "must equal len(trainer.affinity_offsets); if you changed the "
                "offsets you must retrain, not resume."
            )

        # Force fp32. Under autocast the head's output is fp16, and a sum over
        # 15 * 128^3 = 31M elements overflows fp16's 65504 ceiling long before
        # it produces a usable number. nnU-Net's own Dice term escapes this only
        # because autocast promotes softmax to fp32; sigmoid gets no such
        # treatment.
        logits = logits.float()

        valid = target >= 0
        mask = valid.to(logits.dtype)
        positives = target.clamp_min(0).to(logits.dtype)
        n_valid = mask.sum()

        if n_valid == 0:
            # Degenerate patch (everything ignored). Return a zero that still
            # carries a graph so DDP and the grad scaler see a consistent set of
            # participating parameters.
            return logits.sum() * 0.0

        loss = logits.new_zeros(())

        if self.mode in ("bce", "bce_dice"):
            elementwise = F.binary_cross_entropy_with_logits(
                logits, positives, reduction="none"
            )
            loss = loss + (elementwise * mask).sum() / n_valid

        if self.mode in ("dice", "bce_dice"):
            axes = list(range(2, logits.ndim))
            if self.batch_dice:
                axes = [0] + axes
            probabilities = torch.sigmoid(logits) * mask
            masked_target = positives * mask
            intersection = (probabilities * masked_target).sum(axes)
            denominator = probabilities.sum(axes) + masked_target.sum(axes)
            dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
            per_channel = 1.0 - dice
            if self.channel_weights is not None:
                weights = self.channel_weights.to(per_channel.device)
                per_channel = per_channel * weights / weights.mean()
            loss = loss + per_channel.mean()

        if self.mode == "bce_dice":
            loss = loss * 0.5

        return loss


class SegmentationTargetAdapter(nn.Module):
    """Strip auxiliary channels, then delegate to a plain ``(output, target)`` loss."""

    def __init__(self, loss: nn.Module) -> None:
        super().__init__()
        self.loss = loss

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(net_output, target[:, :1])


class SkeletonTargetAdapter(nn.Module):
    """Split ``[label, skeleton]`` channels and delegate to the compound loss."""

    def __init__(self, loss: nn.Module) -> None:
        super().__init__()
        self.loss = loss

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[1] < 2:
            raise ValueError(
                "Expected the segmentation target to carry a skeleton channel "
                f"appended by TubedSkeletonTransform, but it has only "
                f"{target.shape[1]} channel(s). This means get_training_transforms "
                "did not install the transform — check that the trainer actually "
                "in use is nnUNetTrainerSkeletonRecall (nnU-Net records the "
                "trainer name in the results folder name)."
            )
        return self.loss(net_output, target[:, :1], target[:, 1:2])


__all__ = [
    "AffinityLoss",
    "DC_SkelRec_and_CE_loss",
    "SegmentationTargetAdapter",
    "SkeletonTargetAdapter",
    "SoftSkeletonRecallLoss",
]
