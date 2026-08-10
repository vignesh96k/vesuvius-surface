"""Stage 2a — auxiliary affinity head.

The bet
-------
research_log.md §6 measured ``voi_split`` and ``voi_merge`` both around 1.0:
sheets are fragmented and neighbouring sheets are fused, in equal measure, and
the 1st-place team never solved the fused case. Those are the two error classes
VOI was invented to measure, and the connectomics literature attacks them by
predicting affinities between voxel pairs rather than a semantic mask. Funke et
al. (TPAMI 2018) report that long-range affinities improve segmentation *even
when used only as an auxiliary loss and thrown away at inference*.

Stage 2a tests exactly that claim on this dataset and nothing more. Inference is
untouched: the head is a regulariser on the decoder's full-resolution features.
Mutex-watershed agglomeration (Wolf et al., ECCV 2018) is Stage 2b and is
deliberately absent until 2a is measured to help.

Instance labels and augmentation
--------------------------------
Affinity targets need *instance* labels, but nnU-Net carries one integer
semantic array through spatial augmentation. If instance ids do not undergo the
same rotation, scaling and mirroring as the image, the affinities describe a
patch that no longer exists.

We derive instances by connected components **on the augmented patch**, inside
the augmentation pipeline, after every spatial transform has run
(:class:`training.transforms.AffinityTargetTransform`). Correspondence is then
true by construction rather than by careful plumbing — the failure mode above is
unrepresentable.

The alternative — precompute an instance volume and route it through
augmentation as an extra segmentation channel — was rejected. It needs a custom
preprocessor to get the instance volume into the ``.npz``, a custom dataloader
to read it, and it makes instance ids survive nearest-neighbour resampling and
sub-voxel grid sampling, where a rotated boundary can hand a voxel its
neighbour's id. All of that lives in the nnU-Net internals that change between
releases, and it buys one thing: instances that are correct with respect to the
whole *volume* rather than the *patch*.

That one thing is a real cost, and it is the honest weakness of this choice. Two
fragments of the same sheet that are only connected outside the 128^3 patch look
like separate instances inside it, so a long-range affinity across them is
labelled 0 when the volume-level truth is 1. ``scripts/audit_instance_locality.py``
measures how often that happens; run it before trusting the numbers. The
short-range offsets, which carry most of the signal, are almost immune — for
adjacent voxels the connecting path leaves the patch only right at its face.

Nothing here has been executed. See the bottom of research_log.md.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import autocast

from vesuvius_surface.training import compat
from vesuvius_surface.training.affinity import (
    DEFAULT_AFFINITY_OFFSETS,
    INSTANCE_CONNECTIVITY,
    NEAREST_NEIGHBOUR_OFFSETS,
    Offset,
)
from vesuvius_surface.training.losses import AffinityLoss, SegmentationTargetAdapter
from vesuvius_surface.training.network import (
    FullResolutionFeatureTap,
    attach_affinity_head,
    get_affinity_head,
)
from vesuvius_surface.training.transforms import (
    AffinityTargetTransform,
    append_transform,
    auxiliary_channels,
)

nnUNetTrainer = compat.nnUNetTrainer


class nnUNetTrainerAffinity(nnUNetTrainer):  # noqa: N801 - nnU-Net naming
    """``L_Dice + L_CE + w * L_Affinity``, affinities from a full-resolution head.

    Every knob is a class attribute so that a sweep is a set of short
    subclasses. That is not only convenience: nnU-Net puts the trainer name in
    the results folder and in the checkpoint, and ``affinity_offsets``
    determines the head's output channel count, so a checkpoint is only
    loadable by the class that produced it. Editing an attribute in place
    instead of subclassing would produce checkpoints that silently fail to load.
    """

    affinity_offsets: Tuple[Offset, ...] = DEFAULT_AFFINITY_OFFSETS
    weight_affinity: float = 1.0
    affinity_loss_mode: str = "bce_dice"
    affinity_batch_dice: bool = True
    instance_connectivity: int = INSTANCE_CONNECTIVITY
    mask_background_pairs: bool = True

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        # Explicit signature, matching nnUNetTrainer's own -- required for nnU-Net's own
        # self.my_init_kwargs bookkeeping (inspect.signature(self.__init__) indexed against
        # the caller frame's locals()). A generic (*args, **kwargs) signature breaks that
        # (KeyError: 'args'), confirmed by actually running this trainer -- not hypothetical.
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight_aff = float(self.weight_affinity)
        self.affinity_tap: FullResolutionFeatureTap | None = None
        self.affinity_head = None
        self.affinity_loss = AffinityLoss(
            mode=self.affinity_loss_mode, batch_dice=self.affinity_batch_dice
        )
        if self.label_manager.has_regions:
            raise NotImplementedError(
                "nnUNetTrainerAffinity does not support region-based training. "
                "Our dataset.json uses plain labels (background/surface/ignore)."
            )

    # -- network ------------------------------------------------------------

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager,
                                    num_input_channels, num_output_channels,
                                    enable_deep_supervision: bool = True):
        """Build the plans network, then bolt the affinity head onto its decoder.

        This runs at inference too — nnU-Net rebuilds the network through the
        trainer class recorded in the checkpoint. The head must therefore exist
        at prediction time or ``load_state_dict`` would reject the checkpoint's
        extra keys. It is never evaluated during prediction.

        Explicit signature (not *args/**kwargs): nnUNetTrainer.initialize() picks its
        call convention by introspecting inspect.signature(self.build_network_architecture)
        for a 'plans_manager' parameter name -- a generic passthrough signature fails that
        check and silently falls back to a deprecated old-style call convention with the
        wrong argument count, confirmed by actually running this trainer (TypeError: takes
        4 to 5 positional arguments but 6 were given).
        """
        network = super().build_network_architecture(
            plans_manager, configuration_manager, num_input_channels,
            num_output_channels, enable_deep_supervision,
        )
        attach_affinity_head(network, len(cls.affinity_offsets))
        return network

    def _do_i_compile(self) -> bool:
        """Disable ``torch.compile``.

        The affinity logits are read out through a forward pre-hook on the last
        segmentation layer. Dynamo's handling of module hooks is version
        dependent and a silently-not-fired hook here surfaces as a confusing
        "features were not captured" error rather than a compile failure.
        Compilation buys perhaps 10% of step time; correctness of an untested
        code path is worth more.
        """
        return False

    def initialize(self):
        super().initialize()
        self.affinity_tap = FullResolutionFeatureTap(self.network)
        self.affinity_head = get_affinity_head(self.network)
        self.print_to_log_file(
            f"Affinity head: {len(self.affinity_offsets)} offsets, "
            f"weight {self.weight_aff}, loss '{self.affinity_loss_mode}', "
            f"instances by {self.instance_connectivity}-connectivity on the "
            f"augmented patch."
        )
        self.print_to_log_file(f"Affinity offsets: {list(self.affinity_offsets)}")

    # -- loss ---------------------------------------------------------------

    def _build_loss(self):
        loss = compat.DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=compat.MemoryEfficientSoftDiceLoss,
        )

        # Drops the affinity channels before the segmentation loss sees them.
        loss = SegmentationTargetAdapter(loss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = compat.DeepSupervisionWrapper(loss, weights)
        return loss

    # -- data augmentation --------------------------------------------------

    @classmethod
    def get_training_transforms(cls, *args, **kwargs):
        transforms = super().get_training_transforms(*args, **kwargs)
        return append_transform(transforms, cls._affinity_transform())

    @classmethod
    def get_validation_transforms(cls, *args, **kwargs):
        transforms = super().get_validation_transforms(*args, **kwargs)
        return append_transform(transforms, cls._affinity_transform())

    @classmethod
    def _affinity_transform(cls) -> AffinityTargetTransform:
        # Appended last, i.e. after deep-supervision downsampling: the head is
        # full-resolution only, so downsampled copies of a 15-channel target
        # would be pure memory and CPU cost.
        return AffinityTargetTransform(
            offsets=cls.affinity_offsets,
            connectivity=cls.instance_connectivity,
            mask_background_pairs=cls.mask_background_pairs,
        )

    # -- steps --------------------------------------------------------------

    def _to_device(self, target):
        if isinstance(target, list):
            return [i.to(self.device, non_blocking=True) for i in target]
        return target.to(self.device, non_blocking=True)

    def train_step(self, batch: dict) -> dict:
        # Adapted from nnUNetTrainer.train_step; the additions are the tapped
        # forward and the affinity term.
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

            segmentation_loss = self.loss(output, target)
            affinity_loss = self.affinity_loss(
                affinity_logits,
                auxiliary_channels(target, len(self.affinity_offsets)),
            )
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
        # Adapted from nnUNetTrainer.validation_step. Two changes: the affinity
        # term is included so the validation loss is comparable with the
        # training loss, and the auxiliary channels are dropped before the
        # online "fake Dice", which treats the target as a label map.
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
            affinity_loss = self.affinity_loss(
                affinity_logits,
                auxiliary_channels(target, len(self.affinity_offsets)),
            )
            total = segmentation_loss + self.weight_aff * affinity_loss

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
        """Print the two loss components separately.

        Worth the four lines: the single most likely way this experiment fails
        quietly is ``weight_affinity`` being the wrong order of magnitude, so
        that the affinity term either dominates the segmentation loss or is
        numerically irrelevant. That is invisible in the combined number.
        """
        try:
            affinity = float(np.mean([float(o["loss_affinity"]) for o in outputs]))
            segmentation = float(
                np.mean([float(o["loss_segmentation"]) for o in outputs])
            )
        except (KeyError, TypeError, ValueError):
            return
        self.print_to_log_file(
            f"{phase} loss split: segmentation {segmentation:.4f}, "
            f"affinity {affinity:.4f} (x{self.weight_aff:g} = "
            f"{self.weight_aff * affinity:.4f})"
        )


class nnUNetTrainerAffinity_w01(nnUNetTrainerAffinity):  # noqa: N801
    weight_affinity = 0.1


class nnUNetTrainerAffinity_w05(nnUNetTrainerAffinity):  # noqa: N801
    weight_affinity = 0.5


class nnUNetTrainerAffinity_w2(nnUNetTrainerAffinity):  # noqa: N801
    weight_affinity = 2.0


class nnUNetTrainerAffinity_shortrange(nnUNetTrainerAffinity):  # noqa: N801
    """Ablation: nearest-neighbour offsets only.

    This is the control for the whole Stage 2a hypothesis. Funke et al.'s claim
    is specifically that *long-range* affinities are what carry the auxiliary
    benefit; if this variant matches the full offset set, the gain (if any) came
    from the extra head, not from the long-range structure, and Stage 2b's
    premise is weaker than assumed.
    """

    affinity_offsets = NEAREST_NEIGHBOUR_OFFSETS


class nnUNetTrainerAffinity_allpairs(nnUNetTrainerAffinity):  # noqa: N801
    """Ablation: do not mask background-to-background pairs.

    The default drops them because roughly 78% of in-bounds pairs are
    background-to-background and trivially zero. This variant is what Stage 2b
    would need, since mutex watershed reads affinities everywhere.
    """

    mask_background_pairs = False


def offsets_summary(offsets: Sequence[Offset] = DEFAULT_AFFINITY_OFFSETS) -> str:
    """One-line human-readable description of an offset list (for logs)."""
    return ", ".join("(" + ",".join(str(c) for c in o) + ")" for o in offsets)


__all__ = [
    "nnUNetTrainerAffinity",
    "nnUNetTrainerAffinity_allpairs",
    "nnUNetTrainerAffinity_shortrange",
    "nnUNetTrainerAffinity_w01",
    "nnUNetTrainerAffinity_w05",
    "nnUNetTrainerAffinity_w2",
    "offsets_summary",
]
