"""batchgeneratorsv2 transforms that attach auxiliary targets to a patch.

How auxiliary targets reach the loss
------------------------------------
Upstream Skeleton-Recall adds a new ``skel`` key to the batch dict, which means
forking ``nnUNetDataLoader3D.generate_train_batch`` (to collect the key) and
``nnUNetTrainer.get_dataloaders`` (to instantiate the fork) — roughly 140 lines
copied out of nnU-Net, touching ``nnUNetDataset.load_case``, which is precisely
the API that changed in nnU-Net 2.5.

We append the auxiliary targets as extra *channels of the segmentation tensor*
instead. nnU-Net already carries segmentation as ``(C, X, Y, Z)`` and already
stacks it, spatially transforms it and downsamples it channel-wise, so nothing
in the dataloader needs to know these channels exist. The only code that must
know is the loss, and the adapters in :mod:`training.losses` unpack them there.
That trades one line of channel arithmetic for not forking two files that move
between releases.

Both transforms belong at the *end* of the augmentation pipeline. Everything
before them assumes ``segmentation`` has a single label channel (``RemoveLabel``,
``MaskImage`` and the cascade transforms all index channel 0 explicitly), so
appending earlier would be wrong.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import torch

from data.schema import LABEL_IGNORE, LABEL_SURFACE
from training import compat
from training.affinity import (
    DEFAULT_AFFINITY_OFFSETS,
    INSTANCE_CONNECTIVITY,
    Offset,
    affinity_targets,
    validate_offsets,
)
from training.skeleton import DEFAULT_TUBE_DILATIONS, tubed_skeleton

BasicTransform = compat.BasicTransform

#: Located by name so that we do not have to import it, which keeps this module
#: working if nnU-Net moves the class between packages.
DS_TRANSFORM_CLASS_NAME = "DownsampleSegForDSTransform"


class TubedSkeletonTransform(BasicTransform):
    """Append a ``{0, 1}`` tubed-skeleton channel to the segmentation.

    Placed *before* deep-supervision downsampling so the skeleton is downsampled
    alongside the label map and every supervision head gets a matching target,
    which is what upstream does.
    """

    def __init__(
        self,
        do_tube: bool = True,
        n_dilations: int = DEFAULT_TUBE_DILATIONS,
        surface_label: int = LABEL_SURFACE,
    ) -> None:
        super().__init__()
        self.do_tube = do_tube
        self.n_dilations = n_dilations
        self.surface_label = surface_label

    def apply(self, data_dict: dict, **params) -> dict:
        segmentation = data_dict.get("segmentation")
        if segmentation is None:
            raise RuntimeError(
                "TubedSkeletonTransform ran on a batch without a 'segmentation' "
                "key. It must not be used in an inference pipeline."
            )
        data_dict["segmentation"] = self._append(segmentation)
        return data_dict

    def _append(self, segmentation: torch.Tensor) -> torch.Tensor:
        labels = segmentation[0].numpy()
        skeleton = tubed_skeleton(
            labels,
            surface_label=self.surface_label,
            do_tube=self.do_tube,
            n_dilations=self.n_dilations,
        )
        channel = torch.from_numpy(skeleton)[None].to(segmentation.dtype)
        return torch.cat((segmentation, channel), dim=0)


class AffinityTargetTransform(BasicTransform):
    """Append one ``{-1, 0, 1}`` affinity channel per offset.

    Instance labels are derived by connected components **on the augmented
    patch**, at this point in the pipeline. See the class-level note below for
    why that is preferred to routing a precomputed instance volume through
    augmentation.

    Placed *after* deep-supervision downsampling: the affinity head is
    full-resolution only, so downsampled copies of a 15-channel target would be
    pure waste. When deep supervision is on, ``segmentation`` is a list and only
    element 0 (the full-resolution entry) is extended.

    Cost is one ``scipy.ndimage.label`` plus ``len(offsets)`` strided
    comparisons over the patch, on the order of 100 ms for a 128^3 patch. That
    runs inside the data-augmentation worker processes, of which nnU-Net starts
    a dozen, so it should stay hidden behind the GPU step. It has not been
    timed — nothing in this repository has been executed.
    """

    def __init__(
        self,
        offsets: Sequence[Offset] = DEFAULT_AFFINITY_OFFSETS,
        surface_label: int = LABEL_SURFACE,
        ignore_label: int = LABEL_IGNORE,
        connectivity: int = INSTANCE_CONNECTIVITY,
        mask_background_pairs: bool = True,
    ) -> None:
        super().__init__()
        self.offsets = validate_offsets(offsets)
        self.surface_label = surface_label
        self.ignore_label = ignore_label
        self.connectivity = connectivity
        self.mask_background_pairs = mask_background_pairs

    @property
    def num_channels(self) -> int:
        return len(self.offsets)

    def apply(self, data_dict: dict, **params) -> dict:
        segmentation = data_dict.get("segmentation")
        if segmentation is None:
            raise RuntimeError(
                "AffinityTargetTransform ran on a batch without a 'segmentation' "
                "key. It must not be used in an inference pipeline."
            )
        if isinstance(segmentation, list):
            segmentation = list(segmentation)
            segmentation[0] = self._append(segmentation[0])
            data_dict["segmentation"] = segmentation
        else:
            data_dict["segmentation"] = self._append(segmentation)
        return data_dict

    def _append(self, segmentation: torch.Tensor) -> torch.Tensor:
        labels = segmentation[0].numpy()
        targets = affinity_targets(
            labels,
            self.offsets,
            surface_label=self.surface_label,
            ignore_label=self.ignore_label,
            connectivity=self.connectivity,
            mask_background_pairs=self.mask_background_pairs,
        )
        channels = torch.from_numpy(np.ascontiguousarray(targets)).to(segmentation.dtype)
        return torch.cat((segmentation, channels), dim=0)


def _transform_list(compose) -> List:
    transforms = getattr(compose, "transforms", None)
    if transforms is None:
        raise TypeError(
            f"Expected a ComposeTransforms with a .transforms list, got "
            f"{type(compose).__name__}. nnU-Net's get_training_transforms is "
            "supposed to return one; if it no longer does, src/training/trainers "
            "needs updating for this nnU-Net release."
        )
    return transforms


def insert_before_deep_supervision(compose, transform: BasicTransform):
    """Insert ``transform`` just before deep-supervision downsampling.

    Falls back to appending when deep supervision is disabled and no
    downsampling transform is present. Mutates and returns ``compose``.
    """
    transforms = _transform_list(compose)
    index = len(transforms)
    for i, existing in enumerate(transforms):
        if type(existing).__name__ == DS_TRANSFORM_CLASS_NAME:
            index = i
            break
    transforms.insert(index, transform)
    compose.transforms = transforms
    return compose


def append_transform(compose, transform: BasicTransform):
    """Append ``transform`` at the very end of the pipeline. Mutates ``compose``."""
    transforms = _transform_list(compose)
    transforms.append(transform)
    compose.transforms = transforms
    return compose


def strip_auxiliary_channels(
    target: Union[torch.Tensor, List[torch.Tensor]],
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Return only the label channel of a target tensor (or list of them)."""
    if isinstance(target, list):
        return [t[:, :1] for t in target]
    return target[:, :1]


def auxiliary_channels(
    target: Union[torch.Tensor, List[torch.Tensor]], expected: int
) -> torch.Tensor:
    """Return the auxiliary channels of the full-resolution target.

    ``expected`` is checked because a silent mismatch here shows up much later
    as an inscrutable shape error inside the loss.
    """
    full_resolution = target[0] if isinstance(target, list) else target
    auxiliary = full_resolution[:, 1:]
    if auxiliary.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} auxiliary target channel(s) appended to the "
            f"segmentation, found {auxiliary.shape[1]}. The transform that "
            "appends them and the trainer that consumes them disagree — most "
            "likely the trainer's affinity_offsets was changed without also "
            "rebuilding the augmentation pipeline (i.e. training was resumed "
            "rather than restarted)."
        )
    return auxiliary


__all__ = [
    "AffinityTargetTransform",
    "TubedSkeletonTransform",
    "append_transform",
    "auxiliary_channels",
    "insert_before_deep_supervision",
    "strip_auxiliary_channels",
]
