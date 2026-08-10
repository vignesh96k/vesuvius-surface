"""nnU-Net trainer variants.

These are the classes ``nnUNetv2_train -tr <name>`` resolves. nnU-Net finds
trainers by scanning ``nnunetv2/training/nnUNetTrainer`` for a class of the
requested name, so they have to be reachable from a module inside the installed
package — run ``scripts/register_nnunet_trainers.py`` to drop a shim there that
imports these.
"""

from __future__ import annotations

from vesuvius_surface.training.trainers.nnUNetTrainerAffinity import (
    nnUNetTrainerAffinity,
    nnUNetTrainerAffinity_allpairs,
    nnUNetTrainerAffinity_shortrange,
    nnUNetTrainerAffinity_w01,
    nnUNetTrainerAffinity_w05,
    nnUNetTrainerAffinity_w2,
)
from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecall import (
    nnUNetTrainerSkeletonRecall,
    nnUNetTrainerSkeletonRecall_notube,
    nnUNetTrainerSkeletonRecall_w05,
    nnUNetTrainerSkeletonRecall_w2,
    nnUNetTrainerSkeletonRecall_w4,
)

__all__ = [
    "nnUNetTrainerAffinity",
    "nnUNetTrainerAffinity_allpairs",
    "nnUNetTrainerAffinity_shortrange",
    "nnUNetTrainerAffinity_w01",
    "nnUNetTrainerAffinity_w05",
    "nnUNetTrainerAffinity_w2",
    "nnUNetTrainerSkeletonRecall",
    "nnUNetTrainerSkeletonRecall_notube",
    "nnUNetTrainerSkeletonRecall_w05",
    "nnUNetTrainerSkeletonRecall_w2",
    "nnUNetTrainerSkeletonRecall_w4",
]
