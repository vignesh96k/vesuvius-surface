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
from vesuvius_surface.training.trainers.nnUNetTrainerAffinity_100epochs import (
    nnUNetTrainerAffinity_100epochs,
)
from vesuvius_surface.training.trainers.nnUNetTrainerAffinity_700epochs import (
    nnUNetTrainerAffinity_700epochs,
)
from vesuvius_surface.training.trainers.nnUNetTrainerSeeded import (
    nnUNetTrainerSeeded,
    nnUNetTrainerSeeded_100epochs,
)
from vesuvius_surface.training.trainers.nnUNetTrainerSeeded_ClDice_ScheduleFree import (
    nnUNetTrainerSeeded_ClDice_ScheduleFree,
)
from vesuvius_surface.training.trainers.nnUNetTrainerSeeded_ClDice_ScheduleFree_350epochs import (
    nnUNetTrainerSeeded_ClDice_ScheduleFree_350epochs,
)
from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecall import (
    nnUNetTrainerSkeletonRecall,
    nnUNetTrainerSkeletonRecall_notube,
    nnUNetTrainerSkeletonRecall_w05,
    nnUNetTrainerSkeletonRecall_w2,
    nnUNetTrainerSkeletonRecall_w4,
)
from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecall_20epochs import (
    nnUNetTrainerSkeletonRecall_20epochs,
)
from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecall_100epochs import (
    nnUNetTrainerSkeletonRecall_100epochs,
)
from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecall_700epochs import (
    nnUNetTrainerSkeletonRecall_700epochs,
)
from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecallAffinity import (
    nnUNetTrainerSkeletonRecallAffinity,
)
from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecallAffinity_1epoch import (
    nnUNetTrainerSkeletonRecallAffinity_1epoch,
)
from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs import (
    nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs,
)
from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecallCascadeLastLayers_1epoch import (
    nnUNetTrainerSkeletonRecallCascadeLastLayers_1epoch,
)

__all__ = [
    "nnUNetTrainerAffinity",
    "nnUNetTrainerAffinity_100epochs",
    "nnUNetTrainerAffinity_700epochs",
    "nnUNetTrainerAffinity_allpairs",
    "nnUNetTrainerAffinity_shortrange",
    "nnUNetTrainerAffinity_w01",
    "nnUNetTrainerAffinity_w05",
    "nnUNetTrainerAffinity_w2",
    "nnUNetTrainerSeeded",
    "nnUNetTrainerSeeded_100epochs",
    "nnUNetTrainerSeeded_ClDice_ScheduleFree",
    "nnUNetTrainerSeeded_ClDice_ScheduleFree_350epochs",
    "nnUNetTrainerSkeletonRecall",
    "nnUNetTrainerSkeletonRecall_100epochs",
    "nnUNetTrainerSkeletonRecall_20epochs",
    "nnUNetTrainerSkeletonRecall_700epochs",
    "nnUNetTrainerSkeletonRecall_notube",
    "nnUNetTrainerSkeletonRecall_w05",
    "nnUNetTrainerSkeletonRecall_w2",
    "nnUNetTrainerSkeletonRecall_w4",
    "nnUNetTrainerSkeletonRecallAffinity",
    "nnUNetTrainerSkeletonRecallAffinity_1epoch",
    "nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs",
    "nnUNetTrainerSkeletonRecallCascadeLastLayers_1epoch",
]
