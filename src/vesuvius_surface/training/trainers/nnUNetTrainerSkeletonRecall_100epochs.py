"""100-epoch variant of nnUNetTrainerSkeletonRecall (Stage 1), for the comparison phase.
nnUNetTrainerSkeletonRecall doesn't set num_epochs itself, so it inherits nnUNetTrainer's
stock default (1000). This subclass overrides that -- everything else unchanged.

Explicit __init__ signature (plans, configuration, fold, dataset_json, device) -- matches
nnUNetTrainer's own real signature in this installed version (no unpack_dataset parameter,
unlike some other trainer families e.g. STU-Net's) -- required for nnU-Net's own
self.my_init_kwargs introspection; see nnUNetTrainerSkeletonRecall.py for the full
explanation, confirmed by actually running this trainer.
"""

from __future__ import annotations

import torch

from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecall import nnUNetTrainerSkeletonRecall


class nnUNetTrainerSkeletonRecall_100epochs(nnUNetTrainerSkeletonRecall):  # noqa: N801
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
