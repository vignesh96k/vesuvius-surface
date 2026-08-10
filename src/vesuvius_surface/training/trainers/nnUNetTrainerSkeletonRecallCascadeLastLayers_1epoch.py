"""1-epoch smoke-test variant of nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs.
Verifies the frozen-backbone + cascade + skeleton-recall combination actually trains (forward,
backward, optimizer step, checkpoint save) before committing to the real 10-epoch run. Not used
for any real result.
"""

from __future__ import annotations

import torch

from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs import (
    nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs,
)


class nnUNetTrainerSkeletonRecallCascadeLastLayers_1epoch(  # noqa: N801
    nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs
):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 5
        self.num_val_iterations_per_epoch = 2
