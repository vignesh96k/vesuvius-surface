"""1-epoch smoke-test variant of nnUNetTrainerSkeletonRecallAffinity, used only to verify the
cascade config (3d_cascade_fullres, is_cascaded=True, previous-stage seg channel) combines
correctly with the skeleton-recall+affinity losses before committing to a real 100-epoch run.
Not used for any real training result.
"""

from __future__ import annotations

import torch

from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecallAffinity import nnUNetTrainerSkeletonRecallAffinity


class nnUNetTrainerSkeletonRecallAffinity_1epoch(nnUNetTrainerSkeletonRecallAffinity):  # noqa: N801
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 5
        self.num_val_iterations_per_epoch = 2
