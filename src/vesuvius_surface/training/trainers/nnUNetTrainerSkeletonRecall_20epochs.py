"""20-epoch variant of nnUNetTrainerSkeletonRecall, for a quick diagnostic fine-tune: skeleton
-recall loss only (no affinity), raw CT input (no highpass), short schedule -- isolating the
auxiliary-loss idea from the highpass domain-shift and long-schedule choices used in the
100-epoch highpass fine-tune that regressed badly (see NEXT_TASKS.md).
"""

from __future__ import annotations

import torch

from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecall import nnUNetTrainerSkeletonRecall


class nnUNetTrainerSkeletonRecall_20epochs(nnUNetTrainerSkeletonRecall):  # noqa: N801
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 20
