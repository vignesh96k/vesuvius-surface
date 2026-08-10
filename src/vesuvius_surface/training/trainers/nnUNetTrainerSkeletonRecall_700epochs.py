"""700-epoch extended run of skeleton-recall, if it wins the decision point. 700 not 1000
-- see nnUNetTrainerSeeded's own docstring: the original 1000-epoch baseline run plateaued
at epoch 639 and never improved again through epoch 999."""

from __future__ import annotations

import torch

from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecall import nnUNetTrainerSkeletonRecall


class nnUNetTrainerSkeletonRecall_700epochs(nnUNetTrainerSkeletonRecall):  # noqa: N801
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 700
