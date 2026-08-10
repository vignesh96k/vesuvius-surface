"""30-epoch fine-tune variant of STUNetTrainer_base_ft (58M-param STU-Net-B, pretrained on
TotalSegmentator). "A few epochs" -- enough to see whether the pretrained weights adapt
usefully to our domain without committing to the full 1000-epoch default the upstream class
declares (self.num_epochs = 1000 in STUNetTrainer_base_ft.__init__).

Explicit signature (plans, configuration, fold, dataset_json, device) -- no unpack_dataset;
this installed nnU-Net's own __init__ doesn't have that parameter at all (an older
nnU-Net-v1-era assumption in the upstream STU-Net port), confirmed by actually running this.
"""

from __future__ import annotations

import torch

from vesuvius_surface.training.trainers.STUNetTrainer import STUNetTrainer_base_ft


class STUNetTrainer_base_ft_30epochs(STUNetTrainer_base_ft):  # noqa: N801
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 30
