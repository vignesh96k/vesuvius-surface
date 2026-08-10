"""350-epoch extended run of the clDice+ScheduleFree candidate, if it wins the decision
point. 350 not 700 -- ScheduleFree's own constant-LR, no-schedule design converges faster,
and clDice's confirmed ~69s/epoch overhead (vs ~35s/epoch stock) means 350 epochs costs
about the same wall-clock time as 700 epochs of a stock-loss candidate.

Explicit __init__ signature, not *args/**kwargs -- required for nnU-Net's own
self.my_init_kwargs introspection (inspect.signature(self.__init__) indexed against the
caller frame's locals()); a generic passthrough breaks it (KeyError: 'args'), confirmed
today while fixing the same bug in vesuvius-surface's trainers.
"""

from __future__ import annotations

import torch

from vesuvius_surface.training.trainers.nnUNetTrainerSeeded_ClDice_ScheduleFree import nnUNetTrainerSeeded_ClDice_ScheduleFree


class nnUNetTrainerSeeded_ClDice_ScheduleFree_350epochs(nnUNetTrainerSeeded_ClDice_ScheduleFree):  # noqa: N801
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 350
