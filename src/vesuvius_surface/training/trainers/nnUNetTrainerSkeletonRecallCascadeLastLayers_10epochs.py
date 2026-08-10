"""Last-layers-only cascade fine-tune, 10 epochs.

Targeted diagnostic: take arunodhayan's real cascade checkpoint
(``80003dlowres_80003dcascade_checkpoint_final.pth``) and fine-tune only the final decoder
stage + every deep-supervision segmentation head, with everything else (all 6 encoder stages,
decoder stages 0-3, their transpconvs) frozen at the pretrained weights. Ensemble A and B are
untouched by this experiment entirely -- the cascade's previous-stage input is their zero-shot
combine, matching arunodhayan's real inference recipe.

Loss is unchanged from :class:`nnUNetTrainerSkeletonRecall` -- stock DC+CE (unmodified,
deep-supervision weighted) plus the skeleton-recall auxiliary term. No affinity, no highpass:
both the 100-epoch full fine-tune and a 20-epoch full-network diagnostic showed those hurt (see
research_log / NEXT_TASKS.md), so this run isolates one more variable: does freezing almost the
whole network -- leaving only the head trainable -- avoid the same regression the full
fine-tunes showed, even under the same loss and short schedule?

Layer split verified against the checkpoint's real state_dict, not assumed: encoder has stages
0-5, decoder has stages 0-4 and seg_layers 0-4 (one deep-supervision head per decoder stage,
index 4 = full resolution / final output). Trainable: decoder.stages.4, decoder.transpconvs.4,
decoder.seg_layers.* (all five -- deep supervision computes loss against every one of them, so
freezing some while leaving others trainable underneath a frozen backbone would be incoherent).
Frozen: everything else, including the encoder-inside-decoder aliased parameters (nnU-Net's
ResidualEncoderUNet decoder holds a reference to the same encoder object for skip connections,
so those show up under both `encoder.*` and `decoder.encoder.*` state_dict keys -- freezing by
iterating `named_parameters()` handles this correctly since it's the same underlying tensor,
not a duplicate).
"""

from __future__ import annotations

import torch

from vesuvius_surface.training.trainers.nnUNetTrainerSkeletonRecall import nnUNetTrainerSkeletonRecall

# Trainable if the parameter's dotted name starts with any of these prefixes.
_TRAINABLE_PREFIXES: tuple[str, ...] = (
    "decoder.stages.4.",
    "decoder.transpconvs.4.",
    "decoder.seg_layers.",
)


class nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs(nnUNetTrainerSkeletonRecall):  # noqa: N801
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 10

    def initialize(self) -> None:
        super().initialize()
        self._freeze_all_but_last_layers()

    def _freeze_all_but_last_layers(self) -> None:
        net = self.network
        # No torch.compile in this project (nnUNet_compile=false throughout), so `net` is the
        # raw module and named_parameters() keys match the checkpoint's state_dict keys directly.
        n_trainable_params = n_frozen_params = 0
        n_trainable_tensors = n_frozen_tensors = 0
        for name, param in net.named_parameters():
            if any(name.startswith(prefix) for prefix in _TRAINABLE_PREFIXES):
                param.requires_grad = True
                n_trainable_params += param.numel()
                n_trainable_tensors += 1
            else:
                param.requires_grad = False
                n_frozen_params += param.numel()
                n_frozen_tensors += 1

        total = n_trainable_params + n_frozen_params
        self.print_to_log_file(
            f"[LastLayers] trainable={n_trainable_params:,} params "
            f"({n_trainable_tensors} tensors, {100 * n_trainable_params / total:.2f}%), "
            f"frozen={n_frozen_params:,} params ({n_frozen_tensors} tensors)"
        )
