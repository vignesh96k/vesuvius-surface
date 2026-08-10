#!/usr/bin/env python3
"""Same as run_finetuning_stunet_v281.py, but freezes the two shallowest encoder stages
(conv_blocks_context.0, conv_blocks_context.1) immediately after loading STU-Net's
pretrained weights -- before any training step runs.

Rationale: those stages are the ones closest to raw input, most likely to encode generic
edge/texture/gradient responses that should transfer regardless of domain (clinical CT
organs vs. papyrus micro-CT). The later encoder stages (2-5) and the entire decoder encode
increasingly task-specific semantics (STU-Net's pretraining target was organ/bone/vessel
*shape*, i.e. "body anatomy") that don't apply to a thin ink-surface-detection task at all --
those stay trainable so they can actually be relearned for our task, using the frozen early
layers as a stable, generic feature foundation.

Freezing here only sets requires_grad=False; it does NOT need touching
configure_optimizers() separately -- autograd never populates .grad for frozen params, and
nnUNetTrainer's SGD optimizer already skips params with .grad is None during .step().
"""
from unittest.mock import patch

import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.run.run_training import run_training_entry

FROZEN_STAGE_PREFIXES = ("conv_blocks_context.0.", "conv_blocks_context.1.")


def load_stunet_pretrained_weights_freeze_early(network, fname, verbose=False):
    saved_model = torch.load(fname, weights_only=False)

    if fname.endswith('pth'):
        pretrained_dict = saved_model['network_weights']
    elif fname.endswith('model'):
        pretrained_dict = saved_model['state_dict']
    else:
        raise ValueError(f"Unrecognized checkpoint extension for {fname}")

    skip_strings_in_pretrained = ['seg_outputs']

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    num_inputs = model_dict['conv_blocks_context.0.0.conv1.weight'].shape[1]
    if num_inputs > 1:
        pretrained_conv1_weight = pretrained_dict['conv_blocks_context.0.0.conv1.weight']
        pretrained_conv3_weight = pretrained_dict['conv_blocks_context.0.0.conv3.weight']
        pretrained_dict['conv_blocks_context.0.0.conv1.weight'] = pretrained_conv1_weight.repeat(1, num_inputs, 1, 1, 1)
        pretrained_dict['conv_blocks_context.0.0.conv3.weight'] = pretrained_conv3_weight.repeat(1, num_inputs, 1, 1, 1)

    for key, _ in model_dict.items():
        if all(i not in key for i in skip_strings_in_pretrained):
            assert key in pretrained_dict, (
                f"Key {key} is missing in the pretrained model weights."
            )
            assert model_dict[key].shape == pretrained_dict[key].shape, (
                f"Shape mismatch for {key}: pretrained {pretrained_dict[key].shape} vs "
                f"network {model_dict[key].shape}"
            )

    pretrained_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict.keys() and all(i not in k for i in skip_strings_in_pretrained)
    }
    model_dict.update(pretrained_dict)
    print(f"################### Loading STU-Net pretrained weights from {fname} ###################")
    mod.load_state_dict(model_dict)

    n_frozen, n_trainable = 0, 0
    for name, param in mod.named_parameters():
        if name.startswith(FROZEN_STAGE_PREFIXES):
            param.requires_grad = False
            n_frozen += 1
        else:
            n_trainable += 1
    print(
        f"################### Froze {n_frozen} parameter tensors in "
        f"{FROZEN_STAGE_PREFIXES} ({n_trainable} tensors remain trainable) ###################"
    )


if __name__ == '__main__':
    with patch(
        "nnunetv2.run.run_training.load_pretrained_weights",
        load_stunet_pretrained_weights_freeze_early,
    ):
        run_training_entry()
