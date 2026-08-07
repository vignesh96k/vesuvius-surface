"""Attach an affinity head to nnU-Net's decoder at full resolution.

Placement
---------
The head goes on the **last decoder stage**, not the bottleneck. A 128^3 patch
through the 6 stages of the ResEnc L plan leaves a 4^3 feature map; a dense
voxel-pair target cannot be read off that. The last decoder stage is at input
resolution, and its output is exactly what nnU-Net's own highest-resolution deep
supervision head consumes — so a 1x1x1 convolution to ``n_offsets`` channels is
the same construction nnU-Net already uses for segmentation, just with a
different target.

Why a submodule rather than a wrapper
-------------------------------------
The obvious implementation is a ``nn.Module`` wrapping the U-Net and returning
``(seg, affinities)``. It has three problems, all of which matter here:

1.  ``build_network_architecture`` is what *inference* calls too, so the wrapper
    would exist at prediction time. Stage 2a is supposed to leave inference
    untouched.
2.  Wrapping renames every parameter to ``backbone.*``, which breaks
    ``nnUNetv2_train -pretrained_weights`` against an m7 or STU-Net checkpoint.
3.  ``torch.compile`` and DDP both have opinions about modules whose forward
    return type depends on a Python attribute.

Registering the head as ``decoder.affinity_head`` instead leaves every existing
parameter name untouched (so pretrained loading still matches, and the extra key
is simply absent from the donor checkpoint), leaves ``network(x)`` returning
exactly what it returned before, and still puts the head in ``state_dict`` and
``parameters()`` so it is saved and optimised.

The trainer then gets the full-resolution feature map through a forward
pre-hook on the final segmentation layer, which is fed that exact tensor.

Caveat: forward hooks and ``torch.compile`` interact poorly, so the affinity
trainer disables compilation. See ``nnUNetTrainerAffinity._do_i_compile``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
from torch import nn

try:  # torch >= 2.0; nnU-Net imports it from the same place
    from torch._dynamo import OptimizedModule
except ImportError:  # pragma: no cover - depends on the torch build
    OptimizedModule = None  # type: ignore[assignment]

AFFINITY_HEAD_ATTR = "affinity_head"

_NO_DECODER = (
    "Could not find a '{missing}' on the network built from the plans "
    "({network_type}). The affinity head attaches to nnU-Net's standard "
    "UNetDecoder / UNetResDecoder, which exposes .decoder.seg_layers. If you "
    "are training an architecture that does not (STU-Net's fork, for example, "
    "names things differently), src/training/network.py needs a branch for it."
)


def unwrap_network(network: nn.Module) -> nn.Module:
    """Strip DistributedDataParallel and torch.compile wrappers."""
    module = network
    for _ in range(4):  # DDP(compile(net)) is two layers; 4 is slack
        if isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
            continue
        if OptimizedModule is not None and isinstance(module, OptimizedModule):
            module = module._orig_mod
            continue
        break
    return module


def _decoder_of(network: nn.Module):
    module = unwrap_network(network)
    decoder = getattr(module, "decoder", None)
    if decoder is None:
        raise RuntimeError(
            _NO_DECODER.format(missing="decoder", network_type=type(module).__name__)
        )
    seg_layers = getattr(decoder, "seg_layers", None)
    if seg_layers is None or len(seg_layers) == 0:
        raise RuntimeError(
            _NO_DECODER.format(
                missing="decoder.seg_layers", network_type=type(module).__name__
            )
        )
    return decoder


def attach_affinity_head(network: nn.Module, num_affinities: int) -> nn.Module:
    """Register ``decoder.affinity_head``: a 1x1x1 conv to ``num_affinities``.

    Idempotent — calling twice returns the existing head, so it is safe in
    ``build_network_architecture``, which nnU-Net may call more than once.
    """
    if num_affinities < 1:
        raise ValueError(f"num_affinities must be >= 1, got {num_affinities}")

    decoder = _decoder_of(network)

    existing = getattr(decoder, AFFINITY_HEAD_ATTR, None)
    if existing is not None:
        if existing.out_channels != num_affinities:
            raise ValueError(
                f"An affinity head with {existing.out_channels} outputs is already "
                f"attached but {num_affinities} were requested. The offset list is "
                "baked into the checkpoint; changing it requires a fresh run under "
                "a new trainer name, not a resume."
            )
        return existing

    final_seg_layer = decoder.seg_layers[-1]
    head = type(final_seg_layer)(
        in_channels=final_seg_layer.in_channels,
        out_channels=num_affinities,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
    )
    # Same initialisation nnU-Net uses for its own segmentation heads.
    nn.init.kaiming_normal_(head.weight, a=1e-2)
    if head.bias is not None:
        nn.init.constant_(head.bias, 0)

    setattr(decoder, AFFINITY_HEAD_ATTR, head)
    return head


def get_affinity_head(network: nn.Module) -> nn.Module:
    """Return the attached affinity head, or explain that it is missing."""
    decoder = _decoder_of(network)
    head = getattr(decoder, AFFINITY_HEAD_ATTR, None)
    if head is None:
        raise RuntimeError(
            "No affinity head is attached to this network. It should have been "
            "created by nnUNetTrainerAffinity.build_network_architecture; if you "
            "are loading a checkpoint, make sure the trainer name recorded in the "
            "results folder is the affinity trainer and not the plain one."
        )
    return head


class FullResolutionFeatureTap:
    """Capture the tensor fed to the highest-resolution segmentation layer.

    That tensor is the last decoder stage's output at input resolution. A
    forward pre-hook is used rather than reimplementing the decoder forward, so
    this keeps working when nnU-Net changes decoder internals.

    Capture is opt-in per forward pass (``with tap.capturing():``) because
    holding a reference to an activation keeps its autograd graph alive; during
    inference and validation we want nothing retained.
    """

    def __init__(self, network: nn.Module) -> None:
        decoder = _decoder_of(network)
        self.enabled = False
        self._features: Optional[torch.Tensor] = None
        self._handle = decoder.seg_layers[-1].register_forward_pre_hook(self._hook)

    def _hook(self, module: nn.Module, inputs):  # noqa: ANN001 - torch hook signature
        if self.enabled:
            self._features = inputs[0]

    @contextmanager
    def capturing(self):
        previous = self.enabled
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = previous

    def take(self) -> torch.Tensor:
        """Return the captured features and drop the reference."""
        features = self._features
        self._features = None
        if features is None:
            raise RuntimeError(
                "The full-resolution decoder features were not captured during "
                "the forward pass. Either the forward ran outside "
                "'with tap.capturing():', or torch.compile swallowed the forward "
                "pre-hook. The affinity trainer disables compilation for exactly "
                "this reason; if you re-enabled it with nnUNet_compile=1, that is "
                "the cause."
            )
        return features

    def close(self) -> None:
        self._handle.remove()

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


__all__ = [
    "AFFINITY_HEAD_ATTR",
    "FullResolutionFeatureTap",
    "attach_affinity_head",
    "get_affinity_head",
    "unwrap_network",
]
