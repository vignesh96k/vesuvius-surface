"""Guarded, lazy imports of every third-party symbol this package depends on.

The training code runs on a remote Linux box inside whichever environment
happens to have nnU-Net installed. When a symbol moves between nnU-Net releases
the resulting error is usually a bare ``cannot import name X from Y``, which
tells you nothing about what to do next. Everything resolved through this
module instead raises a message naming the missing module, the symbol, the
version actually installed, and the command that fixes it.

Resolution is lazy (PEP 562 module ``__getattr__``) for a second reason: the
target builders in :mod:`training.affinity` and :mod:`training.skeleton` are
pure numpy and are unit-tested without nnU-Net or batchgeneratorsv2 present.
Importing them must not drag in a deep-learning framework.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

_PYTHONPATH_HINT = (
    "Also make sure this repository's src/ is importable:\n"
    "    export PYTHONPATH=/path/to/vesuvius-surface/src:$PYTHONPATH"
)

# name -> (module path, attribute, pip distribution, install command)
_REGISTRY: Dict[str, Tuple[str, str, str, str]] = {}


def _register(
    name: str, module_path: str, attribute: str, distribution: str, install: str
) -> None:
    _REGISTRY[name] = (module_path, attribute, distribution, install)


_NNUNET_INSTALL = "pip install -U nnunetv2"
_BGV2_INSTALL = "pip install -U batchgeneratorsv2"

for _name, _module, _attr in (
    ("nnUNetTrainer", "nnunetv2.training.nnUNetTrainer.nnUNetTrainer", "nnUNetTrainer"),
    (
        "DeepSupervisionWrapper",
        "nnunetv2.training.loss.deep_supervision",
        "DeepSupervisionWrapper",
    ),
    (
        "MemoryEfficientSoftDiceLoss",
        "nnunetv2.training.loss.dice",
        "MemoryEfficientSoftDiceLoss",
    ),
    ("get_tp_fp_fn_tn", "nnunetv2.training.loss.dice", "get_tp_fp_fn_tn"),
    (
        "RobustCrossEntropyLoss",
        "nnunetv2.training.loss.robust_ce_loss",
        "RobustCrossEntropyLoss",
    ),
    ("DC_and_CE_loss", "nnunetv2.training.loss.compound_losses", "DC_and_CE_loss"),
    ("AllGatherGrad", "nnunetv2.utilities.ddp_allgather", "AllGatherGrad"),
    ("softmax_helper_dim1", "nnunetv2.utilities.helpers", "softmax_helper_dim1"),
    ("dummy_context", "nnunetv2.utilities.helpers", "dummy_context"),
    ("empty_cache", "nnunetv2.utilities.helpers", "empty_cache"),
    (
        "get_network_from_plans",
        "nnunetv2.utilities.get_network_from_plans",
        "get_network_from_plans",
    ),
):
    _register(_name, _module, _attr, "nnunetv2", _NNUNET_INSTALL)

_register(
    "BasicTransform",
    "batchgeneratorsv2.transforms.base.basic_transform",
    "BasicTransform",
    "batchgeneratorsv2",
    _BGV2_INSTALL,
)

for _name, _module, _attr, _dist, _install in (
    ("skeletonize", "skimage.morphology", "skeletonize", "scikit-image", "pip install -U scikit-image"),
    ("cc_label", "scipy.ndimage", "label", "scipy", "pip install -U scipy"),
    ("binary_dilation", "scipy.ndimage", "binary_dilation", "scipy", "pip install -U scipy"),
    (
        "generate_binary_structure",
        "scipy.ndimage",
        "generate_binary_structure",
        "scipy",
        "pip install -U scipy",
    ),
):
    _register(_name, _module, _attr, _dist, _install)


def _installed_version(distribution: str) -> str:
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:  # noqa: BLE001 - version reporting must never be fatal
        return "not installed"


def resolve(name: str) -> Any:
    """Import and return a registered symbol, or raise an actionable ImportError."""
    try:
        module_path, attribute, distribution, install = _REGISTRY[name]
    except KeyError as exc:
        raise AttributeError(f"training.compat has no symbol {name!r}") from exc

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"src/training needs '{attribute}' from '{module_path}', but that "
            f"module could not be imported.\n"
            f"Installed {distribution}: {_installed_version(distribution)}\n"
            f"Fix with:\n    {install}\n{_PYTHONPATH_HINT}"
        ) from exc

    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(
            f"'{module_path}' imported fine but has no attribute '{attribute}'.\n"
            f"Installed {distribution}: {_installed_version(distribution)}\n"
            "src/training targets nnunetv2 >= 2.5 (the batchgeneratorsv2 "
            "transform API). An older release moves or renames this symbol.\n"
            f"Fix with:\n    {install}"
        ) from exc


def __getattr__(name: str) -> Any:  # PEP 562
    value = resolve(name)
    globals()[name] = value  # resolve once, then behave like a normal attribute
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_REGISTRY))


__all__ = sorted(_REGISTRY)
