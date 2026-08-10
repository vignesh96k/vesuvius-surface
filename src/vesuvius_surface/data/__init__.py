"""Surface Detection 3D dataset package.

``dataloader``/``dataset``/``transforms`` need torch; ``io``/``patching``/``schema``/
``validate`` are pure numpy/tifffile. Torch-dependent symbols are resolved lazily (PEP 562
module ``__getattr__``) so importing this package doesn't force torch to load -- real
consumers care: ``postprocess.pipeline`` imports only ``data.io.load_volume``, and needs to
work in the torch-free ``vesuvius_eval`` env (see environment-eval.yml / README.md
Quickstart). Eagerly importing every submodule here previously broke that separation: any
caller of ``load_volume`` alone was forced to import torch anyway, which doesn't exist in
that env -- a real ``ModuleNotFoundError`` hit running ``scripts/evaluation/score_model.py``
there, not a hypothetical.
"""

from __future__ import annotations

import importlib
from typing import Any

from vesuvius_surface.data.io import VolumeRecord, build_volume_index, load_volume, read_metadata_csv
from vesuvius_surface.data.patching import PatchConfig, PatchConfig3D, PatchCoord3D, build_patch_index_3d, extract_patch_3d
from vesuvius_surface.data.schema import LABEL_BG, LABEL_IGNORE, LABEL_SURFACE, VALID_LABELS
from vesuvius_surface.data.validate import ValidationReport, validate_dataset

_LAZY_TORCH_SYMBOLS: dict[str, str] = {
    "build_dataloader": "vesuvius_surface.data.dataloader",
    "build_dataloaders_from_config": "vesuvius_surface.data.dataloader",
    "SegmentationDataset": "vesuvius_surface.data.dataset",
    "SurfacePatchDataset": "vesuvius_surface.data.dataset",
    "SurfaceVolumeDataset": "vesuvius_surface.data.dataset",
    "build_transforms": "vesuvius_surface.data.transforms",
    "normalize_volume": "vesuvius_surface.data.transforms",
}


def __getattr__(name: str) -> Any:  # PEP 562
    module_path = _LAZY_TORCH_SYMBOLS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'vesuvius_surface.data' has no attribute {name!r}")
    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value  # resolve once, then behave like a normal module attribute
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY_TORCH_SYMBOLS))


__all__ = [
    "LABEL_BG",
    "LABEL_IGNORE",
    "LABEL_SURFACE",
    "VALID_LABELS",
    "PatchConfig",
    "PatchConfig3D",
    "PatchCoord3D",
    "SegmentationDataset",
    "SurfacePatchDataset",
    "SurfaceVolumeDataset",
    "ValidationReport",
    "VolumeRecord",
    "build_dataloader",
    "build_dataloaders_from_config",
    "build_patch_index_3d",
    "build_transforms",
    "build_volume_index",
    "extract_patch_3d",
    "load_volume",
    "normalize_volume",
    "read_metadata_csv",
    "validate_dataset",
]
