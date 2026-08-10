"""Surface Detection 3D dataset package."""

from vesuvius_surface.data.dataloader import build_dataloader, build_dataloaders_from_config
from vesuvius_surface.data.dataset import SegmentationDataset, SurfacePatchDataset, SurfaceVolumeDataset
from vesuvius_surface.data.io import VolumeRecord, build_volume_index, load_volume, read_metadata_csv
from vesuvius_surface.data.patching import PatchConfig, PatchConfig3D, PatchCoord3D, build_patch_index_3d, extract_patch_3d
from vesuvius_surface.data.schema import LABEL_BG, LABEL_IGNORE, LABEL_SURFACE, VALID_LABELS
from vesuvius_surface.data.transforms import build_transforms, normalize_volume
from vesuvius_surface.data.validate import ValidationReport, validate_dataset

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
