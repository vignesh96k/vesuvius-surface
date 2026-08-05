"""Surface Detection 3D dataset package."""

from datasets.dataloader import build_dataloader, build_dataloaders_from_config
from datasets.dataset import SegmentationDataset, SurfacePatchDataset, SurfaceVolumeDataset
from datasets.io import VolumeRecord, build_volume_index, load_volume, read_metadata_csv
from datasets.patching import PatchConfig, PatchConfig3D, PatchCoord3D, build_patch_index_3d, extract_patch_3d
from datasets.schema import LABEL_BG, LABEL_IGNORE, LABEL_SURFACE, VALID_LABELS
from datasets.transforms import build_transforms, normalize_volume
from datasets.validate import ValidationReport, validate_dataset

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
