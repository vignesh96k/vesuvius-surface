"""3D Surface Detection PyTorch datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data.io import (
    VolumeRecord,
    build_volume_index,
    clear_volume_cache,
    load_volume,
    load_volume_cached,
    probe_volume,
)
from data.patching import (
    PatchConfig3D,
    PatchCoord3D,
    build_patch_index_3d,
    extract_patch_3d,
)
from data.transforms import normalize_volume

logger = logging.getLogger(__name__)

TransformFn = Callable[[dict[str, Any]], dict[str, Any]]


class SurfaceVolumeDataset(Dataset):
    """Dataset yielding full 3D volumes (for EDA / export — heavy).

    Each item::

        {
          "image": FloatTensor (1, D, H, W),
          "label": LongTensor (D, H, W) or None,
          "volume_id": str,
          "scroll_id": str,
        }
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        volume_ids: Optional[Sequence[str]] = None,
        scroll_ids: Optional[Sequence[str]] = None,
        normalize: str = "zscore",
        normalize_kwargs: Optional[dict[str, Any]] = None,
        transform: Optional[TransformFn] = None,
        require_label: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.normalize = normalize
        self.normalize_kwargs = normalize_kwargs or {}
        self.transform = transform
        self.records: list[VolumeRecord] = build_volume_index(
            self.root,
            split=split,
            volume_ids=volume_ids,
            scroll_ids=scroll_ids,
            require_label=require_label,
        )
        logger.info("SurfaceVolumeDataset: %d volumes (split=%s)", len(self.records), split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        image = load_volume(rec.image_path).astype(np.float32)
        image = normalize_volume(image, method=self.normalize, **self.normalize_kwargs)
        sample: dict[str, Any] = {
            "image": torch.from_numpy(image[None, ...]).float(),
            "volume_id": rec.volume_id,
            "scroll_id": rec.scroll_id,
            "index": index,
        }
        if rec.label_path is not None:
            label = load_volume(rec.label_path)
            sample["label"] = torch.from_numpy(np.ascontiguousarray(label)).long()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def inventory(self) -> list[dict[str, Any]]:
        """Lightweight path inventory without loading voxels."""
        rows = []
        for rec in self.records:
            rows.append(
                {
                    "volume_id": rec.volume_id,
                    "scroll_id": rec.scroll_id,
                    "image_path": str(rec.image_path),
                    "label_path": None if rec.label_path is None else str(rec.label_path),
                    "has_label": rec.has_label,
                }
            )
        return rows


class SurfacePatchDataset(Dataset):
    """Dataset yielding fixed-size 3D patches for training / analysis.

    Memory strategy: probe shapes + load labels once to build the patch index;
    image patches are read via a small volume LRU at ``__getitem__`` time.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        patch_config: Optional[PatchConfig3D] = None,
        volume_ids: Optional[Sequence[str]] = None,
        scroll_ids: Optional[Sequence[str]] = None,
        transform: Optional[TransformFn] = None,
        normalize: str = "zscore",
        normalize_kwargs: Optional[dict[str, Any]] = None,
        require_label: Optional[bool] = None,
        cache_labels: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.patch_config = patch_config or PatchConfig3D()
        self.transform = transform
        self.normalize = normalize
        self.normalize_kwargs = normalize_kwargs or {}
        self.cache_labels = cache_labels

        self.records = build_volume_index(
            self.root,
            split=split,
            volume_ids=volume_ids,
            scroll_ids=scroll_ids,
            require_label=require_label,
        )

        self._shapes: list[tuple[int, int, int]] = []
        self._labels: list[Optional[np.ndarray]] = []
        for rec in self.records:
            info = probe_volume(rec.image_path)
            shape = info["shape"]
            assert isinstance(shape, tuple)
            self._shapes.append(shape)  # type: ignore[arg-type]
            if rec.label_path is not None:
                label = load_volume(rec.label_path)
                if tuple(label.shape) != shape:
                    raise ValueError(
                        f"Label/image shape mismatch for {rec.volume_id}: "
                        f"{label.shape} vs {shape}"
                    )
                self._labels.append(label)
            else:
                self._labels.append(None)

        self.patches: list[PatchCoord3D] = build_patch_index_3d(
            self._labels, self._shapes, self.patch_config
        )
        if not cache_labels:
            self._labels = [None] * len(self.records)

        logger.info(
            "SurfacePatchDataset: volumes=%d patches=%d size=%s stride=%s",
            len(self.records),
            len(self.patches),
            self.patch_config.resolved_size(),
            self.patch_config.resolved_stride(),
        )

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> dict[str, Any]:
        coord = self.patches[index]
        rec = self.records[coord.volume_index]
        size = self.patch_config.resolved_size()

        volume = load_volume_cached(rec.image_path).astype(np.float32)
        image = extract_patch_3d(
            volume, coord.z, coord.y, coord.x, size, pad_value=self.patch_config.pad_value
        )
        image = normalize_volume(image, method=self.normalize, **self.normalize_kwargs)

        sample: dict[str, Any] = {
            "image": torch.from_numpy(np.ascontiguousarray(image[None, ...])).float(),
            "volume_id": rec.volume_id,
            "scroll_id": rec.scroll_id,
            "z": coord.z,
            "y": coord.y,
            "x": coord.x,
            "index": index,
        }

        label_full = self._labels[coord.volume_index]
        if label_full is None and rec.label_path is not None:
            label_full = load_volume(rec.label_path)
        if label_full is not None:
            label = extract_patch_3d(
                label_full,
                coord.z,
                coord.y,
                coord.x,
                size,
                pad_value=self.patch_config.label_pad_value,
            )
            sample["label"] = torch.from_numpy(np.ascontiguousarray(label)).long()

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def clear_caches(self) -> None:
        clear_volume_cache()


# Alias for older imports in stubs / docs.
SegmentationDataset = SurfacePatchDataset
