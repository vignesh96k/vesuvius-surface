"""I/O for Vesuvius Surface Detection 3D TIFF volumes.

Expected layout::

    <root>/
      train.csv | test.csv
      train_images/<id>.tif
      train_labels/<id>.tif
      test_images/<id>.tif
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from data.schema import (
    IMAGE_EXTENSIONS,
    TRAIN_IMAGES_DIRNAME,
    TRAIN_LABELS_DIRNAME,
    csv_name,
    images_dirname,
    label_path,
    labels_dirname,
    volume_path,
)

logger = logging.getLogger(__name__)

try:
    import tifffile

    _HAS_TIFFFILE = True
except ImportError:  # pragma: no cover
    _HAS_TIFFFILE = False


@dataclass(frozen=True)
class VolumeRecord:
    """One volume entry from CSV + resolved filesystem paths."""

    volume_id: str
    scroll_id: str
    image_path: Path
    label_path: Optional[Path]
    split: str

    @property
    def has_label(self) -> bool:
        return self.label_path is not None and self.label_path.exists()


def _require_tifffile() -> None:
    if not _HAS_TIFFFILE:
        raise ImportError(
            "tifffile is required to load Surface Detection volumes. "
            "Install with: pip install tifffile"
        )


def load_volume(path: str | Path) -> np.ndarray:
    """Load a 3D TIFF volume as a contiguous NumPy array ``(D, H, W)``."""
    _require_tifffile()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    array = tifffile.imread(str(path))
    array = np.ascontiguousarray(array)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D/3D volume at {path}, got shape {array.shape}")
    return array


def probe_volume(
    path: str | Path,
    *,
    include_extrema: bool = False,
) -> dict[str, object]:
    """Read shape / dtype / nbytes without keeping the full array in memory.

    Uses TIFF series metadata when possible. Set ``include_extrema=True`` to
    also load the volume and report min/max (slower).
    """
    _require_tifffile()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with tifffile.TiffFile(str(path)) as tif:
        if not tif.series:
            raise ValueError(f"No TIFF series found at {path}")
        series = tif.series[0]
        shape = tuple(int(x) for x in series.shape)
        dtype = np.dtype(series.dtype)

    if len(shape) == 2:
        shape = (1, shape[0], shape[1])
    if len(shape) != 3:
        raise ValueError(f"Expected 2D/3D volume at {path}, got shape {shape}")

    info: dict[str, object] = {
        "shape": shape,
        "dtype": str(dtype),
        "nbytes": int(np.prod(shape) * dtype.itemsize),
    }
    if include_extrema:
        array = load_volume(path)
        info["min"] = float(np.min(array))
        info["max"] = float(np.max(array))
    return info


@lru_cache(maxsize=8)
def _cached_volume(path_str: str) -> np.ndarray:
    """Small LRU for repeated patch extraction from the same volume."""
    return load_volume(path_str)


def load_volume_cached(path: str | Path) -> np.ndarray:
    """Load a volume via process-local LRU (returns a writeable copy)."""
    return _cached_volume(str(Path(path).resolve())).copy()


def clear_volume_cache() -> None:
    """Clear the process-local volume cache."""
    _cached_volume.cache_clear()


def read_metadata_csv(root: str | Path, split: str = "train") -> pd.DataFrame:
    """Load ``train.csv`` / ``test.csv`` and normalize column names."""
    root = Path(root)
    path = root / csv_name(split)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "id" not in cols:
        raise KeyError(f"{path} must contain an 'id' column; got {list(df.columns)}")
    rename = {cols["id"]: "id"}
    if "scroll_id" in cols:
        rename[cols["scroll_id"]] = "scroll_id"
    df = df.rename(columns=rename)
    if "scroll_id" not in df.columns:
        df["scroll_id"] = "unknown"
    df["id"] = df["id"].astype(str)
    df["scroll_id"] = df["scroll_id"].astype(str)
    return df[["id", "scroll_id"]].drop_duplicates(subset=["id"]).reset_index(drop=True)


def list_volume_files(directory: str | Path) -> dict[str, Path]:
    """Map volume id (stem) -> path for TIFFs in a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    out: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            out[path.stem] = path
    return out


def build_volume_index(
    root: str | Path,
    split: str = "train",
    volume_ids: Optional[Iterable[str]] = None,
    scroll_ids: Optional[Iterable[str]] = None,
    require_label: Optional[bool] = None,
) -> list[VolumeRecord]:
    """Build a list of :class:`VolumeRecord` from CSV + filesystem.

    Args:
        root: Dataset root containing CSV and image/label folders.
        split: ``train`` / ``val`` / ``test``. ``val`` uses train folders
            filtered by ``scroll_ids`` or ``volume_ids`` when provided.
        volume_ids: Optional whitelist of volume ids.
        scroll_ids: Optional whitelist of scroll ids.
        require_label: Defaults to True for train/val, False for test.
    """
    root = Path(root)
    if require_label is None:
        require_label = split.lower() not in {"test", "predict", "inference"}

    meta = read_metadata_csv(root, split="test" if split.lower() == "test" else "train")
    if volume_ids is not None:
        allow = set(str(v) for v in volume_ids)
        meta = meta[meta["id"].isin(allow)]
    if scroll_ids is not None:
        allow_s = set(str(s) for s in scroll_ids)
        meta = meta[meta["scroll_id"].isin(allow_s)]

    img_dir_name = images_dirname("test" if split.lower() == "test" else "train")
    lab_dir_name = labels_dirname("test" if split.lower() == "test" else "train")

    deprecated_images = list_volume_files(root / f"deprecated_{TRAIN_IMAGES_DIRNAME}")
    deprecated_labels = list_volume_files(root / f"deprecated_{TRAIN_LABELS_DIRNAME}")

    records: list[VolumeRecord] = []
    n_deprecated = 0
    n_missing_image = 0
    n_missing_label = 0
    for row in meta.itertuples(index=False):
        vid = str(row.id)
        sid = str(row.scroll_id)
        img = volume_path(root, img_dir_name, vid)
        lab: Optional[Path] = None
        if lab_dir_name is not None:
            lab = label_path(root, vid)
        if not img.exists():
            if vid in deprecated_images:
                n_deprecated += 1
                logger.debug(
                    "Skipping deprecated id=%s (under deprecated_%s/)",
                    vid,
                    TRAIN_IMAGES_DIRNAME,
                )
            else:
                n_missing_image += 1
                logger.warning("Missing image for id=%s expected %s", vid, img)
            continue
        if require_label and (lab is None or not lab.exists()):
            if vid in deprecated_labels:
                n_deprecated += 1
                logger.debug(
                    "Skipping deprecated id=%s (under deprecated_%s/)",
                    vid,
                    TRAIN_LABELS_DIRNAME,
                )
            else:
                n_missing_label += 1
                logger.warning("Missing label for id=%s expected %s", vid, lab)
            continue
        records.append(
            VolumeRecord(
                volume_id=vid,
                scroll_id=sid,
                image_path=img,
                label_path=lab if lab is not None and lab.exists() else None,
                split=split,
            )
        )

    if not records:
        raise FileNotFoundError(
            f"No volumes indexed under {root} (split={split}). "
            "Expected train.csv + train_images/*.tif (+ train_labels for train)."
        )
    if n_deprecated:
        logger.info(
            "Skipped %d deprecated train.csv id(s) not present under %s/",
            n_deprecated,
            img_dir_name,
        )
    if n_missing_image or n_missing_label:
        logger.warning(
            "Skipped %d missing image(s) and %d missing label(s) for split=%s",
            n_missing_image,
            n_missing_label,
            split,
        )
    logger.info("Indexed %d volume(s) for split=%s under %s", len(records), split, root)
    return records


def scroll_id_groups(records: Sequence[VolumeRecord]) -> dict[str, list[str]]:
    """Map scroll_id -> list of volume ids."""
    groups: dict[str, list[str]] = {}
    for rec in records:
        groups.setdefault(rec.scroll_id, []).append(rec.volume_id)
    return groups
