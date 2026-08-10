"""Label and path schema for Vesuvius Surface Detection."""

from __future__ import annotations

from pathlib import Path
from typing import Final

# Official label encoding (Kaggle competition data description).
LABEL_BG: Final[int] = 0
LABEL_SURFACE: Final[int] = 1
LABEL_IGNORE: Final[int] = 2

VALID_LABELS: frozenset[int] = frozenset({LABEL_BG, LABEL_SURFACE, LABEL_IGNORE})

LABEL_NAMES: dict[int, str] = {
    LABEL_BG: "background",
    LABEL_SURFACE: "surface",
    LABEL_IGNORE: "ignore",
}

TRAIN_IMAGES_DIRNAME: Final[str] = "train_images"
TRAIN_LABELS_DIRNAME: Final[str] = "train_labels"
TEST_IMAGES_DIRNAME: Final[str] = "test_images"
TRAIN_CSV_NAME: Final[str] = "train.csv"
TEST_CSV_NAME: Final[str] = "test.csv"

IMAGE_EXTENSIONS: tuple[str, ...] = (".tif", ".tiff")


def images_dirname(split: str) -> str:
    """Return the image folder name for ``train`` / ``val`` / ``test``."""
    split_l = split.lower()
    if split_l in {"train", "val", "validation"}:
        return TRAIN_IMAGES_DIRNAME
    if split_l in {"test", "predict", "inference"}:
        return TEST_IMAGES_DIRNAME
    raise ValueError(f"Unknown split: {split}")


def labels_dirname(split: str) -> str | None:
    """Return label folder name, or ``None`` when labels are unavailable."""
    split_l = split.lower()
    if split_l in {"train", "val", "validation"}:
        return TRAIN_LABELS_DIRNAME
    return None


def csv_name(split: str) -> str:
    """Return metadata CSV filename for the split."""
    split_l = split.lower()
    if split_l in {"train", "val", "validation"}:
        return TRAIN_CSV_NAME
    if split_l in {"test", "predict", "inference"}:
        return TEST_CSV_NAME
    raise ValueError(f"Unknown split: {split}")


def volume_path(root: str | Path, images_dir: str, volume_id: str) -> Path:
    """Resolve ``root/images_dir/<id>.tif`` (tries ``.tiff`` as fallback)."""
    root = Path(root)
    for ext in IMAGE_EXTENSIONS:
        candidate = root / images_dir / f"{volume_id}{ext}"
        if candidate.exists():
            return candidate
    return root / images_dir / f"{volume_id}.tif"


def label_path(root: str | Path, volume_id: str) -> Path:
    """Resolve ``root/train_labels/<id>.tif``."""
    return volume_path(root, TRAIN_LABELS_DIRNAME, volume_id)
