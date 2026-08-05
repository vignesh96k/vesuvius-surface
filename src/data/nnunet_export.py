"""Export Vesuvius Surface Detection volumes to nnU-Net v2 raw layout.

Layout produced::

    <out>/DatasetXXX_Name/
      dataset.json
      imagesTr/<case_id>_0000.tif
      labelsTr/<case_id>.tif
      imagesTs/<case_id>_0000.tif   # optional
      scroll_groups.json            # case_id -> scroll_id (for custom splits)

Images/labels are symlinked by default (no 27GB copy). Use ``mode="copy"``
only when the consumer cannot follow symlinks.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

from data.io import build_volume_index, read_metadata_csv
from data.schema import LABEL_BG, LABEL_IGNORE, LABEL_SURFACE, TEST_IMAGES_DIRNAME

logger = logging.getLogger(__name__)

LinkMode = Literal["symlink", "hardlink", "copy"]


@dataclass(frozen=True)
class NnUNetExportResult:
    dataset_dir: Path
    n_train: int
    n_test: int
    dataset_json: Path
    scroll_groups_json: Path


def case_id_from_volume_id(volume_id: str) -> str:
    """nnU-Net case identifier (filesystem-safe)."""
    return str(volume_id).strip()


def image_tr_name(case_id: str) -> str:
    return f"{case_id}_0000.tif"


def label_tr_name(case_id: str) -> str:
    return f"{case_id}.tif"


def _link_or_copy(src: Path, dst: Path, mode: LinkMode) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    src = src.resolve()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    # symlink (relative when possible so the dataset folder is movable within same FS)
    try:
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    except OSError:
        os.symlink(src, dst)


def build_dataset_json(
    *,
    name: str,
    train_case_ids: Sequence[str],
    file_ending: str = ".tif",
    channel_name: str = "CT",
) -> dict:
    """nnU-Net v2 dataset.json with ignore label ``2`` named ``ignore``."""
    return {
        "name": name,
        "description": "Vesuvius Challenge — Surface Detection (3D micro-CT)",
        "reference": "https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection",
        "licence": "Competition Data",
        "release": "1.0",
        "channel_names": {"0": channel_name},
        "labels": {
            "background": LABEL_BG,
            "surface": LABEL_SURFACE,
            "ignore": LABEL_IGNORE,
        },
        "numTraining": len(train_case_ids),
        "file_ending": file_ending,
        # Helps nnU-Net pick a TIFF-capable reader when available.
        "overwrite_image_reader_writer": "Tiff3DIO",
    }


def export_nnunet_dataset(
    data_root: str | Path,
    output_root: str | Path,
    *,
    dataset_id: int = 100,
    dataset_name: str = "VesuviusSurface",
    mode: LinkMode = "symlink",
    include_test: bool = True,
    train_volume_ids: Optional[Iterable[str]] = None,
    train_scroll_ids: Optional[Iterable[str]] = None,
    max_train_volumes: Optional[int] = None,
) -> NnUNetExportResult:
    """Create an nnU-Net raw dataset directory from the Kaggle extract.

    Args:
        data_root: Kaggle dataset root (``train_images``, ``train_labels``, …).
        output_root: Parent folder (typically ``$nnUNet_raw``).
        dataset_id: Numeric dataset id (``Dataset100_…``).
        dataset_name: Suffix name without spaces.
        mode: ``symlink`` (default), ``hardlink``, or ``copy``.
        include_test: Also link ``test_images`` into ``imagesTs/``.
        train_volume_ids / train_scroll_ids: Optional filters.
        max_train_volumes: Optional cap for dry-runs / smoke tests.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    folder = f"Dataset{int(dataset_id):03d}_{dataset_name}"
    dataset_dir = output_root / folder
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    records = build_volume_index(
        data_root,
        split="train",
        volume_ids=train_volume_ids,
        scroll_ids=train_scroll_ids,
        require_label=True,
    )
    if max_train_volumes is not None:
        records = records[: max(0, int(max_train_volumes))]

    scroll_groups: dict[str, str] = {}
    train_case_ids: list[str] = []

    for rec in records:
        case_id = case_id_from_volume_id(rec.volume_id)
        assert rec.label_path is not None
        _link_or_copy(rec.image_path, images_tr / image_tr_name(case_id), mode)
        _link_or_copy(rec.label_path, labels_tr / label_tr_name(case_id), mode)
        train_case_ids.append(case_id)
        scroll_groups[case_id] = rec.scroll_id

    n_test = 0
    if include_test:
        images_ts.mkdir(parents=True, exist_ok=True)
        test_dir = data_root / TEST_IMAGES_DIRNAME
        if test_dir.is_dir():
            # Prefer test.csv when present; otherwise all TIFFs in test_images/.
            try:
                test_meta = read_metadata_csv(data_root, split="test")
                test_ids = [str(x) for x in test_meta["id"].tolist()]
            except FileNotFoundError:
                test_ids = sorted(
                    {p.stem for p in test_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"}}
                )
            for vid in test_ids:
                src = None
                for ext in (".tif", ".tiff"):
                    cand = test_dir / f"{vid}{ext}"
                    if cand.exists():
                        src = cand
                        break
                if src is None:
                    logger.warning("Test image missing for id=%s under %s", vid, test_dir)
                    continue
                case_id = case_id_from_volume_id(vid)
                _link_or_copy(src, images_ts / image_tr_name(case_id), mode)
                n_test += 1

    dataset_payload = build_dataset_json(
        name=dataset_name,
        train_case_ids=train_case_ids,
    )
    dataset_json_path = dataset_dir / "dataset.json"
    dataset_json_path.write_text(json.dumps(dataset_payload, indent=2) + "\n", encoding="utf-8")

    scroll_path = dataset_dir / "scroll_groups.json"
    scroll_path.write_text(json.dumps(scroll_groups, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Manifest for debugging / resume
    manifest = {
        "data_root": str(data_root.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "mode": mode,
        "n_train": len(train_case_ids),
        "n_test": n_test,
        "train_case_ids": train_case_ids,
    }
    (dataset_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    logger.info(
        "nnU-Net export ready: %s (train=%d test=%d mode=%s)",
        dataset_dir,
        len(train_case_ids),
        n_test,
        mode,
    )
    return NnUNetExportResult(
        dataset_dir=dataset_dir,
        n_train=len(train_case_ids),
        n_test=n_test,
        dataset_json=dataset_json_path,
        scroll_groups_json=scroll_path,
    )


def write_scroll_holdout_split(
    scroll_groups_json: str | Path,
    output_json: str | Path,
    val_scroll_ids: Sequence[str],
) -> Path:
    """Write a 1-fold nnU-Net ``splits_final.json`` from scroll holdout.

    Place the result at::

        $nnUNet_preprocessed/DatasetXXX_Name/splits_final.json

    before training (after planning), or use nnU-Net's custom split hooks.
    """
    groups = json.loads(Path(scroll_groups_json).read_text(encoding="utf-8"))
    val_set = {str(s) for s in val_scroll_ids}
    train_ids = sorted(cid for cid, sid in groups.items() if sid not in val_set)
    val_ids = sorted(cid for cid, sid in groups.items() if sid in val_set)
    if not val_ids:
        raise ValueError(f"No cases found for val_scroll_ids={sorted(val_set)}")
    if not train_ids:
        raise ValueError("Train set empty after scroll holdout")
    payload = [{"train": train_ids, "val": val_ids}]
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Wrote scroll holdout split: train=%d val=%d -> %s",
        len(train_ids),
        len(val_ids),
        out,
    )
    return out
