"""High-level Surface Detection dataset inspection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from eda.statistics import ClassStats, IntensityStats, compute_class_stats, compute_intensity_stats
from eda.utils import file_size_bytes, format_bytes, resolve_data_root
from data.io import VolumeRecord, build_volume_index, load_volume, scroll_id_groups
from data.validate import ValidationReport, validate_dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VolumeReport:
    volume_id: str
    scroll_id: str
    shape: tuple[int, int, int]
    dtype: str
    image_bytes: int
    label_bytes: int
    frac_bg: float
    frac_surface: float
    frac_ignore: float

    def to_dict(self) -> dict[str, Any]:
        d, h, w = self.shape
        return {
            "volume_id": self.volume_id,
            "scroll_id": self.scroll_id,
            "shape": f"{d}×{h}×{w}",
            "shape_d": d,
            "shape_h": h,
            "shape_w": w,
            "dtype": self.dtype,
            "disk_size": format_bytes(self.image_bytes + self.label_bytes),
            "image_bytes": self.image_bytes,
            "label_bytes": self.label_bytes,
            "frac_bg": self.frac_bg,
            "frac_surface": self.frac_surface,
            "frac_ignore": self.frac_ignore,
        }


class SurfaceDatasetInspector:
    """Inspect Surface Detection volumes for EDA notebooks."""

    def __init__(
        self,
        data_root: Optional[str | Path] = None,
        split: str = "train",
        volume_ids: Optional[Sequence[str]] = None,
        scroll_ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.data_root = resolve_data_root(data_root)
        self.split = split
        self.records: list[VolumeRecord] = build_volume_index(
            self.data_root,
            split=split,
            volume_ids=volume_ids,
            scroll_ids=scroll_ids,
        )
        logger.info(
            "SurfaceDatasetInspector: root=%s split=%s n=%d",
            self.data_root,
            split,
            len(self.records),
        )

    @property
    def volume_ids(self) -> list[str]:
        return [r.volume_id for r in self.records]

    @property
    def scroll_ids(self) -> list[str]:
        return sorted({r.scroll_id for r in self.records})

    def get_record(self, volume_id: str) -> VolumeRecord:
        for rec in self.records:
            if rec.volume_id == volume_id:
                return rec
        raise KeyError(volume_id)

    def load_image(self, volume_id: str) -> np.ndarray:
        return load_volume(self.get_record(volume_id).image_path)

    def load_label(self, volume_id: str) -> Optional[np.ndarray]:
        rec = self.get_record(volume_id)
        if rec.label_path is None:
            return None
        return load_volume(rec.label_path)

    def default_volume_id(self) -> str:
        return self.records[0].volume_id

    def validate(self) -> ValidationReport:
        return validate_dataset(self.data_root, split=self.split)

    def build_volume_report(self, volume_id: str) -> VolumeReport:
        rec = self.get_record(volume_id)
        image = self.load_image(volume_id)
        label = self.load_label(volume_id)
        if label is None:
            cs = ClassStats(image.size, 0, 0, 0, 0, 0)
        else:
            cs = compute_class_stats(label)
        return VolumeReport(
            volume_id=rec.volume_id,
            scroll_id=rec.scroll_id,
            shape=tuple(int(x) for x in image.shape),  # type: ignore[arg-type]
            dtype=str(image.dtype),
            image_bytes=file_size_bytes(rec.image_path),
            label_bytes=0 if rec.label_path is None else file_size_bytes(rec.label_path),
            frac_bg=cs.frac_bg,
            frac_surface=cs.frac_surface,
            frac_ignore=cs.frac_ignore,
        )

    def inventory(self, max_volumes: Optional[int] = None) -> pd.DataFrame:
        rows = []
        for i, rec in enumerate(self.records):
            if max_volumes is not None and i >= max_volumes:
                break
            rows.append(self.build_volume_report(rec.volume_id).to_dict())
        return pd.DataFrame(rows)

    def scroll_summary(self) -> pd.DataFrame:
        groups = scroll_id_groups(self.records)
        rows = []
        for sid, vids in sorted(groups.items()):
            rows.append({"scroll_id": sid, "n_volumes": len(vids), "volume_ids": ",".join(vids)})
        return pd.DataFrame(rows)

    def overview_text(self) -> str:
        inv = self.inventory()
        lines = [
            "SURFACE DETECTION DATASET OVERVIEW",
            "=" * 48,
            f"Data root   : {self.data_root}",
            f"Split       : {self.split}",
            f"Volumes     : {len(self.records)}",
            f"Scrolls     : {len(self.scroll_ids)}",
            "",
            "Per-volume:",
        ]
        for _, row in inv.iterrows():
            lines.append(
                f"  • {row['volume_id']:>16} | scroll={row['scroll_id']:<10} | "
                f"{row['shape']:>14} | surf={row['frac_surface']*100:5.2f}% | "
                f"ignore={row['frac_ignore']*100:5.2f}% | {row['disk_size']}"
            )
        return "\n".join(lines)

    def intensity_stats(self, volume_id: str) -> IntensityStats:
        return compute_intensity_stats(self.load_image(volume_id), self.load_label(volume_id))

    def class_stats(self, volume_id: str) -> ClassStats:
        label = self.load_label(volume_id)
        if label is None:
            raise ValueError(f"No label for {volume_id}")
        return compute_class_stats(label)
