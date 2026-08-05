"""Dataset validation for Surface Detection volumes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from data.io import list_volume_files, load_volume, read_metadata_csv
from data.schema import (
    LABEL_IGNORE,
    LABEL_NAMES,
    LABEL_SURFACE,
    TRAIN_IMAGES_DIRNAME,
    TRAIN_LABELS_DIRNAME,
    TEST_IMAGES_DIRNAME,
    VALID_LABELS,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Single validation finding."""

    severity: str  # error | warning | info
    volume_id: str
    message: str


@dataclass
class ValidationReport:
    """Aggregate validation results."""

    root: Path
    split: str
    n_csv_rows: int = 0
    n_volumes_ok: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    inventory: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def ok(self) -> bool:
        return self.n_errors == 0

    def summary(self) -> str:
        lines = [
            f"ValidationReport root={self.root} split={self.split}",
            f"  csv_rows={self.n_csv_rows} volumes_ok={self.n_volumes_ok}",
            f"  errors={self.n_errors} warnings={self.n_warnings}",
        ]
        for issue in self.issues[:50]:
            lines.append(f"  [{issue.severity}] {issue.volume_id}: {issue.message}")
        if len(self.issues) > 50:
            lines.append(f"  … {len(self.issues) - 50} more")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["severity", "volume_id", "message"])
        return pd.DataFrame(
            [{"severity": i.severity, "volume_id": i.volume_id, "message": i.message} for i in self.issues]
        )


def validate_dataset(
    root: str | Path,
    split: str = "train",
    max_volumes_to_scan: Optional[int] = None,
    check_label_values: bool = True,
) -> ValidationReport:
    """Validate CSV / image / label consistency for Surface Detection.

    Hard errors: missing train label, shape mismatch, illegal label values.
    Warnings: orphan files, empty volumes, extreme sparsity.
    """
    root = Path(root)
    report = ValidationReport(root=root, split=split)
    require_label = split.lower() not in {"test", "predict", "inference"}

    try:
        meta = read_metadata_csv(root, split="test" if split.lower() == "test" else "train")
    except FileNotFoundError as exc:
        report.issues.append(ValidationIssue("error", "*", str(exc)))
        return report

    report.n_csv_rows = len(meta)

    img_dir = root / (TEST_IMAGES_DIRNAME if split.lower() == "test" else TRAIN_IMAGES_DIRNAME)
    lab_dir = root / TRAIN_LABELS_DIRNAME
    disk_images = list_volume_files(img_dir)
    disk_labels = list_volume_files(lab_dir) if require_label else {}

    csv_ids = set(meta["id"].astype(str))
    for orphan in sorted(set(disk_images) - csv_ids):
        report.issues.append(
            ValidationIssue("warning", orphan, f"Image on disk not listed in CSV: {disk_images[orphan]}")
        )
    if require_label:
        for orphan in sorted(set(disk_labels) - csv_ids):
            report.issues.append(
                ValidationIssue("warning", orphan, f"Label on disk not listed in CSV: {disk_labels[orphan]}")
            )

    rows: list[dict[str, Any]] = []
    scanned = 0
    for row in meta.itertuples(index=False):
        if max_volumes_to_scan is not None and scanned >= max_volumes_to_scan:
            break
        vid = str(row.id)
        sid = str(row.scroll_id)
        scanned += 1

        img_p = disk_images.get(vid)
        if img_p is None:
            report.issues.append(ValidationIssue("error", vid, f"Missing image under {img_dir}"))
            continue

        lab_p = disk_labels.get(vid) if require_label else None
        if require_label and lab_p is None:
            report.issues.append(ValidationIssue("error", vid, f"Missing label under {lab_dir}"))
            continue

        try:
            image = load_volume(img_p)
        except Exception as exc:  # noqa: BLE001
            report.issues.append(ValidationIssue("error", vid, f"Failed to read image: {exc}"))
            continue

        label = None
        if lab_p is not None:
            try:
                label = load_volume(lab_p)
            except Exception as exc:  # noqa: BLE001
                report.issues.append(ValidationIssue("error", vid, f"Failed to read label: {exc}"))
                continue
            if label.shape != image.shape:
                report.issues.append(
                    ValidationIssue(
                        "error",
                        vid,
                        f"Shape mismatch image{image.shape} vs label{label.shape}",
                    )
                )
                continue
            if check_label_values:
                uniq = set(int(x) for x in np.unique(label))
                illegal = uniq - set(VALID_LABELS)
                if illegal:
                    report.issues.append(
                        ValidationIssue("error", vid, f"Illegal label values: {sorted(illegal)}")
                    )
                    continue

        if image.size == 0:
            report.issues.append(ValidationIssue("error", vid, "Empty volume"))
            continue

        entry: dict[str, Any] = {
            "volume_id": vid,
            "scroll_id": sid,
            "shape_d": int(image.shape[0]),
            "shape_h": int(image.shape[1]),
            "shape_w": int(image.shape[2]),
            "dtype": str(image.dtype),
            "image_bytes": int(img_p.stat().st_size),
        }
        if label is not None:
            flat = label.ravel()
            n = flat.size
            for val, name in LABEL_NAMES.items():
                entry[f"frac_{name}"] = float((flat == val).sum()) / n
            labeled = flat != LABEL_IGNORE
            n_lab = int(labeled.sum())
            entry["frac_labeled"] = n_lab / n
            entry["frac_surface_among_labeled"] = (
                float((flat[labeled] == LABEL_SURFACE).sum()) / n_lab if n_lab else 0.0
            )
            if entry["frac_labeled"] < 0.01:
                report.issues.append(
                    ValidationIssue("warning", vid, "Very few labeled voxels (<1%)")
                )
        rows.append(entry)
        report.n_volumes_ok += 1

    report.inventory = pd.DataFrame(rows)
    logger.info("Validation finished: %s", report.summary().splitlines()[0:3])
    return report
