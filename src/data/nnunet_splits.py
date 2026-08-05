"""Reconstruct nnU-Net's deterministic cross-validation splits.

nnU-Net creates ``splits_final.json`` via::

    all_keys_sorted = sorted(dataset.identifiers)
    generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)

Because the seed and the sort order are fixed, the fold assignment is
reproducible from the case-id list alone. That lets us recover which cases a
published ``fold_N`` checkpoint never trained on, so it can be evaluated
without leakage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_SEED = 12345
DEFAULT_N_SPLITS = 5

TIFF_SUFFIXES = {".tif", ".tiff"}
Split = dict[str, list[str]]


@dataclass(frozen=True)
class FoldReport:
    fold: int
    n_train: int
    n_val: int
    val_case_ids: list[str]
    source: str
    scroll_counts: dict[str, int]


def list_case_ids(dataset_dir: str | Path) -> list[str]:
    """Sorted case ids from an nnU-Net raw dataset directory."""
    dataset_dir = Path(dataset_dir)
    labels_dir = dataset_dir / "labelsTr"
    images_dir = dataset_dir / "imagesTr"

    ids: set[str] = set()
    if labels_dir.is_dir():
        ids = {p.stem for p in labels_dir.iterdir() if p.suffix.lower() in TIFF_SUFFIXES}
    elif images_dir.is_dir():
        for path in images_dir.iterdir():
            if path.suffix.lower() not in TIFF_SUFFIXES:
                continue
            stem = path.stem
            if "_" in stem and stem.rsplit("_", 1)[-1].isdigit():
                stem = stem.rsplit("_", 1)[0]
            ids.add(stem)
    else:
        raise FileNotFoundError(f"No labelsTr/ or imagesTr/ under {dataset_dir}")

    if not ids:
        raise FileNotFoundError(f"No TIFF cases found under {dataset_dir}")
    return sorted(ids)


def generate_default_splits(
    case_ids: Sequence[str],
    seed: int = DEFAULT_SEED,
    n_splits: int = DEFAULT_N_SPLITS,
) -> list[Split]:
    """Replicate nnU-Net's default K-fold assignment.

    Uses nnU-Net's own helper when installed so the result cannot drift from
    the framework; otherwise falls back to the equivalent scikit-learn call.
    """
    keys = sorted(str(c) for c in case_ids)

    try:
        from nnunetv2.utilities.crossval_split import generate_crossval_split
    except ImportError:
        from sklearn.model_selection import KFold

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits: list[Split] = []
        for train_idx, val_idx in kfold.split(keys):
            splits.append(
                {
                    "train": [keys[i] for i in train_idx],
                    "val": [keys[i] for i in val_idx],
                }
            )
        logger.debug("Generated splits with scikit-learn fallback")
        return splits

    raw = generate_crossval_split(keys, seed=seed, n_splits=n_splits)
    return [
        {"train": [str(x) for x in s["train"]], "val": [str(x) for x in s["val"]]}
        for s in raw
    ]


def load_splits(path: str | Path) -> list[Split]:
    """Load an existing ``splits_final.json``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a list of folds")
    return [
        {"train": [str(x) for x in fold["train"]], "val": [str(x) for x in fold["val"]]}
        for fold in payload
    ]


def resolve_splits(
    dataset_dir: str | Path,
    splits_json: Optional[str | Path] = None,
    seed: int = DEFAULT_SEED,
    n_splits: int = DEFAULT_N_SPLITS,
) -> tuple[list[Split], str]:
    """Prefer a shipped split file, else reconstruct it deterministically.

    Returns ``(splits, source)`` where ``source`` records provenance so reports
    never imply a reconstruction was an authoritative file.
    """
    if splits_json is not None:
        path = Path(splits_json)
        if path.is_file():
            logger.info("Using split file: %s", path)
            return load_splits(path), f"file:{path}"
        logger.warning("Split file not found, falling back to reconstruction: %s", path)

    case_ids = list_case_ids(dataset_dir)
    logger.info(
        "Reconstructing default %d-fold split (seed=%d) over %d cases",
        n_splits,
        seed,
        len(case_ids),
    )
    return generate_default_splits(case_ids, seed=seed, n_splits=n_splits), "reconstructed"


def load_scroll_groups(path: str | Path) -> dict[str, str]:
    """Load ``scroll_groups.json`` written by the exporter."""
    path = Path(path)
    if not path.is_file():
        return {}
    return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def scroll_distribution(
    case_ids: Sequence[str],
    scroll_groups: dict[str, str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case_id in case_ids:
        scroll = scroll_groups.get(str(case_id), "unknown")
        counts[scroll] = counts.get(scroll, 0) + 1
    return dict(sorted(counts.items()))


def describe_fold(
    dataset_dir: str | Path,
    fold: int = 0,
    splits_json: Optional[str | Path] = None,
    seed: int = DEFAULT_SEED,
    n_splits: int = DEFAULT_N_SPLITS,
) -> FoldReport:
    """Summarize one fold, including which scrolls its validation cases span."""
    dataset_dir = Path(dataset_dir)
    splits, source = resolve_splits(dataset_dir, splits_json, seed=seed, n_splits=n_splits)
    if fold >= len(splits):
        raise IndexError(f"fold {fold} requested but only {len(splits)} folds available")

    val_ids = list(splits[fold]["val"])
    scroll_groups = load_scroll_groups(dataset_dir / "scroll_groups.json")
    return FoldReport(
        fold=fold,
        n_train=len(splits[fold]["train"]),
        n_val=len(val_ids),
        val_case_ids=val_ids,
        source=source,
        scroll_counts=scroll_distribution(val_ids, scroll_groups),
    )


def write_holdout_manifest(
    report: FoldReport,
    output_json: str | Path,
    note: str = "",
) -> Path:
    """Persist the unseen-case list used to evaluate a published checkpoint."""
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fold": report.fold,
        "source": report.source,
        "n_train": report.n_train,
        "n_val": report.n_val,
        "scroll_counts": report.scroll_counts,
        "val_case_ids": report.val_case_ids,
        "note": note,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote holdout manifest (%d cases) -> %s", report.n_val, out)
    return out
