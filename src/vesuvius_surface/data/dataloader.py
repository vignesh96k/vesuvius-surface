"""DataLoader factory for Surface Detection datasets."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _seed_worker(worker_id: int) -> None:
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    _ = worker_id


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = 2,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> DataLoader:
    """Create a PyTorch ``DataLoader`` with research defaults."""
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    generator = None
    worker_init_fn = kwargs.pop("worker_init_fn", None)
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        if worker_init_fn is None:
            worker_init_fn = _seed_worker

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "worker_init_fn": worker_init_fn,
        "generator": generator,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader_kwargs.update(kwargs)
    loader = DataLoader(**loader_kwargs)
    logger.info(
        "DataLoader ready: batch_size=%s shuffle=%s workers=%s len=%s",
        batch_size,
        shuffle,
        num_workers,
        len(dataset),  # type: ignore[arg-type]
    )
    return loader


def build_dataloaders_from_config(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Build train/val loaders from merged experiment config."""
    from vesuvius_surface.data.dataset import SurfacePatchDataset
    from vesuvius_surface.data.patching import PatchConfig3D
    from vesuvius_surface.data.transforms import build_transforms

    data_cfg = config.get("data", {})
    patch_size = data_cfg.get("patch_size", [128, 128, 128])
    stride = data_cfg.get("stride", [64, 64, 64])
    if isinstance(patch_size, int):
        patch_size = [patch_size] * 3
    if isinstance(stride, int):
        stride = [stride] * 3

    patch_cfg = PatchConfig3D(
        patch_size=tuple(patch_size),  # type: ignore[arg-type]
        stride=tuple(stride),  # type: ignore[arg-type]
        pad_value=float(data_cfg.get("pad_value", 0.0)),
        min_labeled_ratio=float(data_cfg.get("min_labeled_ratio", 0.1)),
        min_foreground_ratio=float(data_cfg.get("min_foreground_ratio", 0.0)),
        ignore_index=int(data_cfg.get("ignore_index", 2)),
    )

    root = data_cfg.get("root", "data")
    seed = config.get("experiment", {}).get("seed", None)
    normalize = str(data_cfg.get("normalize", "zscore"))
    normalize_kwargs = dict(data_cfg.get("normalize_kwargs", {}))
    transform_cfg = data_cfg.get("transforms", {})

    loaders: dict[str, DataLoader] = {}
    val_scrolls = data_cfg.get("val_scroll_ids")
    train_scrolls = data_cfg.get("train_scroll_ids")
    # If val scrolls are set but train scrolls are not, train = all except val.
    if train_scrolls is None and val_scrolls:
        from vesuvius_surface.data.io import build_volume_index

        all_recs = build_volume_index(root, split="train")
        all_scrolls = sorted({r.scroll_id for r in all_recs})
        exclude = set(str(s) for s in val_scrolls)
        train_scrolls = [s for s in all_scrolls if s not in exclude]

    train_ds = SurfacePatchDataset(
        root=root,
        split="train",
        patch_config=patch_cfg,
        scroll_ids=train_scrolls,
        volume_ids=data_cfg.get("train_volume_ids"),
        transform=build_transforms("train", transform_cfg),
        normalize=normalize,
        normalize_kwargs=normalize_kwargs,
    )
    loaders["train"] = build_dataloader(
        train_ds,
        batch_size=int(config.get("train", {}).get("batch_size", data_cfg.get("batch_size", 2))),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=True,
        seed=seed,
    )

    if val_scrolls:
        val_ds = SurfacePatchDataset(
            root=root,
            split="val",
            patch_config=patch_cfg,
            scroll_ids=val_scrolls,
            volume_ids=data_cfg.get("val_volume_ids"),
            transform=build_transforms("val", transform_cfg),
            normalize=normalize,
            normalize_kwargs=normalize_kwargs,
        )
        loaders["val"] = build_dataloader(
            val_ds,
            batch_size=int(config.get("train", {}).get("batch_size", data_cfg.get("batch_size", 2))),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 4)),
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            drop_last=False,
            seed=seed,
        )
    else:
        logger.warning(
            "No data.val_scroll_ids set — skipping val loader. "
            "Prefer scroll-level holdout for Surface Detection."
        )

    return loaders
