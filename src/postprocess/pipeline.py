"""Directory / staged runners for post-processing ablations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from data.io import load_volume
from postprocess.first_place import (
    FIRST_PLACE_STAGES,
    PostprocessConfig,
    apply_first_place,
)

logger = logging.getLogger(__name__)

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None


def _write_tif(path: Path, array: np.ndarray) -> None:
    if tifffile is None:
        raise ImportError("tifffile is required to write post-processed volumes")
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), array.astype(np.uint8))


def _prediction_paths(predictions_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in predictions_dir.iterdir()
        if p.suffix.lower() in (".tif", ".tiff") and not p.name.endswith("_probs.tif")
    )


def run_staged(
    volume: np.ndarray,
    config: Optional[PostprocessConfig] = None,
    stages: Optional[Sequence[str]] = None,
) -> dict[str, np.ndarray]:
    """Return ``{stage_name: mask}`` for each cumulative stage."""
    cfg = config or PostprocessConfig()
    wanted = list(stages) if stages is not None else list(FIRST_PLACE_STAGES)
    return {stage: apply_first_place(volume, cfg, through_stage=stage) for stage in wanted}


def run_directory(
    predictions_dir: str | Path,
    output_dir: str | Path,
    *,
    config: Optional[PostprocessConfig] = None,
    through_stage: str = "fill",
    stages: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> list[Path]:
    """Post-process every prediction TIF.

    If ``stages`` is set, write one subfolder per cumulative stage (ablation).
    Otherwise write a single folder for ``through_stage``.
    """
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    cfg = config or PostprocessConfig()
    paths = _prediction_paths(predictions_dir)
    if limit is not None:
        paths = paths[:limit]

    stage_list = list(stages) if stages is not None else [through_stage]
    for stage in stage_list:
        if stage not in FIRST_PLACE_STAGES:
            raise ValueError(f"Unknown stage {stage!r}; choose from {FIRST_PLACE_STAGES}")

    written: list[Path] = []
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(paths, desc="postprocess", unit="vol")
    except ImportError:
        iterator = paths

    meta = {
        "config": cfg.to_dict(),
        "stages": stage_list,
        "n_inputs": len(paths),
        "source": str(predictions_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "postprocess_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    for path in iterator:
        volume = load_volume(path)
        case_id = path.stem
        for stage in stage_list:
            out = (
                output_dir / stage / f"{case_id}.tif"
                if len(stage_list) > 1
                else output_dir / f"{case_id}.tif"
            )
            if out.exists() and not overwrite:
                written.append(out)
                continue
            processed = apply_first_place(volume, cfg, through_stage=stage)
            _write_tif(out, processed)
            written.append(out)

    logger.info("Wrote %d volume(s) under %s", len(written), output_dir)
    return written
