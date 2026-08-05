"""Per-case scoring with resumable progress and per-scroll aggregation.

Scoring is slow (the competition warns a full run can take hours), so results
are appended to a JSONL file as each case finishes and completed cases are
skipped on restart.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from data.io import load_volume
from data.schema import LABEL_IGNORE, LABEL_SURFACE
from evaluation.metric_adapter import METRIC_WEIGHTS, score_pair

logger = logging.getLogger(__name__)

IgnoreMode = str  # "neutralize" | "background"


@dataclass
class CaseScore:
    case_id: str
    scroll_id: Optional[str]
    topo_score: float
    surface_dice: float
    voi_score: float
    composite: float
    seconds: float


def composite_score(parts: dict[str, float]) -> float:
    return float(sum(METRIC_WEIGHTS[k] * float(parts[k]) for k in METRIC_WEIGHTS))


def apply_ignore(
    prediction: np.ndarray,
    label: np.ndarray,
    mode: IgnoreMode = "neutralize",
) -> tuple[np.ndarray, np.ndarray]:
    """Binarise label and reconcile the ignore class (2).

    Training labels mark unlabelled voxels as 2, which the schema says must not
    be scored. There is no way to remove voxels from a 3D connected-component
    or topology computation, so we pick between two approximations:

    - ``neutralize``: force the prediction to match ground truth inside ignore
      regions, so they contribute no disagreement. Can still perturb topology
      by joining or splitting components across the region.
    - ``background``: treat ignore as background and leave the prediction
      alone, penalising anything predicted there.

    Neither is exactly what the hidden test set does, since its ground truth is
    presumably fully labelled. Report which mode was used.
    """
    gt = label == LABEL_SURFACE
    pred = prediction.astype(bool)

    if mode == "background":
        return pred, gt
    if mode != "neutralize":
        raise ValueError(f"Unknown ignore mode: {mode!r}")

    ignore = label == LABEL_IGNORE
    if ignore.any():
        pred = pred.copy()
        pred[ignore] = gt[ignore]
    return pred, gt


def _load_done(results_path: Path) -> dict[str, CaseScore]:
    if not results_path.exists():
        return {}
    done: dict[str, CaseScore] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            done[str(payload["case_id"])] = CaseScore(**payload)
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Skipping malformed result line in %s", results_path)
    return done


def _find_prediction(predictions_dir: Path, case_id: str) -> Optional[Path]:
    for ext in (".tif", ".tiff"):
        candidate = predictions_dir / f"{case_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def discover_cases(predictions_dir: Path) -> list[str]:
    """Case ids with a prediction file, ignoring nnU-Net's sidecar outputs."""
    cases: set[str] = set()
    for path in predictions_dir.iterdir():
        if path.suffix.lower() not in (".tif", ".tiff"):
            continue
        cases.add(path.stem)
    return sorted(cases)


def evaluate_directory(
    predictions_dir: str | Path,
    labels_dir: str | Path,
    results_path: str | Path,
    *,
    case_ids: Optional[Iterable[str]] = None,
    scroll_map: Optional[dict[str, str]] = None,
    ignore_mode: IgnoreMode = "neutralize",
    resume: bool = True,
    progress: bool = True,
) -> list[CaseScore]:
    """Score every case, appending each result as soon as it completes."""
    predictions_dir = Path(predictions_dir)
    labels_dir = Path(labels_dir)
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    cases = list(case_ids) if case_ids is not None else discover_cases(predictions_dir)
    done = _load_done(results_path) if resume else {}
    if done:
        logger.info("Resuming: %d case(s) already scored", len(done))

    pending = [c for c in cases if c not in done]
    logger.info("Scoring %d case(s) (%d already done)", len(pending), len(done))

    iterator: Iterable[str] = pending
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(pending, desc="score", unit="vol")
        except ImportError:
            pass

    scores: list[CaseScore] = list(done.values())

    with results_path.open("a", encoding="utf-8") as handle:
        for case_id in iterator:
            pred_path = _find_prediction(predictions_dir, case_id)
            if pred_path is None:
                logger.warning("No prediction for %s, skipping", case_id)
                continue
            label_path = _find_prediction(labels_dir, case_id)
            if label_path is None:
                logger.warning("No label for %s, skipping", case_id)
                continue

            started = time.perf_counter()
            prediction = load_volume(pred_path)
            label = load_volume(label_path)
            if prediction.shape != label.shape:
                logger.warning(
                    "Shape mismatch for %s: pred %s vs label %s, skipping",
                    case_id,
                    prediction.shape,
                    label.shape,
                )
                continue

            pred_bin, gt_bin = apply_ignore(prediction, label, mode=ignore_mode)
            parts = score_pair(pred_bin, gt_bin)
            elapsed = time.perf_counter() - started

            score = CaseScore(
                case_id=case_id,
                scroll_id=(scroll_map or {}).get(case_id),
                topo_score=float(parts["topo_score"]),
                surface_dice=float(parts["surface_dice"]),
                voi_score=float(parts["voi_score"]),
                composite=composite_score(parts),
                seconds=elapsed,
            )
            handle.write(json.dumps(asdict(score)) + "\n")
            handle.flush()
            scores.append(score)

    return scores


def aggregate_by_scroll(scores: list[CaseScore]) -> dict[str, dict[str, float]]:
    """Mean sub-scores per scroll plus an overall row."""
    buckets: dict[str, list[CaseScore]] = {}
    for score in scores:
        buckets.setdefault(score.scroll_id or "unknown", []).append(score)

    def summarise(items: list[CaseScore]) -> dict[str, float]:
        return {
            "n": float(len(items)),
            "topo_score": float(np.mean([s.topo_score for s in items])),
            "surface_dice": float(np.mean([s.surface_dice for s in items])),
            "voi_score": float(np.mean([s.voi_score for s in items])),
            "composite": float(np.mean([s.composite for s in items])),
        }

    out = {scroll: summarise(items) for scroll, items in sorted(buckets.items())}
    if scores:
        out["ALL"] = summarise(scores)
    return out
