"""Per-case scoring with resumable progress and per-scroll aggregation.

Scoring is slow (the competition warns a full run can take hours), so each
result is appended to a JSONL file as it completes and finished cases are
skipped on restart.

Labels are passed to the metric untouched — the package handles the ignore
class itself via `ignore_label=2`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from data.io import load_volume
from data.schema import LABEL_SURFACE
from evaluation.metric_adapter import score_pair

logger = logging.getLogger(__name__)


@dataclass
class CaseScore:
    case_id: str
    scroll_id: Optional[str]
    score: Optional[float]
    topo_score: Optional[float]
    surface_dice: Optional[float]
    voi_score: Optional[float]
    voi_split: Optional[float]
    voi_merge: Optional[float]
    topo_f1_dim0: Optional[float]
    topo_f1_dim1: Optional[float]
    topo_f1_dim2: Optional[float]
    n_foreground: Optional[float]
    seconds: float


_MEAN_FIELDS: tuple[str, ...] = (
    "score",
    "topo_score",
    "surface_dice",
    "voi_score",
    "voi_split",
    "voi_merge",
    "topo_f1_dim0",
    "topo_f1_dim1",
    "topo_f1_dim2",
)


def _load_done(results_path: Path) -> dict[str, CaseScore]:
    if not results_path.exists():
        return {}
    known = {f.name for f in fields(CaseScore)}
    done: dict[str, CaseScore] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            done[str(payload["case_id"])] = CaseScore(
                **{k: v for k, v in payload.items() if k in known}
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Skipping malformed result line in %s", results_path)
    return done


def _find_volume(directory: Path, case_id: str) -> Optional[Path]:
    for ext in (".tif", ".tiff"):
        candidate = directory / f"{case_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def discover_cases(predictions_dir: Path) -> list[str]:
    """Case ids with a prediction file, ignoring nnU-Net's sidecar outputs."""
    return sorted(
        path.stem
        for path in predictions_dir.iterdir()
        if path.suffix.lower() in (".tif", ".tiff")
    )


def evaluate_directory(
    predictions_dir: str | Path,
    labels_dir: str | Path,
    results_path: str | Path,
    *,
    case_ids: Optional[Iterable[str]] = None,
    scroll_map: Optional[dict[str, str]] = None,
    binarize_prediction: bool = False,
    metric_overrides: Optional[dict[str, Any]] = None,
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
            pred_path = _find_volume(predictions_dir, case_id)
            label_path = _find_volume(labels_dir, case_id)
            if pred_path is None or label_path is None:
                logger.warning("Missing prediction or label for %s, skipping", case_id)
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

            if binarize_prediction:
                prediction = (prediction == LABEL_SURFACE).astype(np.uint8)

            try:
                parts = score_pair(prediction, label, **(metric_overrides or {}))
            except Exception:
                logger.exception("Scoring failed for %s, skipping", case_id)
                continue

            score = CaseScore(
                case_id=case_id,
                scroll_id=(scroll_map or {}).get(case_id),
                seconds=time.perf_counter() - started,
                **{k: parts.get(k) for k in _MEAN_FIELDS + ("n_foreground",)},
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
        row: dict[str, float] = {"n": float(len(items))}
        for field in _MEAN_FIELDS:
            values = [
                getattr(item, field) for item in items if getattr(item, field) is not None
            ]
            row[field] = float(np.mean(values)) if values else float("nan")
        return row

    out = {scroll: summarise(items) for scroll, items in sorted(buckets.items())}
    if scores:
        out["ALL"] = summarise(scores)
    return out
