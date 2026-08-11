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
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from vesuvius_surface.data.io import load_volume
from vesuvius_surface.data.schema import LABEL_SURFACE
from vesuvius_surface.evaluation.metric_adapter import score_pair

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


def _score_one_case(args: tuple) -> tuple[str, Optional[dict], Optional[str]]:
    """Picklable worker body for the ``workers > 1`` path. Returns
    ``(case_id, payload_or_None, error_or_None)`` -- never raises, so a Pool.imap_unordered
    can't be killed by one bad case; the caller decides what "skip" means (matches the
    warn-and-continue behavior of the sequential loop below)."""
    case_id, predictions_dir, labels_dir, scroll_id, binarize_prediction, metric_overrides = args
    pred_path = _find_volume(Path(predictions_dir), case_id)
    label_path = _find_volume(Path(labels_dir), case_id)
    if pred_path is None or label_path is None:
        return case_id, None, "missing prediction or label"

    started = time.perf_counter()
    prediction = load_volume(pred_path)
    label = load_volume(label_path)
    if prediction.shape != label.shape:
        return case_id, None, f"shape mismatch: pred {prediction.shape} vs label {label.shape}"

    if binarize_prediction:
        prediction = (prediction == LABEL_SURFACE).astype(np.uint8)

    try:
        parts = score_pair(prediction, label, **(metric_overrides or {}))
    except Exception as exc:  # noqa: BLE001 -- reported in the main process, not raised here
        return case_id, None, f"scoring failed: {exc!r}"

    payload = {
        "scroll_id": scroll_id,
        "seconds": time.perf_counter() - started,
        **{k: parts.get(k) for k in _MEAN_FIELDS + ("n_foreground",)},
    }
    return case_id, payload, None


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
    workers: int = 1,
) -> list[CaseScore]:
    """Score every case, appending each result as soon as it completes.

    ``workers`` > 1 parallelizes across cases (each case is independent -- its own
    volume load + metric call), via ``multiprocessing.Pool``. The metric is CPU-bound
    and expensive (~60-90s/volume, Betti-matching dominates -- see unmerge.py's module
    docstring), so this is the same shape of parallelism scripts/evaluation/score_model.py
    already proved safe. IMPORTANT, matching that script's own top-of-file warning: the
    caller must set OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS/NUMEXPR_NUM_THREADS=1
    *before numpy is imported anywhere in the process* (not just before calling this
    function) -- a forked worker inherits whatever thread pool numpy's BLAS backend already
    initialized in the parent, so setting the vars here would be too late. See
    scripts/run_postprocess.py's own top-of-file guard for where that actually has to live.
    """
    predictions_dir = Path(predictions_dir)
    labels_dir = Path(labels_dir)
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    cases = list(case_ids) if case_ids is not None else discover_cases(predictions_dir)
    done = _load_done(results_path) if resume else {}
    pending = [c for c in cases if c not in done]
    logger.info("Scoring %d case(s) (%d already done)", len(pending), len(done))

    scores: list[CaseScore] = list(done.values())

    if workers <= 1:
        iterator: Iterable[str] = pending
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(pending, desc="score", unit="vol")
            except ImportError:
                pass

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

    # workers > 1: same per-case logic, moved into _score_one_case so it can run in a
    # separate process; this process only writes the (already-serializable) results.
    units = [
        (cid, str(predictions_dir), str(labels_dir), (scroll_map or {}).get(cid),
         binarize_prediction, metric_overrides)
        for cid in pending
    ]
    iterator = None
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(total=len(units), desc="score", unit="vol")
        except ImportError:
            pass

    with results_path.open("a", encoding="utf-8") as handle, Pool(workers) as pool:
        for case_id, payload, error in pool.imap_unordered(_score_one_case, units):
            if iterator is not None:
                iterator.update(1)
            if error is not None:
                logger.warning("%s for %s, skipping", error, case_id)
                continue
            score = CaseScore(case_id=case_id, **payload)
            handle.write(json.dumps(asdict(score)) + "\n")
            handle.flush()
            scores.append(score)
    if iterator is not None:
        iterator.close()

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
