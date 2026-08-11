"""Thin adapter over the official competition metric package (`topometrics`).

Installed from the Kaggle dataset `sohier/vesuvius-metric-resources` via
`packages/vesuvius_evaluation/scripts/install_topometrics.sh` (see README.md Quickstart).
Everything that depends on its API lives here.

    Score = 0.30*TopoScore + 0.35*SurfaceDice@2.0 + 0.35*VOI_score

BUG, FIXED: this module used to call `compute_leaderboard_score` with no explicit
parameters, on the stated theory that the package's own defaults already matched the
leaderboard. That was never checked against what `scripts/evaluation/score_model.py` (and
every one-off scoring script that produced this project's actual reported numbers) explicitly
passes. They differ on exactly one parameter: the package's own default is `voi_alpha=1.0`;
every real reported number in this project was computed with `voi_alpha=0.3`.
Verified directly: scoring the same real case both ways gives score=0.5204 (alpha=1.0,
this module's old behavior) vs. score=0.6182 (alpha=0.3, matches the number already on
record for that exact case in a prior scoring run). Since `apply_unmerge` in
postprocess/unmerge.py calls this function directly for its own accept/reject gate, this
bug meant the unmerge novelty layer's accept/reject decisions were made against a metric
that doesn't match what the project reports everywhere else -- not just a scoring-time
issue. `DEFAULT_METRIC_KWARGS` below now pins every parameter explicitly, matching
score_model.py's call exactly, so there is one source of truth instead of two silently
divergent ones.
"""

from __future__ import annotations

import importlib
from typing import Any, Final, Optional

import numpy as np

METRIC_WEIGHTS: Final[dict[str, float]] = {
    "topo_score": 0.30,
    "surface_dice": 0.35,
    "voi_score": 0.35,
}

DEFAULT_METRIC_KWARGS: Final[dict[str, Any]] = {
    "dims": (0, 1, 2),
    "spacing": (1.0, 1.0, 1.0),
    "surface_tolerance": 2.0,
    "voi_connectivity": 26,
    "voi_transform": "one_over_one_plus",
    "voi_alpha": 0.3,
    "combine_weights": (0.30, 0.35, 0.35),
    "fg_threshold": None,
    "ignore_label": 2,
    "ignore_mask": None,
}

_CANDIDATE_MODULES: Final[tuple[str, ...]] = (
    "topometrics",
    "topological_metrics",
    "topological_metrics_kaggle",
)


class MetricUnavailable(RuntimeError):
    """Raised when the official metric package is not importable."""


def load_metric_module():
    """Import the official metric package, or explain how to install it."""
    for name in _CANDIDATE_MODULES:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise MetricUnavailable(
        "Official metric package not found (tried: "
        + ", ".join(_CANDIDATE_MODULES)
        + "). Install it with:\n    bash packages/vesuvius_evaluation/scripts/install_topometrics.sh"
    )


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_pair(
    prediction: np.ndarray,
    label: np.ndarray,
    **overrides: Any,
) -> dict[str, Any]:
    """Score one prediction against one raw label volume.

    `label` should be the untouched label with its ignore class intact.
    Calls `compute_leaderboard_score` with `DEFAULT_METRIC_KWARGS` (matching
    scripts/evaluation/score_model.py exactly -- see this module's docstring for why that
    match matters), overridden by anything passed in `overrides`.
    """
    module = load_metric_module()
    kwargs = {**DEFAULT_METRIC_KWARGS, **overrides}
    report = module.compute_leaderboard_score(prediction, label, **kwargs)

    topo = getattr(report, "topo", None)
    voi = getattr(report, "voi", None)
    topo_f1 = dict(getattr(topo, "topoF1_by_dim", {}) or {})

    return {
        "score": _as_float(getattr(report, "score", None)),
        "topo_score": _as_float(getattr(topo, "toposcore", None)),
        "surface_dice": _as_float(getattr(report, "surface_dice", None)),
        "voi_score": _as_float(getattr(voi, "voi_score", None)),
        "voi_split": _as_float(getattr(voi, "voi_split", None)),
        "voi_merge": _as_float(getattr(voi, "voi_merge", None)),
        "voi_total": _as_float(getattr(voi, "voi_total", None)),
        "n_foreground": _as_float(getattr(voi, "n_foreground", None)),
        # k=0 components, k=1 tunnels/handles, k=2 cavities.
        "topo_f1_dim0": _as_float(topo_f1.get(0)),
        "topo_f1_dim1": _as_float(topo_f1.get(1)),
        "topo_f1_dim2": _as_float(topo_f1.get(2)),
    }
