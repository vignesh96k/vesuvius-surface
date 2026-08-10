"""Thin adapter over the official competition metric package (`topometrics`).

Installed from the Kaggle dataset `sohier/vesuvius-metric-resources` via
`scripts/setup_metric.sh`. Everything that depends on its API lives here.

    Score = 0.30*TopoScore + 0.35*SurfaceDice@2.0 + 0.35*VOI_score

We call `compute_leaderboard_score` with its own defaults and override nothing,
so local numbers stay in parity with the package. In particular the package
handles the ignore class itself (`ignore_label=2`) — do not pre-mask labels.
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
        + "). Install it with:\n    bash scripts/setup_metric.sh"
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
    `overrides` are passed straight through to `compute_leaderboard_score`;
    leave empty for leaderboard parity.
    """
    module = load_metric_module()
    report = module.compute_leaderboard_score(prediction, label, **overrides)

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
