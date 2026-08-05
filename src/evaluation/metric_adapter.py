"""Thin adapter over the official competition metric package.

The scoring code ships as the Kaggle dataset `sohier/vesuvius-metric-resources`
(install via `scripts/setup_metric.sh`). Everything that depends on its exact
API is isolated here so the rest of the harness stays stable.

    Score = 0.30 * TopoScore + 0.35 * SurfaceDice@2.0 + 0.35 * VOI_score
"""

from __future__ import annotations

import importlib
from typing import Final

import numpy as np

METRIC_WEIGHTS: Final[dict[str, float]] = {
    "topo_score": 0.30,
    "surface_dice": 0.35,
    "voi_score": 0.35,
}

SURFACE_DICE_TAU: Final[float] = 2.0

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


def score_pair(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Score one prediction/ground-truth pair of binary volumes.

    Returns the three sub-scores; the caller combines them via
    :func:`evaluation.harness.composite_score`.
    """
    module = load_metric_module()

    # The public entry point differs between releases of the package. Resolve it
    # at call time and fail loudly rather than guessing at a signature.
    raise MetricUnavailable(
        "score_pair() is not wired to the installed package yet.\n"
        f"Imported module: {getattr(module, '__name__', '?')} "
        f"({getattr(module, '__file__', '?')})\n"
        "Run `bash scripts/setup_metric.sh --inspect-only` and wire the entry "
        "point in src/evaluation/metric_adapter.py."
    )
