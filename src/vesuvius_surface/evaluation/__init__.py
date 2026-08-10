"""Local scoring against the official competition metric."""

from vesuvius_surface.evaluation.harness import (
    CaseScore,
    aggregate_by_scroll,
    discover_cases,
    evaluate_directory,
)
from vesuvius_surface.evaluation.metric_adapter import METRIC_WEIGHTS, MetricUnavailable, score_pair

__all__ = [
    "METRIC_WEIGHTS",
    "CaseScore",
    "MetricUnavailable",
    "aggregate_by_scroll",
    "discover_cases",
    "evaluate_directory",
    "score_pair",
]
