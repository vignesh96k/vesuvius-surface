"""Local scoring against the official competition metric."""

from evaluation.harness import (
    CaseScore,
    aggregate_by_scroll,
    discover_cases,
    evaluate_directory,
)
from evaluation.metric_adapter import METRIC_WEIGHTS, MetricUnavailable, score_pair

__all__ = [
    "METRIC_WEIGHTS",
    "CaseScore",
    "MetricUnavailable",
    "aggregate_by_scroll",
    "discover_cases",
    "evaluate_directory",
    "score_pair",
]
