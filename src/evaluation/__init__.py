"""Local scoring against the competition metric."""

from evaluation.harness import (
    CaseScore,
    aggregate_by_scroll,
    apply_ignore,
    composite_score,
    evaluate_directory,
)
from evaluation.metric_adapter import METRIC_WEIGHTS, MetricUnavailable, score_pair

__all__ = [
    "METRIC_WEIGHTS",
    "CaseScore",
    "MetricUnavailable",
    "aggregate_by_scroll",
    "apply_ignore",
    "composite_score",
    "evaluate_directory",
    "score_pair",
]
