"""Surface Detection EDA and modeling-insight utilities."""

from eda.dataset_analysis import SurfaceDatasetInspector, VolumeReport
from eda.modeling_insights import (
    ImbalanceReport3D,
    ThicknessStats,
    compute_imbalance_report_3d,
    compute_scroll_domain_shift,
    compute_surface_thickness,
    validation_strategy_recommendation,
)
from eda.statistics import ClassStats, IntensityStats, compute_class_stats, compute_intensity_stats

__all__ = [
    "ClassStats",
    "ImbalanceReport3D",
    "IntensityStats",
    "SurfaceDatasetInspector",
    "ThicknessStats",
    "VolumeReport",
    "compute_class_stats",
    "compute_imbalance_report_3d",
    "compute_intensity_stats",
    "compute_scroll_domain_shift",
    "compute_surface_thickness",
    "validation_strategy_recommendation",
]
