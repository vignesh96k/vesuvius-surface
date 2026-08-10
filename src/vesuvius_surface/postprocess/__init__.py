"""Post-processing baselines and extensions for surface predictions.

``first_place`` reproduces the 1st-place morphological / height-map chain
(control). Novelty layers (merge cuts, metric-tuned params) build on top of it.
"""

from __future__ import annotations

from vesuvius_surface.postprocess.first_place import (
    FIRST_PLACE_STAGES,
    PostprocessConfig,
    apply_first_place,
    binarize_prediction,
    make_ball_footprint,
    plug_holes_lut,
    remove_small_components,
)
from vesuvius_surface.postprocess.pipeline import run_directory, run_staged

__all__ = [
    "FIRST_PLACE_STAGES",
    "PostprocessConfig",
    "apply_first_place",
    "binarize_prediction",
    "make_ball_footprint",
    "plug_holes_lut",
    "remove_small_components",
    "run_directory",
    "run_staged",
]
