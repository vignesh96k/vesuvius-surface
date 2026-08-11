"""Unit tests for the metric-guided fragment-bridging novelty layer's pure geometry logic
(propose_bridges and its helpers) -- no GPU, no dataset, no metric package needed.

This project's other novel contribution alongside unmerge (see
postprocess/bridge.py's module docstring) -- addresses the opposite failure mode (components
wrongly split apart, not wrongly fused together). Same testing approach as
tests/unit/test_unmerge.py: exercise the actual candidate-detection/bridging logic directly on
small synthetic volumes with known, hand-verifiable geometry.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import label as cc_label

from vesuvius_surface.postprocess.bridge import BridgeConfig, propose_bridges


def _aligned_gap(gap: int = 4) -> np.ndarray:
    """Two short line segments along the same axis, separated by `gap` voxels -- a broken
    line that should continue in the same direction, the exact shape bridging targets."""
    mask = np.zeros((30, 30, 30), dtype=bool)
    mask[5:15, 14:16, 10] = True
    mask[15 + gap:25 + gap, 14:16, 10] = True
    return mask


def _perpendicular_near_miss() -> np.ndarray:
    """Two fragments close together but perpendicular -- looks like a candidate by distance
    alone, but the directional test should reject it (this is exactly the case unmerge's own
    Voronoi cut targets instead, not bridging)."""
    mask = np.zeros((30, 30, 30), dtype=bool)
    mask[5:15, 14:16, 10] = True
    mask[19:21, 5:15, 10] = True
    return mask


class TestProposeBridges:
    def test_aligned_gap_is_bridged_into_one_component(self):
        mask = _aligned_gap(gap=4)
        cfg = BridgeConfig(max_gap=8, angle_tol_deg=50, min_component_size=10, tube_radius=0)
        result, n_bridged = propose_bridges(mask, cfg)

        assert n_bridged == 1
        _, n_components_after = cc_label(result, structure=np.ones((3, 3, 3)))
        assert n_components_after == 1, "the two aligned fragments should merge into one component"

    def test_perpendicular_fragments_are_not_bridged(self):
        mask = _perpendicular_near_miss()
        cfg = BridgeConfig(max_gap=8, angle_tol_deg=50, min_component_size=8, tube_radius=0)
        result, n_bridged = propose_bridges(mask, cfg)

        assert n_bridged == 0
        _, n_components_after = cc_label(result, structure=np.ones((3, 3, 3)))
        assert n_components_after == 2, "unrelated perpendicular fragments must stay separate"

    def test_gap_wider_than_max_gap_is_not_bridged(self):
        mask = _aligned_gap(gap=20)
        cfg = BridgeConfig(max_gap=8, angle_tol_deg=50, min_component_size=10, tube_radius=0)
        result, n_bridged = propose_bridges(mask, cfg)

        assert n_bridged == 0
        np.testing.assert_array_equal(result.astype(bool), mask)

    def test_single_component_has_no_candidates(self):
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[5:15, 5:15, 5:15] = True
        cfg = BridgeConfig(min_component_size=10)
        result, n_bridged = propose_bridges(mask, cfg)

        assert n_bridged == 0
        np.testing.assert_array_equal(result.astype(bool), mask)

    def test_empty_mask_returns_no_bridges(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        result, n_bridged = propose_bridges(mask)
        assert n_bridged == 0
        assert result.sum() == 0

    def test_components_below_min_size_are_ignored(self):
        """Same aligned-gap geometry, but with min_component_size set higher than either
        fragment's voxel count -- must not be treated as a candidate (mirrors
        test_unmerge.py's analogous min_seed_size calibration test)."""
        mask = _aligned_gap(gap=4)
        cfg = BridgeConfig(max_gap=8, angle_tol_deg=50, min_component_size=10_000, tube_radius=0)
        result, n_bridged = propose_bridges(mask, cfg)

        assert n_bridged == 0
        np.testing.assert_array_equal(result.astype(bool), mask)

    def test_tube_radius_thickens_the_bridge(self):
        mask = _aligned_gap(gap=4)
        thin_result, _ = propose_bridges(mask, BridgeConfig(max_gap=8, angle_tol_deg=50,
                                                              min_component_size=10, tube_radius=0))
        thick_result, _ = propose_bridges(mask, BridgeConfig(max_gap=8, angle_tol_deg=50,
                                                               min_component_size=10, tube_radius=2))
        assert thick_result.sum() > thin_result.sum()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
