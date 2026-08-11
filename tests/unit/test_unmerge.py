"""Unit tests for the metric-guided unmerge novelty layer's pure geometry logic
(propose_cuts and its helpers) -- no GPU, no dataset, no metric package needed.

This is the project's own novel contribution, and had zero test
coverage before this file -- these tests exercise the actual bridge-detection/cutting logic
directly on small synthetic volumes with known, hand-verifiable geometry, the same way
tests/unit/test_affinity_targets.py already does for the affinity-target math.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import label as cc_label

from vesuvius_surface.postprocess.unmerge import UnmergeConfig, propose_cuts


def _two_blobs_thin_bridge(blob_size: int = 6, gap_len: int = 3) -> np.ndarray:
    """Two `blob_size`^3 cubes connected by a 1-voxel-wide, `gap_len`-voxel-long bridge --
    the exact shape the unmerge layer targets: erosion should sever the thin bridge while
    leaving both cube interiors intact as separate seeds."""
    shape = (blob_size, blob_size, blob_size * 2 + gap_len)
    vol = np.zeros(shape, dtype=np.uint8)
    vol[:, :, :blob_size] = 1
    vol[:, :, blob_size + gap_len:] = 1
    mid = blob_size // 2
    vol[mid, mid, blob_size:blob_size + gap_len] = 1  # the 1-voxel bridge
    return vol


def _single_solid_blob(size: int = 8) -> np.ndarray:
    return np.ones((size, size, size), dtype=np.uint8)


class TestProposeCuts:
    def test_thin_bridge_is_cut_into_two_pieces(self):
        mask = _two_blobs_thin_bridge()
        cfg = UnmergeConfig(erosion_radius=1, min_seed_size=10, min_piece_size=10, cut_width=1)
        result, infos = propose_cuts(mask, cfg)

        assert len(infos) == 1
        assert infos[0].cut is True
        assert infos[0].n_seeds == 2

        _, n_components_after = cc_label(result)
        assert n_components_after == 2, "the two blobs should be separate components after cutting"

    def test_solid_blob_is_not_a_candidate(self):
        """A single solid blob (no thin neck) must never be cut -- erosion should never
        produce >=2 surviving seeds from genuinely solid material."""
        mask = _single_solid_blob()
        cfg = UnmergeConfig(erosion_radius=1, min_seed_size=10, min_piece_size=10)
        result, infos = propose_cuts(mask, cfg)

        assert len(infos) == 1
        assert infos[0].cut is False
        assert infos[0].reason == "not_candidate"
        np.testing.assert_array_equal(result, mask)

    def test_empty_mask_returns_no_components(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        result, infos = propose_cuts(mask)
        assert infos == []
        assert result.sum() == 0

    def test_small_seeds_below_threshold_are_not_candidates(self):
        """The same two-blob-thin-bridge geometry, but with min_seed_size set higher than
        either blob's post-erosion size -- must NOT be treated as a candidate (this is
        exactly a real calibration mistake hit during development:
        erosion_radius=2 eroded whole sheets away and found zero candidates)."""
        mask = _two_blobs_thin_bridge(blob_size=4)
        cfg = UnmergeConfig(erosion_radius=1, min_seed_size=10_000, min_piece_size=1)
        result, infos = propose_cuts(mask, cfg)

        assert infos[0].cut is False
        np.testing.assert_array_equal(result, mask)

    def test_max_candidates_per_volume_budget_is_respected(self):
        """Two independent bridge structures in one volume; with a budget of 1, only one
        should actually be cut."""
        blob = _two_blobs_thin_bridge(blob_size=5, gap_len=2)
        vol = np.zeros((blob.shape[0], blob.shape[1] * 2 + 2, blob.shape[2]), dtype=np.uint8)
        vol[:, : blob.shape[1], :] = blob
        vol[:, blob.shape[1] + 2:, :] = blob

        cfg = UnmergeConfig(erosion_radius=1, min_seed_size=5, min_piece_size=5, max_candidates_per_volume=1)
        _, infos = propose_cuts(vol, cfg)

        cut_count = sum(1 for info in infos if info.cut)
        assert cut_count == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
