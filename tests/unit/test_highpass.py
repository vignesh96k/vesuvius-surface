"""Unit test for scripts/data_prep/highpass.py -- the Laplacian high-pass sub-band function
behind the 100-epoch 5-way comparison's "laplacian" and "highpassonly" candidates.

Verified separately (not just here) against the real, already-computed channel-1 data from
the original Dataset101_VesuviusSurfaceLaplacian run: exact byte-for-byte match. This test
covers the function's basic properties without needing that real data file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data_prep"))
from highpass import laplacian_highpass_3d  # noqa: E402


class TestLaplacianHighpass3D:
    def test_constant_volume_has_zero_highpass(self):
        # A uniform volume has no high-frequency content -- the residual should vanish.
        vol = np.full((8, 8, 8), 42.0, dtype=np.float32)
        result = laplacian_highpass_3d(vol, sigma=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-4)

    def test_output_shape_and_dtype(self):
        vol = np.random.randint(0, 256, size=(10, 12, 14)).astype(np.uint8)
        result = laplacian_highpass_3d(vol, sigma=1.0)
        assert result.shape == vol.shape
        assert result.dtype == np.float32

    def test_isolated_bright_voxel_produces_local_positive_residual(self):
        # A single bright spike in a dark volume: the highpass residual at the spike
        # itself should be strongly positive (original >> its own local blur there).
        vol = np.zeros((9, 9, 9), dtype=np.float32)
        vol[4, 4, 4] = 100.0
        result = laplacian_highpass_3d(vol, sigma=1.0)
        assert result[4, 4, 4] > 0

    def test_larger_sigma_increases_residual_magnitude_for_a_step_edge(self):
        # A bigger blur radius removes more low-frequency structure, leaving a larger
        # residual at an edge -- checked on a simple step volume.
        vol = np.zeros((16, 16, 16), dtype=np.float32)
        vol[8:, :, :] = 100.0
        small = laplacian_highpass_3d(vol, sigma=0.5)
        large = laplacian_highpass_3d(vol, sigma=2.0)
        assert np.abs(large).sum() > np.abs(small).sum()
