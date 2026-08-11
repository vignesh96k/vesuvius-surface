#!/usr/bin/env python3
"""First (finest) high-pass sub-band of a 3D Laplacian pyramid: the original volume minus
its own Gaussian-blurred version. Captures fine-scale edge/texture detail at the original
resolution, before any downsampling -- the same sub-band decomposition idea from M-SCQALE
(the user's own prior published work), generalized from 2D image quality assessment to a
3D CT volume as an auxiliary input channel for surface segmentation.

Real, previously-executed code (not a reconstruction) -- the exact function that produced
Dataset101_VesuviusSurfaceLaplacian's channel 1 (see build_dataset101_laplacian.py), which
in turn was reused as Dataset102_VesuviusSurfaceHighpassOnly's sole channel. These two
datasets are the "laplacian" and "highpassonly" candidates in the 100-epoch 5-way
comparison.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def laplacian_highpass_3d(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """volume: 3D array (any numeric dtype). Returns float32 high-pass residual,
    same shape as input: volume - gaussian_blur(volume, sigma)."""
    vol_f = volume.astype(np.float32)
    blurred = gaussian_filter(vol_f, sigma=sigma, mode="reflect")
    return vol_f - blurred
