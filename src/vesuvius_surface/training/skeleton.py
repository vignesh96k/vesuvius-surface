"""Tubed skeletonisation of the surface class.

Ported from Kirchhoff et al., *Skeleton Recall Loss for Connectivity Conserving
and Resource Efficient Segmentation of Thin Tubular Structures* (ECCV 2024),
https://github.com/MIC-DKFZ/Skeleton-Recall — specifically
``nnunetv2/training/data_augmentation/custom_transforms/skeletonization.py``
and ``calculate_tubed_skeleton`` in ``data_loader_3d_skel.py``.

Two deliberate departures from upstream, both forced by our label encoding:

1.  Upstream binarises with ``seg > 0``. Our label 2 is *ignore*, and on a
    representative volume ignore is 58% of voxels against 4.9% surface, so
    ``seg > 0`` would skeletonise the union of surface and unlabelled space —
    an object with no relationship to the papyrus sheet. We binarise on
    ``seg == LABEL_SURFACE`` instead. Upstream's own trainer warns that ignore
    label support is "experimental"; this is the concrete reason why.

2.  Upstream tubes the skeleton with two calls to ``skimage.morphology.dilation``
    and relies on the default footprint. That default changed between
    scikit-image releases (a connectivity-1 cross in <= 0.22, a full 3x3x3 box
    in >= 0.25), which silently changes the tube radius. We pass an explicit
    connectivity-1 structuring element, reproducing the behaviour of the
    version the paper was written against without drifting under upgrades.
"""

from __future__ import annotations

import numpy as np

from vesuvius_surface.data.schema import LABEL_SURFACE
from vesuvius_surface.training import compat

DEFAULT_TUBE_DILATIONS = 2


def tubed_skeleton(
    segmentation: np.ndarray,
    *,
    surface_label: int = LABEL_SURFACE,
    do_tube: bool = True,
    n_dilations: int = DEFAULT_TUBE_DILATIONS,
) -> np.ndarray:
    """Return the tubed skeleton of ``segmentation`` as a ``{0, 1}`` int16 array.

    ``segmentation`` is a single spatial array (no channel axis), 2D or 3D.

    The tube is intersected back with the surface mask, exactly as upstream
    does with ``skel *= seg_all``. That keeps the target inside the annotated
    structure: dilation widens the skeleton towards the sheet's own boundary
    rather than bleeding into background or ignore.
    """
    if segmentation.ndim not in (2, 3):
        raise ValueError(
            "tubed_skeleton expects a 2D or 3D array without a channel axis, "
            f"got shape {segmentation.shape}"
        )

    surface = np.ascontiguousarray(segmentation == surface_label)
    skeleton = np.zeros(segmentation.shape, dtype=np.int16)
    if not surface.any():
        return skeleton

    thin = np.asarray(compat.skeletonize(surface), dtype=bool)

    if do_tube and n_dilations > 0:
        footprint = compat.generate_binary_structure(segmentation.ndim, 1)
        thin = compat.binary_dilation(thin, structure=footprint, iterations=n_dilations)

    # Restrict back to the surface class. This also guarantees the result is
    # strictly {0, 1}: the loss scatters it into a one-hot tensor whose channel
    # count is num_segmentation_heads (2 here, because the ignore label gets no
    # output channel), so a stray 2 would be an out-of-bounds scatter index and
    # a singularly unhelpful CUDA assert.
    np.logical_and(thin, surface, out=thin)
    skeleton[thin] = 1
    return skeleton


__all__ = ["DEFAULT_TUBE_DILATIONS", "tubed_skeleton"]
