"""Affinity targets for the Stage 2a auxiliary head.

Why affinities at all
---------------------
``VOI_split`` and ``VOI_merge`` are connectomics metrics (SNEMI3D, CREMI). The
standard method in that field is not to predict a semantic mask and post-process
it, but to predict *affinities* between neighbouring voxels and agglomerate the
resulting graph (Funke et al., "Large Scale Image Segmentation with Structured
Loss Based Deep Learning for Connectome Reconstruction", TPAMI 2018; Wolf et
al., "The Mutex Watershed", ECCV 2018). Funke et al. report that long-range
affinities improve results *even when used only as an auxiliary loss and
discarded at inference* — that is exactly what Stage 2a tests, and it is why
this module ships without any agglomeration code.

Definition used here
--------------------
For a voxel ``v`` and an offset ``o``, the affinity of the pair ``(v, v + o)``
is stored at index ``v`` of channel ``k`` (``offsets[k] == o``) and is

* ``1``  if both voxels are surface *and* belong to the same sheet instance,
* ``0``  if the pair is valid but not both-surface-same-instance,
* ``-1`` if the pair must not contribute to the loss.

Packing the validity mask into the target as ``-1`` rather than emitting a
second mask array halves the memory that rides through the augmentation
pipeline, which matters because these arrays travel as extra segmentation
channels of a 128^3 patch.

A pair is invalid when ``v + o`` leaves the patch, or when either voxel carries
``LABEL_IGNORE``. Optionally (and by default) a pair is also invalid when
neither voxel is surface: with 37% background against 4.9% surface, roughly 78%
of in-bounds pairs are background-to-background and trivially zero, and they
would dominate the loss without teaching anything. See ``mask_background_pairs``.

Sheet instances
---------------
Instances are connected components of the *surface* class, with 26-connectivity
to match ``voi_connectivity=26`` in the installed metric package. This is
deliberate: the competition metric derives its instances the same way, so two
sheets that genuinely touch in the ground truth are one instance for the metric
and must be one instance here too. The affinity target teaches the metric's
notion of an instance, not an idealised one.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from data.schema import LABEL_IGNORE, LABEL_SURFACE
from training import compat

Offset = Tuple[int, ...]

#: Value stored where a pair must not contribute to the loss.
AFFINITY_INVALID = -1

#: Connectivity used to derive sheet instances, matched to the metric's
#: ``voi_connectivity`` default (see docs/metric.md).
INSTANCE_CONNECTIVITY = 26

#: Directed unit offsets. Three offsets cover the full 6-neighbourhood: the
#: edge {v, v+e} and the edge {v+e, v} are the same undirected edge, seen from
#: the two ends. Connectomics work (Funke et al., Lee et al. 2017) uses the
#: three positive directions for exactly this reason.
NEAREST_NEIGHBOUR_OFFSETS: Tuple[Offset, ...] = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)

#: Long-range offsets on a geometric ladder 2, 4, 8, 16 along each axis.
#:
#: The ladder is chosen from measurements in research_log.md rather than copied
#: from SNEMI3D. Median papyrus sheet thickness is 2 voxels, so an offset of 2
#: straddles a single sheet; 4 to 16 spans the gap to a neighbouring wrap, and
#: those are the pairs that must be predicted 0 for the network to learn that
#: two adjacent sheets are distinct objects. Fused neighbouring sheets are the
#: failure mode the 1st-place team never solved and the reason voi_merge is
#: ~1.0. The Mutex Watershed paper's {1, 3, 9, 27} ladder is anisotropic and
#: tuned to EM voxel sizes; ours is isotropic because our data is, and stops at
#: 16 because beyond that nearly every pair crosses several sheets and the
#: target degenerates to a constant 0.
LONG_RANGE_OFFSETS: Tuple[Offset, ...] = (
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
    (4, 0, 0),
    (0, 4, 0),
    (0, 0, 4),
    (8, 0, 0),
    (0, 8, 0),
    (0, 0, 8),
    (16, 0, 0),
    (0, 16, 0),
    (0, 0, 16),
)

#: 15 offsets: 3 short-range plus 12 long-range.
DEFAULT_AFFINITY_OFFSETS: Tuple[Offset, ...] = (
    NEAREST_NEIGHBOUR_OFFSETS + LONG_RANGE_OFFSETS
)


def parse_offsets(spec: str) -> Tuple[Offset, ...]:
    """Parse ``"1,0,0;0,1,0;0,0,4"`` into a tuple of integer offsets.

    Used by the audit script and by anyone overriding the trainer's offset list
    from a shell script. Trainer subclasses should set ``affinity_offsets``
    directly instead.
    """
    offsets = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        try:
            offset = tuple(int(p) for p in parts)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse offset {chunk!r}. Expected comma-separated "
                'integers, offsets separated by ";", e.g. "1,0,0;0,1,0".'
            ) from exc
        if len(offset) != 3:
            raise ValueError(
                f"Offset {chunk!r} has {len(offset)} components; 3D offsets need 3."
            )
        if all(c == 0 for c in offset):
            raise ValueError("The zero offset is degenerate (affinity of a voxel to itself).")
        offsets.append(offset)
    if not offsets:
        raise ValueError(f"No offsets parsed from {spec!r}.")
    return tuple(offsets)


def validate_offsets(offsets: Sequence[Offset], ndim: int = 3) -> Tuple[Offset, ...]:
    """Check every offset is ``ndim``-dimensional, integral and non-zero."""
    validated = []
    for offset in offsets:
        offset = tuple(int(c) for c in offset)
        if len(offset) != ndim:
            raise ValueError(
                f"Offset {offset} has {len(offset)} components but the data is "
                f"{ndim}D. Offsets must match the patch dimensionality."
            )
        if all(c == 0 for c in offset):
            raise ValueError("The zero offset is degenerate (affinity of a voxel to itself).")
        validated.append(offset)
    if not validated:
        raise ValueError("At least one offset is required.")
    return tuple(validated)


def instance_labels(
    segmentation: np.ndarray,
    *,
    surface_label: int = LABEL_SURFACE,
    connectivity: int = INSTANCE_CONNECTIVITY,
) -> np.ndarray:
    """Label connected components of the surface class.

    Returns an int32 array, 0 outside the surface class. ``connectivity`` is
    given in the neighbourhood-size convention used by the metric package (6 or
    26 in 3D), not scipy's rank convention.
    """
    ndim = segmentation.ndim
    if ndim not in (2, 3):
        raise ValueError(
            "instance_labels expects a 2D or 3D array without a channel axis, "
            f"got shape {segmentation.shape}"
        )

    rank = _connectivity_to_rank(connectivity, ndim)
    structure = compat.generate_binary_structure(ndim, rank)
    labelled, _ = compat.cc_label(segmentation == surface_label, structure=structure)
    return labelled.astype(np.int32, copy=False)


def _connectivity_to_rank(connectivity: int, ndim: int) -> int:
    """Map a neighbourhood size (6/18/26 in 3D, 4/8 in 2D) to a scipy rank."""
    table = {
        3: {6: 1, 18: 2, 26: 3},
        2: {4: 1, 8: 2},
    }[ndim]
    try:
        return table[connectivity]
    except KeyError as exc:
        raise ValueError(
            f"connectivity={connectivity} is not a valid {ndim}D neighbourhood "
            f"size. Choose one of {sorted(table)}."
        ) from exc


def _pair_slices(offset: Offset, shape: Sequence[int]):
    """Slices selecting the voxels ``v`` and the voxels ``v + offset``.

    For each axis the valid source range is ``[max(0, -o), dim - max(0, o))``,
    so both members of the pair stay inside the array. Offsets may be negative.
    """
    source, target = [], []
    for offset_component, dim in zip(offset, shape):
        source.append(slice(max(0, -offset_component), min(dim, dim - offset_component)))
        target.append(slice(max(0, offset_component), min(dim, dim + offset_component)))
    return tuple(source), tuple(target)


def affinity_targets(
    segmentation: np.ndarray,
    offsets: Sequence[Offset] = DEFAULT_AFFINITY_OFFSETS,
    *,
    instances: np.ndarray | None = None,
    surface_label: int = LABEL_SURFACE,
    ignore_label: int = LABEL_IGNORE,
    connectivity: int = INSTANCE_CONNECTIVITY,
    mask_background_pairs: bool = True,
) -> np.ndarray:
    """Build the affinity target stack for ``segmentation``.

    Parameters
    ----------
    segmentation
        Spatial label array (no channel axis) holding ``LABEL_BG`` /
        ``LABEL_SURFACE`` / ``LABEL_IGNORE``.
    offsets
        Neighbour offsets, one output channel each.
    instances
        Precomputed instance labels. When ``None`` they are derived from
        ``segmentation`` by connected components — see the module docstring and
        :class:`training.transforms.AffinityTargetTransform` for why that is the
        default in the training pipeline.
    mask_background_pairs
        When true, pairs where *neither* voxel is surface are marked invalid.

    Returns
    -------
    int8 array of shape ``(len(offsets), *segmentation.shape)`` with values in
    ``{-1, 0, 1}``, where ``-1`` is :data:`AFFINITY_INVALID`.
    """
    offsets = validate_offsets(offsets, ndim=segmentation.ndim)
    shape = segmentation.shape

    if instances is None:
        instances = instance_labels(
            segmentation, surface_label=surface_label, connectivity=connectivity
        )
    elif instances.shape != shape:
        raise ValueError(
            f"instances shape {instances.shape} does not match segmentation "
            f"shape {shape}. Instance labels must be spatially aligned with the "
            "segmentation they describe — if you precomputed them, they have to "
            "have gone through the same crop and augmentation."
        )

    is_surface = segmentation == surface_label
    is_ignore = segmentation == ignore_label

    targets = np.full((len(offsets), *shape), AFFINITY_INVALID, dtype=np.int8)

    for channel, offset in enumerate(offsets):
        source, target = _pair_slices(offset, shape)
        if any(s.stop <= s.start for s in source):
            # Offset longer than the patch along some axis: no valid pair at
            # all. Leave the channel fully invalid rather than erroring, so an
            # aggressive offset list still trains (it just contributes nothing
            # on small patches).
            continue

        surface_a = is_surface[source]
        surface_b = is_surface[target]

        valid = ~is_ignore[source] & ~is_ignore[target]
        if mask_background_pairs:
            valid &= surface_a | surface_b

        positive = surface_a & surface_b & (instances[source] == instances[target])

        channel_view = targets[channel]
        channel_view[source] = np.where(valid, positive.astype(np.int8), np.int8(AFFINITY_INVALID))

    return targets


__all__ = [
    "AFFINITY_INVALID",
    "DEFAULT_AFFINITY_OFFSETS",
    "INSTANCE_CONNECTIVITY",
    "LONG_RANGE_OFFSETS",
    "NEAREST_NEIGHBOUR_OFFSETS",
    "Offset",
    "affinity_targets",
    "instance_labels",
    "parse_offsets",
    "validate_offsets",
]
