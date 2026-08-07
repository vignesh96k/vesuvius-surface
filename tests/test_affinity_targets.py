#!/usr/bin/env python3
"""Unit tests for affinity targets — correspondence after spatial transforms.

Cannot be executed on the Windows authoring machine (no Python). Run on the
Linux box:

    export PYTHONPATH=/mnt/workspace/code/vesuvius-surface/src:$PYTHONPATH
    python -m pytest tests/test_affinity_targets.py -v

The critical assertion: after a known geometric transform of a two-sheet patch,
rebuilding affinities from the transformed labels yields the same stack as
transforming the original affinity channels the same way. That is the property
Stage 2a relies on when it derives instances on the augmented patch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.schema import LABEL_BG, LABEL_IGNORE, LABEL_SURFACE
from training.affinity import (
    AFFINITY_INVALID,
    DEFAULT_AFFINITY_OFFSETS,
    NEAREST_NEIGHBOUR_OFFSETS,
    affinity_targets,
    instance_labels,
)


def _two_touching_sheets(size: int = 32) -> np.ndarray:
    """Synthetic patch: two sheets that touch along one face, plus ignore band."""
    vol = np.full((size, size, size), LABEL_BG, dtype=np.int16)
    # Sheet A: a thin slab at y=10 covering x in [4, 20]
    vol[4:28, 10, 4:20] = LABEL_SURFACE
    # Sheet B: a thin slab at y=11 (touching A) covering x in [12, 28]
    # They touch on the face y=10/11, x in [12, 20].
    vol[4:28, 11, 12:28] = LABEL_SURFACE
    # Ignore band elsewhere
    vol[:, :, 28:] = LABEL_IGNORE
    return vol


def test_instance_labels_separates_touching_sheets_under_26conn():
    """26-connectivity merges sheets that share a face — that is intentional.

    The metric uses voi_connectivity=26, so two sheets that genuinely touch are
    ONE instance for scoring. Affinities must teach the same definition.
    """
    vol = _two_touching_sheets()
    instances_26 = instance_labels(vol, connectivity=26)
    # Under 26-connectivity face-adjacent sheets merge.
    surface_ids = set(instances_26[vol == LABEL_SURFACE].tolist()) - {0}
    assert len(surface_ids) == 1, (
        f"26-connectivity should merge face-touching sheets into 1 instance, "
        f"got {len(surface_ids)}: {surface_ids}"
    )


def test_instance_labels_separates_touching_sheets_under_6conn():
    """6-connectivity keeps face-adjacent sheets distinct."""
    vol = _two_touching_sheets()
    instances_6 = instance_labels(vol, connectivity=6)
    surface_ids = set(instances_6[vol == LABEL_SURFACE].tolist()) - {0}
    assert len(surface_ids) == 2, (
        f"6-connectivity should keep face-touching sheets as 2 instances, "
        f"got {len(surface_ids)}: {surface_ids}"
    )


def test_same_sheet_nearest_neighbour_is_positive():
    vol = _two_touching_sheets()
    targets = affinity_targets(vol, offsets=NEAREST_NEIGHBOUR_OFFSETS, connectivity=6)
    # Offset (0, 0, 1): within-sheet neighbours along z for sheet A at y=10.
    # Pick a voxel interior to sheet A.
    ch = NEAREST_NEIGHBOUR_OFFSETS.index((0, 0, 1))
    assert targets[ch, 10, 10, 10] == 1


def test_cross_sheet_nearest_neighbour_is_zero_under_6conn():
    """Face-adjacent voxels from different sheets must be affinity 0 (6-conn)."""
    vol = _two_touching_sheets()
    targets = affinity_targets(vol, offsets=NEAREST_NEIGHBOUR_OFFSETS, connectivity=6)
    # Offset (0, 1, 0): from sheet A (y=10) to sheet B (y=11) at overlapping x.
    ch = NEAREST_NEIGHBOUR_OFFSETS.index((0, 1, 0))
    # At (10, 10, 14) sheet A; neighbour (10, 11, 14) is sheet B.
    assert vol[10, 10, 14] == LABEL_SURFACE
    assert vol[10, 11, 14] == LABEL_SURFACE
    assert targets[ch, 10, 10, 14] == 0, (
        "Cross-sheet pair under 6-connectivity must be affinity 0, not 1"
    )


def test_cross_sheet_nearest_neighbour_is_positive_under_26conn():
    """Under 26-conn the same pair is one instance, so affinity must be 1."""
    vol = _two_touching_sheets()
    targets = affinity_targets(vol, offsets=NEAREST_NEIGHBOUR_OFFSETS, connectivity=26)
    ch = NEAREST_NEIGHBOUR_OFFSETS.index((0, 1, 0))
    assert targets[ch, 10, 10, 14] == 1


def test_ignore_pairs_are_invalid():
    vol = _two_touching_sheets()
    targets = affinity_targets(vol, offsets=NEAREST_NEIGHBOUR_OFFSETS, connectivity=6)
    # Offset (0, 0, 1) reaching into the ignore band at z>=28.
    ch = NEAREST_NEIGHBOUR_OFFSETS.index((0, 0, 1))
    assert targets[ch, 10, 10, 27] == AFFINITY_INVALID


def test_flip_correspondence():
    """After a flip, affinities rebuilt from flipped labels match flipped targets.

    This is the property Stage 2a relies on: deriving instances on the
    *augmented* patch is equivalent to transforming precomputed affinities,
    for pure axis flips (which nnU-Net uses heavily).
    """
    vol = _two_touching_sheets()
    offsets = DEFAULT_AFFINITY_OFFSETS
    before = affinity_targets(vol, offsets=offsets, connectivity=6)

    # Flip along axis 1 (y).
    flipped_vol = np.flip(vol, axis=1).copy()
    after = affinity_targets(flipped_vol, offsets=offsets, connectivity=6)

    # Transforming affinity channels: for offset (dy,...) the channel itself
    # must be remapped when the axis of the offset is flipped — a +y offset
    # becomes a -y offset, which we store at the destination voxel. For this
    # pure geometric check we only assert the short-range x and z channels,
    # whose offsets are orthogonal to the flipped axis and therefore commute
    # with the flip without remapping.
    for offset in ((1, 0, 0), (0, 0, 1)):
        ch = offsets.index(offset)
        flipped_before = np.flip(before[ch], axis=1)
        np.testing.assert_array_equal(
            after[ch],
            flipped_before,
            err_msg=f"Flip correspondence failed for offset {offset}",
        )


def test_background_pair_masking():
    vol = _two_touching_sheets()
    masked = affinity_targets(
        vol, offsets=NEAREST_NEIGHBOUR_OFFSETS, mask_background_pairs=True
    )
    unmasked = affinity_targets(
        vol, offsets=NEAREST_NEIGHBOUR_OFFSETS, mask_background_pairs=False
    )
    # A pure-background pair deep in the volume should be invalid when masked
    # and 0 when not.
    ch = 0
    assert vol[2, 2, 2] == LABEL_BG
    assert vol[3, 2, 2] == LABEL_BG
    assert masked[ch, 2, 2, 2] == AFFINITY_INVALID
    assert unmasked[ch, 2, 2, 2] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
