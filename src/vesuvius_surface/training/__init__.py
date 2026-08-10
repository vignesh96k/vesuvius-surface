"""Topology-aware auxiliary training for nnU-Net.

Two auxiliary objectives live here, both aimed at the 0.65 of the competition
score that is topological (VOI + TopoScore) rather than geometric:

``skeleton``, ``losses.SoftSkeletonRecallLoss``
    Stage 1. Skeleton Recall (Kirchhoff et al., ECCV 2024) — a soft recall term
    on the tubed skeleton of the surface, which penalises breaks in thin
    structures far more than a volumetric Dice does.

``affinity``, ``network.attach_affinity_head``
    Stage 2a. An auxiliary affinity head on the full-resolution decoder
    feature map. Affinities are the standard connectomics representation for
    the split/merge errors VOI measures; Funke et al. (TPAMI 2018) showed that
    long-range affinities help even when used only as an auxiliary loss.

Nothing in this package is imported by ``src/data`` or ``src/evaluation``, and
nothing here assumes a particular conda environment. The only hard requirement
is a working ``nnunetv2`` in the active environment; see :mod:`training.compat`
for how missing pieces are reported.
"""

from __future__ import annotations

__all__ = [
    "affinity",
    "compat",
    "losses",
    "network",
    "skeleton",
    "transforms",
]
