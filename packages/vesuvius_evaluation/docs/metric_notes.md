# Vesuvius Surface Detection: the competition metric

```
Score = 0.30 x TopoScore + 0.35 x SurfaceDice@tau=2.0 + 0.35 x VOI_score
```

Computed per test volume, then averaged across all volumes (`score_directory`'s
`mean_score` / the official `score()` function's return value).

## Sub-metrics

- **SurfaceDice@tau**: fraction of predicted-surface voxels within `tau=2` (spacing units) of
  a ground-truth-surface voxel, and vice versa (symmetric). Rewards getting the *boundary*
  right, tolerant of small offsets.
- **VOI (Variation of Information) score**: `1 / (1 + alpha * VOI_total)`, `alpha=0.3`, over
  26-connected components of predicted vs. ground-truth foreground. `VOI_total` is an
  information-theoretic measure of how well components correspond (splits + merges).
  Penalizes over/under-segmentation (e.g. one true sheet predicted as two disconnected
  pieces).
- **TopoScore**: agreement of Betti numbers (b0 = component count, b1 = independent loops/
  tunnels, b2 = enclosed cavities) between prediction and ground truth. Penalizes topological
  errors -- holes, tunnels, spurious handles -- even when voxel-level overlap looks fine.
  This is the metric the official `topometrics` package computes via a compiled
  Betti-Matching-3D persistent-homology library; `approx_score.py`'s version instead derives
  Betti numbers from the Euler characteristic + connected-component counts, a legitimate but
  different (and simpler) computation -- see `docs/setup.md` for why the two backends'
  TopoScore numbers won't exactly agree.

`ignore_label=2` (matches `baselinerun`'s own `dataset.json` labels: 0=background, 1=surface,
2=ignore) -- voxels with this label in the ground truth are excluded from scoring on both
sides.

## Why this connects to the fold="all" discussion

Both backends need real ground truth to score against. Predictions from a model evaluated
under `fold="all"` (baselinerun's default -- see its own docs) were produced by a model that
trained on those exact volumes, so scoring them here would be circular/optimistic, same as
nnU-Net's own internal pseudo-Dice under `fold="all"`. For a meaningful local score, run
inference on cases the model never trained on -- either a proper held-out fold (`fold=0`
instead of `"all"` in `baselinerun`'s config) or the competition's own `test_images/` (for
which we don't have ground truth locally, only the leaderboard does).

## Sources

- Official demo: https://www.kaggle.com/code/sohier/vesuvius-2025-metric-demo
- Official metric package: https://www.kaggle.com/datasets/sohier/vesuvius-metric-resources
- Pure-Python approximation: https://www.kaggle.com/code/jirkaborovec/replicate-lb-score-topology-aware-3d-surface-seg
