# Competition metric

```text
Score = 0.30 * TopoScore + 0.35 * SurfaceDice@tau + 0.35 * VOI_score
```

All three sub-scores are bounded in `[0, 1]`, higher is better.

| Term | Weight | Measures |
|---|---:|---|
| `SurfaceDice@tau` (tau = 2.0) | 0.35 | Boundary proximity within a 2-voxel tolerance |
| `VOI_score` | 0.35 | Instance split / merge over connected components |
| `TopoScore` | 0.30 | Betti matching over components, tunnels, cavities |

## SurfaceDice@tau

Overlap of prediction and ground-truth **surfaces**, counting a boundary voxel
as matched when it lies within `tau = 2.0` of the other surface. This is not
volumetric Dice — it is forgiving to small displacements and unforgiving to
missing or extra structure.

## VOI_score

Variation of Information between the connected-component labelings of
prediction and ground truth (6-connectivity in 3D):

```text
VOI_split = H(GT | Pred)     # over-segmentation
VOI_merge = H(Pred | GT)     # under-segmentation
VOI_total = VOI_split + VOI_merge
VOI_score = 1 / (1 + alpha * VOI_total),  alpha = 0.3
```

Two sheets merged into one component is penalised here, which is why touching
sheets are the dominant failure mode in this competition.

## TopoScore

Betti-number matching from algebraic topology, comparing topological features
per homology dimension: `k=0` components, `k=1` tunnels/handles, `k=2`
cavities. A per-dimension topological F1 is computed from matched features and
averaged over the active dimensions.

## Why this shapes the approach

Topology and instance structure carry **0.65** of the weight (VOI 0.35 +
TopoScore 0.30); only 0.35 rewards geometric accuracy, and that within a
2-voxel tolerance. Measured median sheet thickness is ~2 voxels, so the
tolerance is the same order as the sheet — sub-voxel boundary precision is not
where the points are.

Volumetric Dice appears nowhere in the score. A change that improves Dice but
merges two touching sheets can lower the score.

This matches the 1st-place ablation, where every gain came from topological
post-processing:

| Step | Public | Private |
|---|---:|---:|
| No post-processing | .572 | .596 |
| Remove small components | .586 | .614 |
| Plug small holes | .598 | .622 |
| Patch large holes | .601 | .625 |
| Binary closing | .606 | .627 |
| Fill holes | .606 | .627 |

By the winners' own account threshold tuning was worth "1e-3 level" by
comparison.

## Local scoring

Do not reimplement this — Betti matching is easy to get subtly wrong. The
official implementation ships as the Kaggle dataset
`sohier/vesuvius-metric-resources` (package `topological-metrics-kaggle`), and
requires the Betti-Matching-3D C++ submodule to be compiled:

```bash
bash scripts/setup_metric.sh
```

Scoring is slow — the competition warns a full run can take hours — so budget
accordingly and prefer resumable, per-case scoring.

Note that the 1st-place team had **no local validation set** and tuned their
threshold by spending submissions. A working offline scorer is therefore a
genuine advantage rather than a convenience.

## Sources

- Competition evaluation page (weights, tolerance, alpha, connectivity)
- [1st place writeup](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su) (post-processing ablation)
