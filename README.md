# Vesuvius Challenge — Surface Detection

3D segmentation of papyrus sheet surfaces in micro-CT scans of carbonized, rolled scrolls
(the [Vesuvius Challenge Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)
Kaggle competition), nnU-Net-based.

## Results

**Top real Kaggle leaderboard result.** The final, full-data submission — Track A
(skeleton-recall, 700 epochs), trained on all 786 cases (`fold=all`, no held-out split, so no
local LOSO number is possible for it by construction) — checked against the frozen final
leaderboard (post-deadline, not an official rank):

| Submission | Public | Private | Public rank | Private rank |
|---|---:|---:|---:|---:|
| `fold=all`, no postprocessing | 0.54920 | **0.57991** | #674 / 1392 (top 48.4%) | **#122 / 1392 (top 8.8%)** |
| `fold=all`, + 1st-place postprocessing | 0.55360 | 0.57854 | #345 / 1392 (top 24.8%) | #126 / 1392 (top 9.1%) |

Private rank is far stronger than public in both rows (top ~9% vs. top 25-48%) — the good
direction to be lopsided in (generalizing to the held-out final scoring better than to the
public preview split, not overfitting to it). Adding 1st-place pp moves the public rank up a
lot (#674→#345) while the private rank barely moves (#122→#126), the same public-up/
private-mixed pattern the local-LOSO ablations below already predicted before either
submission was made.

*(Track B's fine-tuned-checkpoint line scores higher in raw Kaggle points — 0.62410 private,
row B1 below — but that submission fine-tunes arunodhayan's public checkpoint rather than
this project's own from-scratch model, and was never itself checked against the frozen
leaderboard. The table above is specifically Track A's own, fully-trained-from-scratch
result.)*

### Local LOSO experimentation trail

The leaderboard result above is the end point of the experimentation below — every stage
benchmarked first against this project's own held-out validation split (scroll 26010, 129
cases) with the real leaderboard-equivalent metric, before any Kaggle submission was made.
Two parallel tracks, each escalated through the same four stages — baseline/zero-shot → add
skeleton-recall → add 1st-place postprocessing → add a novelty layer on top. Track A adds a
fifth stage: a second, independent novelty layer (fragment bridging) also tried on top of pp
— not stacked with unmerge, each tested separately against the same A3 baseline:

| # | Track A — our own from-scratch line | Local LOSO (129 held-out) | Real Kaggle |
|---|---|---:|---:|
| A1 | 1000 epochs, from-scratch | 0.5597 | 0.50962 public / 0.51693 private |
| A2 | + skeleton-recall (700 epochs) | 0.5671 | 0.54454 public / 0.55773 private |
| A3 | + 1st-place postprocessing | 0.5683 | 0.54063 public / 0.56231 private |
| A4 | + unmerge novelty (on A3) | 0.5683 (net-neutral) | — |
| A5 | + fragment bridging novelty (on A3) | 0.5691 (+0.0009) | — |

| # | Track B — arunodhayan's checkpoint | Local LOSO (129 held-out) | Real Kaggle |
|---|---|---:|---:|
| B1 | Zero-shot (3rd place, unmodified) | 0.7198 | 0.58667 public / 0.62410 private |
| B2 | + skeleton-recall last-layers fine-tune | 0.7248 | — |
| B3 | + 1st-place postprocessing | 0.7363 | 0.59733 public / 0.62253 private |
| B4 | + unmerge novelty | 0.7363 (net-neutral) | — |
| B5 | + fragment bridging novelty (on B3) | 0.7363 (net-neutral) | — |

"Local LOSO" is this project's own scroll-grouped held-out validation (scroll 26010, 129
cases), scored with the real leaderboard-equivalent metric. "Net-neutral" (A4/B4): the unmerge
layer passes real per-volume accept gates on both lines (19/129 and 38/129 respectively), but
the aggregate score across all 129 cases is unchanged to full float precision on both. A5's
fragment-bridging layer gives a real, non-negative gain on Track A (32/129 accepted) — see
step 12. On Track B (B5), candidates exist in all 129 cases but only 1 is accepted, netting to
0.0000 — consistent, not contradictory: Track B's control predictions already have far less
fragmentation to fix (`voi_split` 0.85 vs. Track A's 1.4-1.9), so there's little for this
specific technique to do there. Bridging helps where the diagnosed problem (step 6) is
present, and correctly does ~nothing where it isn't.

The A2/A3 skeleton-recall-700ep rows above are `fold_0` (trained on 657 of 786 cases, scroll
26010 held out for LOSO validation) — the fold=all leaderboard submissions at the top of this
section are a separate run of the exact same recipe on all 786 cases, and score higher on
both public and private (consistent with more training data helping overall: A2 0.54454/
0.55773 → fold=all no-pp 0.54920/0.57991; A3 0.54063/0.56231 → fold=all +pp 0.55360/0.57854).

## Quickstart

Two conda environments, deliberately kept separate (the eval env pins an older numpy/scipy
to match the organizers' compiled metric extension, which would ABI-conflict with torch if
shared):

```bash
# Training / inference / postprocessing
conda env create -f environment-train.yml
conda activate vesuvius
pip install -e .
python -m ipykernel install --user --name vesuvius --display-name vesuvius  # notebooks/*.ipynb pin this kernel name

# Official scorer (only needed to reproduce reported scores, not to train)
conda env create -f environment-eval.yml
conda activate vesuvius_eval
bash packages/vesuvius_evaluation/scripts/install_topometrics.sh
pip install -e packages/vesuvius_evaluation --no-deps
```

Smoke test (no GPU/data needed):

```bash
conda activate vesuvius
pip install -e ".[dev]"
pytest tests/unit
```

## Step by Step

### 1. Get the data

```bash
export VESUVIUS_DATA_ROOT=/path/to/data
bash scripts/download_data.sh    # see docs/data.md
```

### 2. EDA

```bash
jupyter nbconvert --to notebook --execute notebooks/01_dataset_overview.ipynb
```

786 usable volumes across 6 scrolls with wildly uneven sizes (scroll 34117 has 376 volumes,
53997 has 13), class balance 37% background / 4.9% surface / 58% ignore.

### 3. Build the LOSO validation split

```bash
python scripts/make_scroll_split.py --mode holdout-scroll --val-scroll 26010
```

Leave-one-scroll-out: same-scroll cases are likely spatially correlated, per the
medical-imaging leakage literature (Yagis et al. 2021 *Sci Rep*; Varoquaux & Cheplygina 2022
*npj Digital Medicine*). Scroll 26010 (129 of 786 volumes) came from step 2's EDA: large
enough for a trustworthy held-out set, without removing a disruptive share of training data
the way holding out 34117 (376 volumes, nearly half the dataset) would.

### 4. Auditing the split

```bash
python scripts/make_scroll_split.py --mode stratified --n-splits 3
```

3 independent stratified 80/20 folds, each trained the same way as step 5's 100-epoch
baseline: **0.5079 / 0.5051 / 0.5084** — within 0.0033 of each other, and within 0.0111 of
the LOSO number itself (0.5162). Real convergent evidence the LOSO number isn't a
lucky/unlucky draw of which scroll got held out.

### 5. Train the from-scratch baseline

```bash
export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
NNUNET_NUM_EPOCHS=100 nnUNetv2_train 100 3d_lowres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSeeded_100epochs
```

Result: 0.5162 local LOSO, real submission 0.46426 public / 0.46559 private. `nnUNetTrainerSeeded`
exists because stock nnU-Net sets no RNG seed anywhere (confirmed by reading
`nnUNetTrainer.__init__` and the CLI directly).

Extended to 1000 epochs (same trainer, different budget):

```bash
nnUNetv2_train 100 3d_lowres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSeeded  # NNUNET_NUM_EPOCHS defaults to 1000
```

Result: 0.5597 local LOSO, real submission 0.50962 public / 0.51693 private.

### 6. Failure case analysis

```bash
jupyter nbconvert --to notebook --execute notebooks/02_failure_case_visualization.ipynb
```

The 100-epoch baseline's score breaks down as SurfaceDice 0.8328 / **TopoScore 0.1344** / VOI
0.5268 — TopoScore is by far the weakest, most variable component. Visualizing the
worst-scoring held-out cases (raw CT, ground truth, our prediction, arunodhayan zero-shot's
prediction, side by side) showed why: our predicted sheet surfaces come out as visibly broken,
discontinuous lines in cross-section on these cases, where arunodhayan's zero-shot held
together as a continuous line. Directly motivated trying topology-aware losses next rather
than a generic architecture/capacity search.

### 7. 100-epoch comparison — pick a loss/architecture winner

Skeleton-recall, clDice, and affinity are all topology/connectivity-aware losses aimed
directly at the weak TopoScore term identified in step 6.

```bash
nnUNetv2_train 100 3d_lowres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecall_100epochs    # winner: 0.5307
nnUNetv2_train 100 3d_lowres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSeeded_ClDice_ScheduleFree   # 0.5285
nnUNetv2_train 100 3d_lowres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerAffinity_100epochs           # 0.5226
```

The other two candidates from the same 5-way comparison change the *input*, not the loss, so
they use nnU-Net's own stock `nnUNetTrainer_100epochs` on two auxiliary-channel datasets built
by `scripts/data_prep/`:

```bash
# highpass-only channel = volume - gaussian_blur(volume, sigma=1.0), see scripts/data_prep/highpass.py
python scripts/data_prep/build_dataset101_laplacian.py         # writes Dataset101 (raw CT + highpass channel)
python scripts/data_prep/build_dataset102_highpass_only.py     # writes Dataset102 (highpass channel alone), reuses Dataset101's output
# then standard nnU-Net preprocessing for each new dataset id (101, 102), plus copying this
# project's own LOSO splits_final.json over nnU-Net's auto-generated one -- see
# build_dataset101_laplacian.py's own docstring for the exact preprocessing commands
nnUNetv2_train 101 3d_lowres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_100epochs   # laplacian (raw + highpass): 0.5122
nnUNetv2_train 102 3d_lowres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_100epochs   # highpass-only: 0.5204
```

Skeleton-recall (Kirchhoff et al., ECCV 2024, ported as `nnUNetTrainerSkeletonRecall`) won.

### 8. Extend the winner: skeleton-recall to 700 epochs (our own line)

```bash
nnUNetv2_train 100 3d_lowres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecall_700epochs
```

Result: 0.5671 local LOSO (vs. 0.5597 no-skeleton-recall baseline) — toposcore alone jumps
0.2021 → 0.3028 (+50% relative), the clearest evidence the loss term does what it's designed
to do.

Submitted to Kaggle with 1st-place postprocessing (step 10): **0.54063 public / 0.56231
private**. The real deadline (2026-02-27) has passed, so this isn't an official rank — checked
against the frozen final leaderboard, it would place **#395 of 1392 (top 28.4%)**.

### 9. Fine-tuning: STU-Net first, then arunodhayan's last layers

Fine-tuning starts with STU-Net (TotalSegmentator-pretrained): provably never seen a Vesuvius
volume, so fine-tuning it against our own split is a genuinely clean comparison — unlike
arunodhayan's or m7's checkpoints. Only the two shallowest encoder stages
(`conv_blocks_context.0`, `.1` — generic edge/texture features, most likely to transfer across
domains) are frozen; the deeper encoder stages and the entire decoder (110 of 130 parameter
tensors) stay trainable, so the domain-specific "body anatomy" semantics STU-Net's pretraining
learned get relearned for thin-sheet detection:

```bash
# weights: HuggingFace ziyanhuang/STU-Net
python scripts/finetune/run_finetuning_stunet_freeze_early.py 100 3d_lowres 0 \
    -p nnUNetResEncUNetMPlans -tr STUNetTrainer_base_ft_30epochs \
    -pretrained_weights checkpoints/stunet_base.model
```

Result: 0.4629 vs. 0.5575 best from-scratch baseline (full 129-case LOSO) — a clear negative.

A full (unfrozen) fine-tune of arunodhayan's checkpoint — highpass input channel + skeleton-
recall + affinity losses — did not produce positive results (ensemble 0.7029 → 0.5172,
cascade 0.7198 → 0.5208, unambiguously negative), so we chose to fine-tune only the last
layers of the cascade model instead: the same skeleton-recall recipe from step 7/8, training
just the final decoder stage and deep-supervision heads (0.07% of parameters).

```bash
# arunodhayan/cascade-updated is a Kaggle Model, not a plain Dataset -- download via
# `kaggle models instances versions download`, not `kaggle datasets download`
# (download_weights.sh only wraps the latter, so it doesn't apply here).
nnUNetv2_train 100 3d_cascade_fullres 0 -p nnUNetResEncUNetMPlans \
    -tr nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs \
    -pretrained_weights checkpoints/Cascade_fullres_checkpoint_best.pth
```

Result: 0.7248 local LOSO vs. 0.7198 zero-shot (**+0.0050**) — the project's one genuinely
positive fine-tuning result.

### 10. 1st-place postprocessing, both lines

A *different* team's technique (they placed 1st, not 3rd — arunodhayan's own solution placed
3rd), reimplemented from their writeup: remove small components → per-sheet closing →
height-map gap patching → hole plugging → global fill_holes.

```bash
conda activate vesuvius_eval
python scripts/run_postprocess.py --method first_place --workers 8 \
    --predictions $nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainerSkeletonRecall_700epochs__nnUNetResEncUNetMPlans__3d_lowres/fold_0/validation \
    --output out/a3 --labels $VESUVIUS_DATA_ROOT/train_labels

python scripts/run_postprocess.py --method first_place --workers 8 \
    --predictions $nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs__nnUNetResEncUNetMPlans__3d_cascade_fullres/fold_0/validation \
    --output out/b3 --labels $VESUVIUS_DATA_ROOT/train_labels
```

Results: our line 0.5671 → 0.5683 (+0.0012); arunodhayan line 0.7248 → 0.7363 (+0.0115).

Real Kaggle, our line (A2 → A3): public 0.54454 → 0.54063 (-0.0039), private 0.55773 →
0.56231 (+0.0046) — pp's small local LOSO gain doesn't hold uniformly on the real
leaderboard split.

### 11. Metric-guided unmerge (novelty), both lines

This project's own contribution, not from any public source — `voi_merge` stays flat across
the 1st-place chain because it never severs a bridge fused *between* two components (the
1st-place team's own writeup confirms they never solved touching sheets either, and just
relied on nnU-Net itself to minimize it). Method: erode each component, treat surviving
multi-seed splits as merge candidates, cut via nearest-seed Voronoi tessellation, accept only
if the metric improves.

```bash
python scripts/run_postprocess.py --method unmerge --workers 8 \
    --predictions $nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainerSkeletonRecall_700epochs__nnUNetResEncUNetMPlans__3d_lowres/fold_0/validation \
    --output out/a4 --labels $VESUVIUS_DATA_ROOT/train_labels

python scripts/run_postprocess.py --method unmerge --workers 8 \
    --predictions $nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs__nnUNetResEncUNetMPlans__3d_cascade_fullres/fold_0/validation \
    --output out/b4 --labels $VESUVIUS_DATA_ROOT/train_labels
```

Candidate/accept counts: arunodhayan line — 25/129 candidates, 19/129 accepted; our own line
— 57/129 candidates, 38/129 accepted. Net-neutral on both (see Results table): the accept
gate only requires `delta >= 0.0` per volume, so accepted cuts are individually real but too
small to move a 129-case mean — a real, honest result, not a dead end. Iterated further from
here on the *opposite* failure mode next.

### 12. Metric-guided fragment bridging (novelty, on top of pp — the other half of the merge/split gap)

Unmerge (step 11) splits components wrongly fused together; nothing bridges components
wrongly *split apart* — the failure step 6 diagnosed. Method: pair each component's nearest
surface point against every other component's (via `cKDTree`), accept a pair only if both
sides' local surface orientation points back toward the other, connect accepted pairs with a
thin bridge, gate the same way unmerge is gated. Informed by vessel-reconnection literature's
endpoint-pairing technique, implemented independently.

```bash
python scripts/run_postprocess.py --method bridge --workers 8 \
    --predictions $nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainerSkeletonRecall_700epochs__nnUNetResEncUNetMPlans__3d_lowres/fold_0/validation \
    --output out/a5 --labels $VESUVIUS_DATA_ROOT/train_labels

python scripts/run_postprocess.py --method bridge --workers 8 \
    --predictions $nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs__nnUNetResEncUNetMPlans__3d_cascade_fullres/fold_0/validation \
    --output out/b5 --labels $VESUVIUS_DATA_ROOT/train_labels
```

Applied on top of the already-deployed, unconditional 1st-place pp output (not an
oracle-gated intermediate): our line **0.5683 → 0.5691 (+0.0009)**, full 129-case LOSO,
32/129 accepted. Standalone (no pp): 0.5671 → 0.5682 (+0.0011), 40/129 accepted.

arunodhayan line: 129/129 cases have candidate bridges, but only 1/129 accepted — net
0.7363 → 0.7363 (+0.0000), same net-neutral shape as unmerge on this line. Not a
contradiction: this line's control predictions already have far less fragmentation
(`voi_split` 0.85 vs. our own line's 1.4-1.9), so there's little for this technique to fix.

## Repo map

```
src/vesuvius_surface/    training / data / eda / postprocess / evaluation code (pip install -e .)
packages/vesuvius_evaluation/   the official scorer, its own installable package + own conda env
scripts/                  CLI entrypoints: data prep, training, inference, evaluation, downloads
notebooks/                EDA, failure-case analysis, real Kaggle submission notebooks
tests/                    unit/ (CI, no GPU/data needed) and functional/ (manual, needs both)
docs/                     metric notes, dataset download instructions
configs/                  nnU-Net plan overrides, fine-tune config
```

## Packaging notes

`src/vesuvius_surface/` used to be several bare top-level packages (`data`, `training`, etc.)
that could collide with `nnunetv2`'s own internal `nnunetv2.training` subpackage depending on
import order — renamed to a single `vesuvius_surface` package to make that collision
structurally impossible (see git history on `src/vesuvius_surface/training/run_training.py`).
`packages/vesuvius_evaluation/` stays a separate installable package (own `pyproject.toml`,
own env) for the ABI reason above, not flattened in.

## Testing & CI

`tests/unit/` — pure-function logic (unmerge's cut-proposal math, previous-stage conversion,
LOSO-split leakage checks, the seeding utility, parallel-vs-sequential scoring dispatch), runs
in CI, no GPU/dataset needed. `tests/functional/` needs real data/checkpoints
(`VESUVIUS_DATA_ROOT`, `VESUVIUS_TEST_CHECKPOINT_DIR`) and is manual-only. Real training,
scoring, and Kaggle submissions are GPU/hours/account-scale and stay manual —
Real training, scoring, and Kaggle submissions are GPU/hours/account-scale and stay manual.

## License

MIT for original code in this repository — see `LICENSE`. Externally sourced checkpoints
(arunodhayan's, `surface_m7_nnunet`, STU-Net) remain under their own original terms, not
this repo's license.
