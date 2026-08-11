# Vesuvius Challenge — Surface Detection

3D segmentation of papyrus sheet surfaces in micro-CT scans of carbonized, rolled scrolls
(the [Vesuvius Challenge Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)
Kaggle competition), nnU-Net-based. See `experiment_summary.md` for the full, numbered
history of every experiment, and `research_log.md` for the narrative decision log.

## Results

Two parallel tracks, each escalated through the same four stages — baseline/zero-shot →
add skeleton-recall → add 1st-place postprocessing → add a novelty layer on top. Track A adds
a fifth stage: a second, independent novelty layer (fragment bridging) also tried on top of
pp — not stacked with unmerge, each tested separately against the same A3 baseline:

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

**Final-submission variant**: the A2/A3 skeleton-recall-700ep model above is `fold_0` (trained
on 657 of 786 cases, scroll 26010 held out for LOSO validation). A separate run of the exact
same recipe on `fold=all` (all 786 cases, no held-out split — real submission only, no local
LOSO number is possible for it by construction) scored **0.54920 public / 0.57991 private**
without postprocessing and **0.55360 public / 0.57854 private** with 1st-place pp — both
higher than the fold_0 model's equivalents (A2: 0.54454/0.55773; A3: 0.54063/0.56231),
consistent with more training data helping overall. Within the fold=all pair itself, adding
pp raises public (+0.0044) but costs a small amount of private (-0.0014) — the same
public-up/private-mixed pattern already seen with A2→A3, not a one-off.

## Quickstart

Two conda environments, deliberately kept separate (see `docs/reproducibility_notes.md` for a numpy/scipy conflict):

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
# weights: HuggingFace ziyanhuang/STU-Net -- see docs/checkpoints.md
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
# arunodhayan/cascade-updated is a Kaggle Model, not a plain Dataset -- see
# docs/checkpoints.md for the exact download step; download_weights.sh only wraps
# `kaggle datasets download` and doesn't apply here.
nnUNetv2_train 100 3d_cascade_fullres 0 -p nnUNetResEncUNetMPlans \
    -tr nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs \
    -pretrained_weights checkpoints/Cascade_fullres_checkpoint_best.pth
```

Result: 0.7248 local LOSO vs. 0.7198 zero-shot (**+0.0050**) — the project's one genuinely
positive fine-tuning result.

### 10. 1st-place postprocessing, both lines

A *different* team's technique (they placed 1st, not 3rd — see `docs/attribution.md`),
reimplemented from their writeup: remove small components → per-sheet closing → height-map
gap patching → hole plugging → global fill_holes.

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

This project's own contribution, not from any public source (see `docs/attribution.md`) —
`voi_merge` stays flat across the 1st-place chain because it never severs a bridge fused
*between* two components. Method: erode each component, treat surviving multi-seed splits as
merge candidates, cut via nearest-seed Voronoi tessellation, accept only if the metric
improves.

```bash
python scripts/run_postprocess.py --method unmerge --workers 8 \
    --predictions $nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainerSkeletonRecall_700epochs__nnUNetResEncUNetMPlans__3d_lowres/fold_0/validation \
    --output out/a4 --labels $VESUVIUS_DATA_ROOT/train_labels

python scripts/run_postprocess.py --method unmerge --workers 8 \
    --predictions $nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs__nnUNetResEncUNetMPlans__3d_cascade_fullres/fold_0/validation \
    --output out/b4 --labels $VESUVIUS_DATA_ROOT/train_labels
```

Candidate/accept counts: arunodhayan line — 25/129 candidate cuts, 19/129 accepted; our own
line — 57/129 candidate cuts, 38/129 accepted. Net-neutral on both (see Results table): the
accept gate only requires `delta >= 0.0` per volume, so accepted cuts are individually real
but too small to move a 129-case mean. This corrects an earlier version of this section that
predates a fix to a `voi_alpha` default bug in `evaluation.metric_adapter.score_pair` (was
silently using the metric package's own `1.0` instead of the `0.3` every other reported number
here uses — see that module's docstring for the full account).

### 12. Metric-guided fragment bridging (novelty, on top of pp — the other half of the merge/split gap)

Unmerge (step 11) splits components wrongly fused together; nothing bridges components
wrongly *split apart* — the exact failure step 6's failure-case analysis found ("our
predicted sheet surfaces come out as visibly broken, discontinuous lines"). Method: pair
each component's nearest surface point against every other component's (via `cKDTree`),
accept a pair only if both sides' local surface orientation points back toward the other
(rules out bridging unrelated nearby sheets — that's what unmerge's Voronoi cut targets
instead), connect accepted pairs with a thin bridge, gate the whole volume the same way
unmerge is gated. Grounded in real literature (vessel-reconnection endpoint-pairing
techniques; a real top-10 solution in *this* competition independently used a heavier
orientation-aware technique for the same purpose — see `docs/attribution.md`).

```bash
python scripts/run_postprocess.py --method bridge --workers 8 \
    --predictions $nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainerSkeletonRecall_700epochs__nnUNetResEncUNetMPlans__3d_lowres/fold_0/validation \
    --output out/a5 --labels $VESUVIUS_DATA_ROOT/train_labels

python scripts/run_postprocess.py --method bridge --workers 8 \
    --predictions $nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs__nnUNetResEncUNetMPlans__3d_cascade_fullres/fold_0/validation \
    --output out/b5 --labels $VESUVIUS_DATA_ROOT/train_labels
```

Applied on top of the already-deployed, unconditional 1st-place pp output (not an
oracle-gated intermediate — see `docs/decisions.md` decision 21 for a real methodology bug
caught and fixed while validating this): our line **0.5683 → 0.5691 (+0.0009)**, full
129-case LOSO, 32/129 accepted. Standalone (no pp): 0.5671 → 0.5682 (+0.0011), 40/129
accepted, mean accepted-case gain +0.0035 vs. mean rejected-case would-be loss -0.0083 — a
real, well-behaved signal. Applied unconditionally (no gate) is a net negative either way;
the gate is doing real work, not formality.

arunodhayan line: 129/129 cases have candidate bridges, but only 1/129 accepted — net
0.7363 → 0.7363 (+0.0000), same net-neutral shape as unmerge on this line. Not a
contradiction: this line's control predictions already have far less fragmentation
(`voi_split` 0.85 vs. our own line's 1.4-1.9), so there's little for this specific
technique to fix — it targets exactly the failure mode step 6 diagnosed on our own line,
and correctly finds little of it on the other.

## Repo map

```
src/vesuvius_surface/    training / data / eda / postprocess / evaluation code (pip install -e .)
packages/vesuvius_evaluation/   the official scorer, its own installable package + own conda env
scripts/                  CLI entrypoints: data prep, training, inference, evaluation, downloads
notebooks/                EDA, failure-case analysis, real Kaggle submission notebooks
tests/                    unit/ (CI, no GPU/data needed) and functional/ (manual, needs both)
docs/                     attribution, reproducibility gaps, checkpoints, dataset schema, metric
configs/                  nnU-Net plan overrides, fine-tune config
```

`experiment_summary.md` and `research_log.md` are the two narrative documents — start there for
the full story; this README covers setup and reproduction mechanics.

## Known limitations

Full list with detail in `docs/reproducibility_notes.md`.

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
`docs/reproducibility_notes.md` says which reported numbers came from which tier.

## Sources & attribution

Every public dataset, checkpoint, and technique this repo builds on is cited individually in
`docs/attribution.md` — including a correction worth repeating here: **arunodhayan's public
solution placed 3rd, not 1st.** The 1st-place postprocessing chain (`first_place.py`) is
reimplemented from a different team's writeup entirely.

## Use of AI

Claude Code (Anthropic) was used throughout this project's development, under direct human
direction, not as an autonomous decision-maker:

- **Code**: implementing human-specified designs (e.g. "freeze all but the final decoder stage
  and deep-supervision heads" was a human-directed experiment; Claude wrote the trainer
  subclass), debugging real errors (a Kaggle mount-path inconsistency, a namespace collision,
  an OOM from over-parallelized scoring), and this repo's consolidation pass.
- **Research**: reading public writeups to identify techniques worth trying (the 1st-place
  postprocessing steps; arunodhayan's loss/optimizer recipe, verified against his own source),
  and checking real bugs (e.g. confirming a checkpoint's `fold='all'` metadata directly rather
  than trusting a README's claim).
- **Text**: drafting documentation (this README, `docs/`, `experiment_summary.md`) from
  experiment results and decisions already made.

Every non-obvious decision traces to a human instruction or checkable evidence (a source-code
read, a measured number, a real error message) — not an unexplained AI choice. The human
redirected Claude's proposed directions multiple times over the project when they looked wrong
(see `presentation_notes.md`'s `[PUSHBACK]`-tagged entries for the real record).

## License

MIT for original code in this repository — see `LICENSE`. Checkpoints referenced in
`docs/checkpoints.md` remain under their own original terms.
