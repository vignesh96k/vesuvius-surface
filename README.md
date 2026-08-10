# Vesuvius Challenge — Surface Detection

3D segmentation of papyrus sheet surfaces in micro-CT scans of carbonized, rolled scrolls
(the [Vesuvius Challenge Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)
Kaggle competition), nnU-Net-based. See `experiment_summary.md` for the full, numbered
history of every experiment, and `research_log.md` for the narrative decision log.

## Results

| Model | Local LOSO (129 held-out) | Real Kaggle submission |
|---|---:|---:|
| From-scratch, 1000 epochs | 0.5597 | 0.50962 public / 0.51693 private |
| From-scratch, 700 epochs + skeleton-recall (+1st-place pp) | 0.5671 (+pp: 0.5683) | 0.54063 public / 0.56231 private |
| arunodhayan zero-shot (3rd place, unmodified) | 0.7198 | 0.58667 public / 0.62410 private |
| arunodhayan + skeleton-recall last-layers fine-tune (+1st-place pp) | 0.7248 (+pp: 0.7363) | see `experiment_summary.md` |
| Skeleton-recall pipeline validation (partial checkpoint) | — | 0.48812 public / 0.49964 private |

"Local LOSO" is this project's own scroll-grouped held-out validation (scroll 26010, 129
cases), scored with the real leaderboard-equivalent metric — see `docs/reproducibility_notes.md`
for why local and real-leaderboard numbers don't match 1:1. "+pp" = with the 1st-place
postprocessing chain applied.

## Quickstart

Two conda environments, deliberately kept separate (see `docs/reproducibility_notes.md` for
why — a real numpy/scipy ABI conflict, not a style preference):

```bash
# Training / inference / postprocessing
conda env create -f environment-train.yml
conda activate vesuvius
pip install -e .

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

## The story, step by step

This is the actual order the project happened in, and every step below has a real command
attached — this repo doesn't ask you to take any result on faith.

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
53997 has 13), class balance 37% background / 4.9% surface / 58% ignore, median sheet
thickness 2 voxels (matters for every postprocessing/novelty parameter downstream). Also
where the official metric (`0.30×TopoScore + 0.35×SurfaceDice@2vox + 0.35×VOI`, **not**
Dice-like) and the real hidden test-set composition (71% "familiar scroll", 29% "genuinely
novel scroll", from the competition's own discussion forum) were established — see
`docs/metric.md`.

### 3. Build the LOSO validation split

```bash
python scripts/make_scroll_split.py --mode holdout-scroll --val-scroll 26010
```

Holds out scroll 26010 (129 volumes) entirely — never seen during training. This is what
makes local scores a genuine generalization estimate rather than "another region of a
familiar scroll."

### 4. Validate the split — and why arunodhayan/m7 can't be used for experimentation

```bash
python scripts/verify_split.py
```

This is the step that mattered most for everything after it: checking whether the public
`arunodhayan` (3rd place) and `scrollprize/surface_m7_nnunet` (1st place) checkpoints have a
real, usable held-out fold. They don't — both teams' own writeups state they trained on
100% of the data ("we abandoned the traditional K-Fold cross-validation... trained directly
on the entire dataset"), and m7's checkpoint metadata confirms `fold='all'` directly rather
than the `fold_0` its filename implies. **Consequence: no local score against either
checkpoint is a clean generalization estimate** — which is exactly why step 5 trains a
from-scratch baseline against a split we authored ourselves, rather than trusting either
public checkpoint's own reported numbers.

### 5. Train the from-scratch baseline

```bash
export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
NNUNET_NUM_EPOCHS=100 nnUNetv2_train 100 3d_lowres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSeeded_100epochs
```

Result: 0.5162 local LOSO, real submission 0.46426 public / 0.46559 private. `nnUNetTrainerSeeded`
exists specifically because stock nnU-Net sets no RNG seed anywhere — confirmed by reading
`nnUNetTrainer.__init__` and the `nnUNetv2_train` CLI directly, not assumed.

Extended to 1000 epochs (same trainer, different budget):

```bash
nnUNetv2_train 100 3d_lowres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSeeded  # NNUNET_NUM_EPOCHS defaults to 1000
```

Result: 0.5597 local LOSO, real submission 0.50962 public / 0.51693 private.

### 6. 100-epoch comparison — pick a loss/architecture winner

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

### 7. Extend the winner: skeleton-recall to 700 epochs (our own line)

```bash
nnUNetv2_train 100 3d_lowres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecall_700epochs
```

Result: 0.5671 local LOSO (vs. 0.5597 no-skeleton-recall baseline at the same scale) — the
clearest direct evidence in this project that skeleton-recall's loss term does what it's
designed to do (toposcore 0.2021 → 0.3028, +50% relative).

Submitted to Kaggle with 1st-place postprocessing applied (step 9 below): **0.54063 public /
0.56231 private**. The competition's real deadline (2026-02-27) has long passed, so this
isn't an official rank, but checked against the frozen final leaderboard for reference: the
private score would place **#395 of 1392 (top 28.4%)**.

### 8. Fine-tuning: STU-Net first, then arunodhayan's last layers

Fine-tuning starts with STU-Net (TotalSegmentator-pretrained), chosen specifically because
it has none of the leakage problem from step 4 — it's provably never seen a Vesuvius volume,
so fine-tuning it against our own authored split is a genuinely clean comparison, unlike
arunodhayan's or m7's checkpoints.

```bash
bash scripts/setup_stunet.sh --model base
nnUNetv2_train 100 3d_lowres 0 -p nnUNetResEncUNetMPlans \
    -tr STUNetTrainer_base_ft_30epochs \
    -pretrained_weights checkpoints/stunet_base.model
```

Result: 0.4629 vs. 0.5575 best from-scratch baseline (full 129-case LOSO) — a clear negative.

Next: a full (unfrozen) fine-tune of arunodhayan's own checkpoint, adding a highpass input
channel (`scripts/data_prep/highpass.py`, same transform as step 6) and the skeleton-recall +
affinity losses (`nnUNetTrainerSkeletonRecallAffinity`, already used standalone in step 6):

```bash
python scripts/data_prep/build_dataset102_highpass_only.py   # writes Dataset102 (raw space)
python scripts/data_prep/build_dataset102_fullres.py         # fills in its fullres preprocessed tree
# Dataset102's plans (architecture/spacing) are identical to Dataset100's -- copy rather than
# replan: cp $nnUNet_preprocessed/Dataset100_VesuviusSurface/nnUNetResEncUNetMPlans.json \
#            $nnUNet_preprocessed/Dataset102_VesuviusSurfaceHighpassOnly/

# Ensemble member (one of arunodhayan's two 3d_fullres models):
nnUNetv2_train 102 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecallAffinity \
    -pretrained_weights checkpoints/ensembleA_checkpoint_best.pth

# Cascade (needs a previous-stage input directory first -- see
# scripts/inference/convert_previous_stage.py, same tool used to build any cascade's
# coarse-hint channel):
nnUNetv2_train 102 3d_cascade_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecallAffinity \
    -pretrained_weights checkpoints/Cascade_fullres_checkpoint_best.pth
```

Result: ensemble 0.7029 → 0.5172, cascade 0.7198 → 0.5208 — unambiguously negative on both.
That result is why the next attempt freezes almost everything instead of fine-tuning fully:
applying the same skeleton-recall recipe from step 6/7 to arunodhayan's cascade checkpoint,
but training only the final decoder stage and deep-supervision heads (0.07% of parameters).

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

### 9. 1st-place postprocessing, both lines

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

### 10. Metric-guided unmerge (novelty), both lines

This project's own contribution, not from any public source (see `docs/attribution.md`).
Motivated by a direct ablation finding: `voi_merge` sits essentially flat across every stage
of the 1st-place chain — it repairs holes *inside* a component but is merge-blind by
construction, never severing a bridge fused *between* two components. Erodes each component,
treats surviving multi-seed splits as merge candidates, partitions via nearest-seed Voronoi
tessellation, cuts the boundary, and accepts the cut only if the official metric improves on
that volume.

```bash
python scripts/run_postprocess.py --method unmerge --workers 8 \
    --predictions $nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainerSkeletonRecall_700epochs__nnUNetResEncUNetMPlans__3d_lowres/fold_0/validation \
    --output out/a4 --labels $VESUVIUS_DATA_ROOT/train_labels

python scripts/run_postprocess.py --method unmerge --workers 8 \
    --predictions $nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs__nnUNetResEncUNetMPlans__3d_cascade_fullres/fold_0/validation \
    --output out/b4 --labels $VESUVIUS_DATA_ROOT/train_labels
```

Candidate/accept counts (structural, unaffected by the note below): arunodhayan line —
25/129 volumes had candidate cuts, 19/129 accepted; our own line — 57/129 had candidate
cuts, 38/129 accepted. **Aggregate score numbers are being recomputed as of this writing**:
a real bug was just found and fixed in the metric wrapper this command's scoring step goes
through (`evaluation.metric_adapter.score_pair` was silently using the metric package's own
`voi_alpha=1.0` instead of the `voi_alpha=0.3` every other reported number in this project
uses — see that module's docstring and the git history on it for the full account). The
aggregate score previously reported here for the arunodhayan line was computed before that
fix and is not being repeated here since it's now known to be wrong; see
`experiment_summary.md` Phase 5 item 17 for the corrected numbers once the re-score (on the
already-written predictions, no retraining needed) completes.

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

Full list with detail in `docs/reproducibility_notes.md`. Headline item: local LOSO scores
hold out an *entire* scroll, so they validate the harder ~29% "genuinely novel scroll" portion
of real grading (discovered via the competition's own discussion forum, not locally simulable
otherwise) — the easier ~71% "novel case, familiar scroll" majority isn't directly tested by
LOSO at all, since by construction its held-out scroll has zero presence in training.

## Packaging notes

`src/vesuvius_surface/` used to be several bare top-level packages (`data`, `training`, etc.)
that could collide with `nnunetv2`'s own internal `nnunetv2.training` subpackage depending on
import order. Renamed to a single `vesuvius_surface` package specifically to make that
collision structurally impossible, not just better-documented — see git history on
`src/vesuvius_surface/training/run_training.py`. `packages/vesuvius_evaluation/` is kept as a
separate installable package (own `pyproject.toml`, own env) for the ABI reason above, not
flattened into the main package.

## Testing & CI

`tests/unit/` — pure-function logic (the unmerge novelty layer's cut-proposal math, previous-
stage conversion, LOSO-split leakage checks, the seeding utility, parallel-vs-sequential
scoring dispatch) — runs in CI, no GPU or dataset needed. `tests/functional/` — needs real
data/checkpoints (`VESUVIUS_DATA_ROOT`, `VESUVIUS_TEST_CHECKPOINT_DIR`), documented as
manual-only, not run in CI. Real training, real leaderboard-equivalent scoring, and real
Kaggle submission runs are inherently GPU/hours/real-account-scale and stay manual —
`docs/reproducibility_notes.md` says exactly which reported numbers came from which of these
three tiers.

## Sources & attribution

Every public dataset, checkpoint, and technique this repo builds on is cited individually in
`docs/attribution.md` — including a correction worth repeating here: **arunodhayan's public
solution placed 3rd, not 1st.** The 1st-place postprocessing chain (`first_place.py`) is
reimplemented from a different team's writeup entirely.

## Use of AI

Claude Code (Anthropic) was used throughout this project's development, under direct human
direction, not as an autonomous decision-maker:

- **Code**: implementing designs that were specified up front (e.g. "freeze all but the final
  decoder stage and deep-supervision heads, fine-tune only that" was a human-directed
  experiment design; Claude wrote the trainer subclass implementing it), debugging real errors
  (a Kaggle dataset-mount convention inconsistency, a namespace collision, an OOM from
  over-parallelized scoring), and this repository consolidation/restructuring pass itself.
- **Research**: reading public writeups and notebooks to identify techniques worth trying
  (the 1st-place postprocessing chain's operational steps; arunodhayan's exact loss/optimizer
  recipe, verified against his own source rather than guessed), and investigating real bugs
  (e.g. confirming a checkpoint's `fold='all'` metadata directly rather than trusting a
  README's claim).
- **Text**: drafting documentation (this README, `docs/`, `experiment_summary.md`'s prose)
  from experiment results and decisions that were already made.

Every non-obvious decision has a stated reason traceable to either a human instruction or a
concrete, checkable piece of evidence (a source-code read, a measured number, a real error
message) — not an unexplained AI choice. Where Claude proposed a direction, the human
explicitly redirected it multiple times over the course of this project when the direction
looked wrong (see `presentation_notes.md`'s `[PUSHBACK]`-tagged entries for the real record of
this, kept specifically because it's better evidence than a retrospective summary).

## License

MIT for original code in this repository — see `LICENSE`. Checkpoints referenced in
`docs/checkpoints.md` remain under their own original terms.
