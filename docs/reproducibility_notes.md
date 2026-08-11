# Reproducibility notes — honest gaps, not aspirational claims

This project made one real mistake earlier on that shapes how every claim below is worded:
`vesuvius-surface/README.md` once claimed the public `scrollprize/surface_m7_nnunet`
checkpoint was genuinely `fold_0` of a real cross-validation split, with an elaborate,
well-built reconstruction of which cases it never saw. Checked directly against the
checkpoint's own embedded metadata: `fold='all'`. The 1st-place team's own writeup
independently confirms they trained their nnU-Net baseline on all available data. **The
"fold_0" label was a documentation artifact, not a real split** — a well-built reconstruction
had been mistaken for evidence. Every reproducibility claim below is deliberately conservative
and checkable as a result of that lesson, not because of excess caution for its own sake.

## What is exactly reproducible

- **The metric-guided unmerge novelty layer** (`src/vesuvius_surface/postprocess/unmerge.py`).
  Pure numpy/scipy logic, no external checkpoints needed. `tests/unit/test_unmerge.py` covers
  it directly.
- **The 1st-place postprocessing chain** (`src/vesuvius_surface/postprocess/first_place.py`).
  Same — pure logic, deterministic given an input volume.
- **The official-metric scoring** (`scripts/evaluation/score_model.py`,
  `packages/vesuvius_evaluation/`). Deterministic given predictions + ground truth.
- **Every trainer class's architecture/loss definition** (`src/vesuvius_surface/training/trainers/`).
  Training itself is not bit-exact (no full determinism forced — see below) but the same
  trainer + same data + same seed reproduces the same *class* of result.

## What is approximately reproducible (same recipe, not bit-exact)

- **Every from-scratch training result** (`nnUNetTrainerSeeded` and its variants). Seeded
  (torch/numpy/random), but `cudnn.benchmark=True` / `cudnn.deterministic=False` is left at
  nnU-Net's own default for training speed — full bit-exact determinism was judged not worth
  the training-time cost for a project that only needs run-to-run *comparability*, not
  identical checkpoints. Re-running should land within noise of the reported numbers.
- **The last-layers-only cascade fine-tune** (the one genuinely positive fine-tuning result).
  Same caveat: seeded, not bit-exact-deterministic.
- **The full arunodhayan fine-tune (Phase 3, item 12)** — highpass input
  (`scripts/data_prep/highpass.py`, `build_dataset102_*.py`) + skeleton-recall + affinity loss
  (`nnUNetTrainerSkeletonRecallAffinity`, already a real, committed, tested trainer class),
  applied via `-pretrained_weights` to both the fullres ensemble and the cascade. **Corrected
  from an earlier, wrong version of this note**, which claimed this experiment's code no
  longer existed anywhere — that conclusion came from an insufficiently thorough search (it
  only checked for a dedicated trainer *class* named after the experiment, when the real
  mechanism reused an existing one) and didn't survive a proper one: the real checkpoints
  (`nnUNet_results_ensembleA_ft/`, `nnUNet_results_cascade_ft/`, both under
  `Dataset102_VesuviusSurfaceHighpassOnly`) have `trainer_name=nnUNetTrainerSkeletonRecallAffinity`
  embedded directly in their own metadata — real, checkable evidence, not inferred.
  `README.md` step 9 condenses this negative result to one line; real commands:

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
  # scripts/inference/convert_previous_stage.py):
  nnUNetv2_train 102 3d_cascade_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecallAffinity \
      -pretrained_weights checkpoints/Cascade_fullres_checkpoint_best.pth
  ```

## Known, real gaps — stated plainly, not silently worked around

1. **The clean/TTA-consistent confirmatory run may not be finished.** The fast last-layers
   result (Phase 4, item 14) trained on non-TTA cascade previous-stage data (for speed) but
   was scored against a TTA-generated baseline — a real train/eval mismatch, caught and named
   rather than silently accepted. A second run using TTA-consistent data throughout was built
   and launched specifically to close this gap. Check `experiment_summary.md` for whether it
   completed before this repo's last update; if not, the fast result's direction is likely
   real (the fine-tune only touches 0.07% of parameters, too little capacity to have learned
   to exploit the TTA/non-TTA statistical difference specifically) but not confirmed clean.

2. **`nnunetv2` version pin.** Pinned to `2.8.1` in `environment-train.yml` — the version
   actually installed and used for every real result in this repo, confirmed via
   `pip show nnunetv2` at the time of writing, not inferred from a changelog.

3. **Some older scripts under `scripts/` (predating this restructuring pass) still default
   to this project's original development machine's absolute paths** (e.g.
   `scripts/export_nnunet.py`'s `--data-root` default). These are not broken
   — every one of them is overridable via a CLI flag or environment variable, and the README's
   quickstart shows the override pattern — but they are not yet migrated to the
   `VESUVIUS_DATA_ROOT`-style convention used by the newer scripts/notebooks in this repo
   (`scripts/download_data.sh`, `notebooks/01_dataset_overview.ipynb`,
   `scripts/evaluation/score_model.py`, etc.). Stated here rather than silently left
   inconsistent: a full pass to unify every script onto one convention was judged lower value
   than the consolidation work in this pass, given every affected script already has a working
   override mechanism.

4. **`src/vesuvius_surface/evaluation/`'s wrapper duplication.** `metric_adapter.py` wraps the
   same underlying `topometrics` package that `packages/vesuvius_evaluation` also wraps, via a
   different (array-based vs. path-based) interface. Numerically identical — same
   `compute_leaderboard_score` call, same weights — so this is wrapper-interface duplication,
   not a scoring discrepancy, but it's real duplication left in place because
   `src/vesuvius_surface/evaluation/harness.py` (resumable JSONL scoring, per-scroll
   aggregation) is actively relied on for this repo's real reported numbers and deserved
   dedicated testing time before being rewired, not a rushed change made in the same pass that
   produced those numbers. See `packages/README.md`.

## Why two conda environments, not one

`environment-train.yml` (nnU-Net, torch, this project's own training/postprocessing code) and
`environment-eval.yml` (the official scorer) are deliberately separate. The scorer's pinned
numpy/scipy versions (1.26.4 / 1.15.3) are older than what a modern torch build needs, and
force-installing them into the training environment risks a numpy 2.x↔1.x-class C-ABI break
that can silently corrupt torch/nnU-Net's own compiled extensions. This is not a hypothetical
— see `packages/vesuvius_evaluation/docs/setup.md` for the same reasoning stated at the point
the split was first made. Evaluation only ever reads `.tif` files off disk; it never needs to
run in the same process as training.

## Why nnU-Net's own package name collision doesn't apply anymore

An earlier version of this repo had a top-level `training` package that could collide with
`nnunetv2`'s own internal `nnunetv2.training` subpackage depending on import order (previously
worked around via a pre-import cache trick in a `run_training_wrapper.py`). Renaming to
`vesuvius_surface.training` (this repo's current structure) makes that collision structurally
impossible — different top-level names can't be confused regardless of import order. See the
git history on `src/vesuvius_surface/training/run_training.py` for the full before/after.
