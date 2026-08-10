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

## Known, real gaps — stated plainly, not silently worked around

1. **The full arunodhayan fine-tune (Phase 3, item 12) has no surviving training code at
   all — corrected from an earlier, wrong version of this note.** An earlier pass of this
   documentation claimed `third_party/arunodhayan_source/train.py` was "the script that
   produced" item 12's checkpoint, with the gap being merely that it's hardcoded/unportable.
   That claim didn't survive a direct check: `train.py` contains zero mentions of
   "highpass"/"skeleton"/"affinity" anywhere, and its active `CUSTOM_TRAINER =
   "nnUNetTrainer_RotFlip_ClDice_prob0.8_arun"` is arunodhayan's *original* ClDice/RotFlip
   recipe — independently confirmed by `docs/attribution.md`'s own, separately-written
   citation of that same recipe for a different purpose (the item 11 ablation,
   `nnUNetTrainerSeeded_ClDice_ScheduleFree`). This script produced the **zero-shot
   checkpoint** (Phase 2, item 8) that item 12 fine-tuned, not item 12's fine-tuned result
   itself.

   The actual training code for item 12 (highpass input + skeleton-recall + affinity loss,
   hand-adapted for that one run, applied to both the fullres ensemble and the cascade) is not
   present anywhere in this repo, in `third_party/`, or in the old `finetune/` working
   directory it was vendored from. `research_log.md` §16 already admitted the real
   orchestration (shell scripts wiring training/prediction/scoring for that overnight run) was
   "not committed verbatim... disposable by construction" — this now appears to have taken the
   trainer *code* itself with it, not just the orchestration around it. This is a real,
   deeper gap than a portability issue: **item 12's exact training code cannot be reconstructed
   without guessing**, and this project already has one real lesson (the m7 `fold='all'`
   incident, above) about the cost of presenting a guessed reconstruction as evidence — so no
   attempt was made to rebuild it.

   What *does* survive and is independently reproducible: the individual loss components as
   clean, tested, from-scratch trainer classes (`nnUNetTrainerSkeletonRecall`,
   `nnUNetTrainerAffinity`, `nnUNetTrainerSkeletonRecallAffinity`), and — more importantly —
   the follow-up diagnostic that isolated skeleton-recall alone without highpass/affinity
   (Phase 3, item 13, `nnUNetTrainerSkeletonRecall_20epochs`, real committed trainer class,
   0.5768). That diagnostic supports the same conclusion item 12 pointed to (fine-tuning a
   strong pretrained checkpoint carries real regression risk) and *is* fully reproducible.
   `src/vesuvius_surface/training/finetune/arunodhayan_cascade_driver.py` is a generic
   `-pretrained_weights` fine-tune entrypoint written from how nnU-Net transfer learning
   works — not extracted from item 12's lost code, and explicitly not a reproduction of its
   highpass+affinity recipe.

   **Recommendation for reporting/presenting this project:** cite item 12's numbers as a real,
   logged, cross-validated observation (cascade and ensemble both regressed consistently, and
   the item-13 diagnostic independently corroborates the direction) — but if asked to defend
   its reproducibility live, point to item 13 (fully reproducible) rather than item 12 itself.

2. **No ensembling/TTA-combination script exists anywhere in this project's history.** The
   real A/B ensemble predictions (weights 0.65/0.35, matching arunodhayan's own hardcoded
   weights) were combined via ad-hoc code during the actual experiments, not a committed,
   reusable tool. `scripts/inference/convert_previous_stage.py` covers the *cascade
   previous-stage conversion* step that consumes an already-combined result, but not the
   combination step itself.

3. **The clean/TTA-consistent confirmatory run may not be finished.** The fast last-layers
   result (Phase 4, item 14) trained on non-TTA cascade previous-stage data (for speed) but
   was scored against a TTA-generated baseline — a real train/eval mismatch, caught and named
   rather than silently accepted. A second run using TTA-consistent data throughout was built
   and launched specifically to close this gap. Check `experiment_summary.md` for whether it
   completed before this repo's last update; if not, the fast result's direction is likely
   real (the fine-tune only touches 0.07% of parameters, too little capacity to have learned
   to exploit the TTA/non-TTA statistical difference specifically) but not confirmed clean.

4. **`nnunetv2` version pin.** Pinned to `2.8.1` in `environment-train.yml` — the version
   actually installed and used for every real result in this repo, confirmed via
   `pip show nnunetv2` at the time of writing, not inferred from a changelog.

5. **Several older scripts under `scripts/` (predating this restructuring pass) still default
   to this project's original development machine's absolute paths** (e.g.
   `scripts/export_nnunet.py`'s `--data-root` default, `scripts/nnunet_predict.sh`'s
   `nnUNet_raw`/`nnUNet_preprocessed`/`nnUNet_results` env-var defaults). These are not broken
   — every one of them is overridable via a CLI flag or environment variable, and the README's
   quickstart shows the override pattern — but they are not yet migrated to the
   `VESUVIUS_DATA_ROOT`-style convention used by the newer scripts/notebooks in this repo
   (`scripts/download_data.sh`, `notebooks/01_dataset_overview.ipynb`,
   `scripts/evaluation/score_model.py`, etc.). Stated here rather than silently left
   inconsistent: a full pass to unify every script onto one convention was judged lower value
   than the consolidation work in this pass, given every affected script already has a working
   override mechanism.

5. **`src/vesuvius_surface/evaluation/`'s wrapper duplication.** `metric_adapter.py` wraps the
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
