# Experiment summary — complete list, chronological, with real numbers

Compiled by reading through `research_log.md` (vesuvius-surface + baselinerun), `presentation_notes.md`,
`NEXT_TASKS.md`, `code_merge_plan.md`, `checkingnight_0808.txt`, and the raw training/scoring logs — not
reconstructed from memory. Intended as presentation-prep source material: every number here traces back
to a real log line, not an estimate.

## Phase 0 — Understanding the problem

1. **EDA**: dataset structure (786 usable volumes, 6 scrolls, wildly uneven scroll sizes — 34117 has 376
   volumes, 53997 has 13), class balance (37% background / 4.9% surface / 58% ignore), sheet thickness
   measurement via 3D distance transform (median 2 voxels — this number matters everywhere downstream).
2. Discovered the eval metric is **not** Dice-like: `0.30×TopoScore + 0.35×SurfaceDice@2vox + 0.35×VOI`.
   Installed the official scorer package (Betti-Matching-3D based) rather than reimplementing it.
3. Reverse-engineered the real hidden test set's composition via the Kaggle discussion forum API (not
   guessed): 71% of test samples are from scroll IDs also present in training, 29% from genuinely novel
   scrolls. Directly shaped validation design; stated as an explicit limitation of every local number
   since (local scores can only validate the easier 71% "seen scroll" majority).

## Phase 1 — Validation protocol

4. Decided LOSO (leave-one-scroll-out, scroll 26010 held out — 657 train / 129 val) as the primary
   protocol, cross-checked with stratified 80/20 k-fold. Three independent stratified folds landed
   within ~0.01 of each other (0.5162 / 0.5079 / 0.5051) — real convergent evidence the split isn't a
   lucky/unlucky draw, not assumed.
5. Discovered a contamination problem: **both** arunodhayan's checkpoint (3rd place) and the actual
   1st-place checkpoint (`scrollprize/surface_m7_nnunet`, public on HuggingFace, Apache-2.0 — a third
   public reference source) were trained on 100% of available data ("we abandoned the traditional K-Fold
   cross-validation... trained directly on the entire dataset" — arunodhayan's own writeup). No clean
   holdout exists for either. Designed a before/after-delta methodology: score original vs. fine-tuned
   on the *same* held-out set — contamination is identical in both conditions, so the delta isolates
   whether fine-tuning helped, even though neither number alone predicts real leaderboard performance.
6. Verified (not assumed) that m7's "fold_0" label is a documentation artifact, not a real split — its
   embedded checkpoint metadata says `fold='all'`, confirmed independently by the 1st-place team's own
   writeup ("We used all available data for training and built our baseline nnU-Net model").

## Phase 2 — Baselines (three separate public reference points)

7. **Our own from-scratch training** (`nnUNetTrainerSeeded`, 3d_lowres):
   - 100-epoch: local LOSO 0.5162, real Kaggle submission 0.46426 public
   - 1000-epoch: real Kaggle submission **0.50962 public / 0.51693 private**, local LOSO 0.5597
     (computed 2026-08-10)
   - Discovered the 1000-epoch checkpoint's EMA pseudo-dice plateaued at epoch 639 and never improved
     again through epoch 999 (360 epochs, over a third of the run, zero measurable gain) — confirmed two
     independent ways: local EMA tracking, and two real Kaggle submissions ("epoch ~640 snapshot" vs
     "final 1000-epoch") scoring within 0.00004 of each other on both public and private leaderboards
     because they were literally the same weights. Directly motivated dropping later long-run epoch
     budgets from 1000 to ~700.
8. **arunodhayan zero-shot** (3rd place, unmodified pipeline: ensemble A+B fullres + cascade refinement):
   real Kaggle submission **0.58667 public / 0.62410 private** — the best real result for most of the
   week. Local LOSO (129 cases, official metric): ensemble 0.7029, full cascade 0.7198.
9. **m7 zero-shot** (1st place, unmodified, `scrollprize/surface_m7_nnunet`): local 3-volume sample
   composite 0.5889 (SurfaceDice 0.9852, VOI 0.3226, TopoScore 0.4372) — used as a reference ceiling
   point, not fine-tuned, same all-data contamination caveat as arunodhayan applies.

## Phase 3 — Full fine-tuning attempts

10. **STU-Net fine-tune** — chosen specifically *because* of the Phase 1 leakage problem: pretrained on
    TotalSegmentator, provably never seen a Vesuvius volume, so fine-tuning it against a split we author
    yields a genuinely clean holdout. Result: clear negative, **stunet_after 0.4629 vs best_baseline
    0.5575** (n=129 full LOSO). Ruled out.
11. **100-epoch "pick a winner" comparison**, 5 candidate techniques head-to-head, full 129-case LOSO:

    | candidate | score |
    |---|---:|
    | **skeletonrecall (winner)** | **0.5307** |
    | cldice_schedulefree | 0.5285 |
    | affinity | 0.5226 |
    | highpassonly | 0.5204 |
    | laplacian | 0.5122 |

    (clDice was also isolated from the RAdamScheduleFree optimizer separately, via a "clDice loss, stock
    SGD" control, to avoid crediting the wrong ingredient for the combined ablation's earlier +0.0123
    gain over baseline.)
12. **Full arunodhayan fine-tune** (highpass input + skeleton-recall + affinity loss, applied to both the
    fullres ensemble and the cascade): clean negative result across every component metric, confirmed
    stable at n=109 and n=129 (not noise):

    | metric | ensemble_before | ensemble_after | cascade_before | cascade_after |
    |---|---:|---:|---:|---:|
    | score | 0.7029 | 0.5172 | 0.7198 | 0.5208 |
    | surface_dice | 0.9771 | 0.7869 | 0.9791 | 0.7827 |
    | toposcore | 0.4288 | 0.1521 | 0.4819 | 0.1643 |
    | voi_score | 0.6636 | 0.5604 | 0.6643 | 0.5644 |

    Recommendation at the time: do not submit — the fine-tuned cascade (0.5208) scored below
    arunodhayan's own unmodified real submission and roughly at our own weaker from-scratch level.
13. **Diagnostic isolating the cause**: skeleton-recall-only (raw CT, no highpass, no affinity, 20
    epochs): **0.5768** — meaningfully better than the broken full combo (0.5172) but still below
    zero-shot (0.7029). Conclusion: highpass/affinity were part of the regression but not the whole
    story — fine-tuning any of these onto a strong pretrained checkpoint carries real risk.

## Phase 4 — The pivot that actually worked (2026-08-09/10)

14. **Last-layers-only cascade fine-tune** (freeze everything except the final decoder stage + deep
    supervision heads — 0.07% of params trainable, 10 epochs) on arunodhayan's cascade. First genuinely
    positive result of the entire project:
    - Quick n=26 diagnostic: +0.0023 score, +0.0082 toposcore over zero-shot
    - Full n=129 real LOSO fold_0: **0.7248 vs 0.7198 zero-shot (+0.0050)**, toposcore 0.5127 vs 0.4819
      (+0.0308), surface_dice flat (0.9790 vs 0.9791)
15. **Our own from-scratch skeleton-recall model**, resumed and extended to 700 epochs (winner-extension
    of the Phase 3 100-epoch winner): local LOSO **0.5671 vs 0.5597** (no-skelrecall 1000ep baseline,
    +0.0074), surface_dice down (0.8109 vs 0.8893, expected tradeoff), toposcore up sharply (0.3028 vs
    0.2021, +0.1007 absolute / ~+50% relative) — the clearest direct evidence that skeleton-recall's loss
    term does what it's designed to do.

## Phase 5 — Postprocessing

16. **1st-place postprocessing** (a *separate* public source from arunodhayan — different team, they
    placed 1st not 3rd; reimplemented independently in `first_place.py`: remove small components →
    per-sheet closing → height-map gap patching → hole plugging → global fill_holes). Applied to both
    lines:
    - Our skeleton-recall line: 0.5683 (+0.0012 over unprocessed)
    - arunodhayan line: 0.7363 (+0.0115 over the fine-tune alone, +0.0165 cumulative over zero-shot)
17. **Novelty: metric-guided unmerge** (`unmerge.py`, genuinely ours, not from any public source).
    Motivated by a direct ablation finding: `voi_merge` sits essentially flat (1.1230→1.1251) across
    every stage of the 1st-place control chain on a 5-case holdout — it repairs holes *inside* a
    component but is merge-blind by construction, never severing a bridge fused *between* two
    components. Method: erode each component, treat surviving multi-seed splits as merge candidates,
    partition the original component via nearest-seed Voronoi tessellation, cut the partition boundary,
    accept only if the official metric improves on that volume (never a per-cut heuristic).
    `erosion_radius` calibrated from a real distance-transform measurement (median sheet half-thickness
    ~1 voxel), not guessed. Real run tonight on the full 129 cases (arunodhayan line): 25/129 volumes had
    candidate cuts, 19/129 accepted by the score gate. Final aggregate score still computing as of this
    writing (both the arunodhayan-line and our-own-line versions are running in the background).

## Phase 6 — Real Kaggle submissions

18. Multiple pipeline-test + real submissions across the project, including tonight's real submission of
    the 700-epoch skeleton-recall model + 1st-place postprocessing (PENDING scoring as of this writing).
    Best real result to date remains arunodhayan's unmodified zero-shot pipeline (0.58667/0.62410) —
    to be updated once tonight's submission and the unmerge-augmented submission land.

## Also real, also worth naming (not wins, but part of the honest record)

- **Ensemble-weight optimization** (Nelder-Mead search over arunodhayan's checkpoint weights) — launched,
  then killed mid-run once it became clear the 129-case LOSO set wasn't actually held out relative to
  those specific checkpoints (`FOLD="all"`), meaning the search would fit a hyperparameter to memorized
  data. Caught and stopped before ever being reported as valid.
- **Clean/TTA-consistent confirmatory re-run** — currently in progress. Addresses a real train/eval
  mismatch: the fast-track last-layers fine-tune (item 14) trained on non-TTA cascade previous-stage
  data but was scored against TTA-generated validation data. Small last-layers capacity makes this
  unlikely to have driven the result, but it's cheap insurance before submitting that model line to
  Kaggle for real.
- **m7 fold-reconstruction verification** — structural (`dataset_fingerprint.json` case-ordering
  comparison) and empirical (train-vs-holdout score probe) attempts to establish whether m7's checkpoint
  had a real, usable holdout. Both approaches, plus the direct metadata read, converged on "no."

## What was missing from the original 8-item draft list

The original list (EDA → validation decision → baseline train+submit → skeleton-recall loss → 1st-place
postprocessing → postprocessing novelty → arunodhayan finetune → STU-Net finetune → highpass bands →
clDice-vs-skeleton-recall) covers roughly phases 2, 5, and part of 3. Missing:
- The metric-discovery and test-set-composition-probing work (phase 0, items 2-3)
- The contamination discovery and before/after-delta methodology design (phase 1, items 5-6)
- The m7 (1st place checkpoint) reference track — a third public source, separate from arunodhayan
- The affinity auxiliary-head experiment (it was part of the same 5-way comparison as clDice/skeleton-recall,
  not a separate item, and not in the original list at all)
- The ensemble-weight optimization attempt (killed, but real)
- **Everything from Phase 4** — the last-layers-only fine-tune is the *only* genuinely positive
  fine-tuning result of the entire project, and it's not on the original list at all
- The real Kaggle submissions made tonight and the clean/TTA confirmatory run in progress
