# Attribution

Every external source this project builds on, cited individually. Per the assignment's own
rule: "it is usual to continue the work of other persons; it is not permitted to show their
work as your work." Where this project fine-tunes or reimplements something, that's stated
explicitly below, not implied.

## Public checkpoints / pretrained weights

| Source | What it is | Placement / license | Used how |
|---|---|---|---|
| **arunodhayan's public solution** ([`arunodhayan/cascade`](https://www.kaggle.com/) and `arunodhayan/cascade-updated` Kaggle datasets/models) | **3rd place** in the competition. A 2-model fullres ensemble (weights 0.65/0.35) feeding a `3d_cascade_fullres` refinement stage. | Public Kaggle checkpoint, downloaded — see `docs/checkpoints.md`. His own training driver (a 1140-line, hardcoded, notebook-derived script) was read directly to extract his real loss/optimizer recipe (row below), but isn't vendored in this repo — see `docs/decisions.md` decision 16 for why a direct citation replaced vendoring the full script. | Zero-shot reference score (Phase 2, item 8); base for the full fine-tune attempt (Phase 3, item 12, negative); base for the last-layers-only fine-tune (Phase 4, item 14, the one positive fine-tuning result); base for the metric-guided unmerge novelty layer's real run. |
| **`scrollprize/surface_m7_nnunet`** (HuggingFace) | The actual **1st place** solution's own trained checkpoint. | Apache-2.0, public on HuggingFace. | Zero-shot reference point only (Phase 2, item 9) — never fine-tuned, same all-data-training caveat as arunodhayan's checkpoint applies (see `docs/reproducibility_notes.md`). |
| **STU-Net** (TotalSegmentator-pretrained) | A different pretrained CNN backbone family, architecturally distinct from nnU-Net's own. | Public weights, see original STU-Net repo. | Fine-tuned as a deliberately leak-free comparison point (chosen specifically because it's provably never seen a Vesuvius volume) — negative result (Phase 3, item 10). |

## Public code / techniques (not weights)

| Source | What it is | Used how |
|---|---|---|
| **`jirkaborovec`'s Kaggle scaffold notebook** (`surface-nnunet-training-inference-with-2xt4`) | A from-scratch nnU-Net training pipeline scaffold. | Faithfully ported, function-by-function, into a clean modular pipeline (the from-scratch training track absorbed into this repo — `nnUNetTrainerSeeded` and related trainers). Not a checkpoint; code structure only. |
| **The 1st-place team's own writeup** (a *different* team from arunodhayan — see note below) | Describes their post-processing chain: remove small components → per-sheet closing → height-map gap patching → hole plugging → global fill_holes. | Reimplemented independently in `src/vesuvius_surface/postprocess/first_place.py`, following the writeup's description and the mechanics detail from a public reimplementation (`bshepp/volumen`, credited in-module) for the height-map/LUT specifics not fully spelled out in the writeup itself. |
| **MIC-DKFZ/Skeleton-Recall** (Kirchhoff et al., ECCV 2024) | Skeleton recall auxiliary loss for topology-aware segmentation. | Ported to run against a stock `nnunetv2` install (the original targets their own fork) — `src/vesuvius_surface/training/losses.py`, `skeleton.py`, and the `nnUNetTrainerSkeletonRecall*` trainer family. |
| **arunodhayan's loss+optimizer recipe** (DC+CE + 0.2·clDice, RAdamScheduleFree, no LR schedule) | His documented recipe, read directly from his own public training script, not guessed. | Replicated as a controlled ablation (`nnUNetTrainerSeeded_ClDice_ScheduleFree`) on this project's own leakage-free split, to isolate whether the recipe itself helps independent of his checkpoint/data. Lost the 100-epoch comparison to skeleton-recall (0.5285 vs. 0.5307). |
| **Organizers' official metric** (`sohier/vesuvius-metric-resources` Kaggle dataset, wrapping `topometrics`) | The real leaderboard-equivalent scorer (Betti-Matching-3D based). | Installed and called with its own defaults, never reimplemented — `packages/vesuvius_evaluation/` and `src/vesuvius_surface/evaluation/`. |
| **`sohier/vesuvius-2025-metric-demo`** | Organizers' own demo notebook for the metric. | `packages/vesuvius_evaluation/src/vesuvius_evaluation/official_score.py`'s `load_volume`/`score_single_tif` are ported near-verbatim (same params/defaults), with two purely operational deviations (documented in that file's own docstring: no offline-install wrapper since we have real internet, `tifffile` instead of `PIL` for I/O — neither changes scored values). |
| **`jirkaborovec/replicate-lb-score-topology-aware-3d-surface-seg`** | A fast, non-official approximate Betti-number scorer. | Ported as the `approx` backend (`packages/vesuvius_evaluation/src/vesuvius_evaluation/approx_score.py`) — explicitly documented as not leaderboard-equivalent, for fast iteration only. |
| **Vascular-segmentation gap-reconnection literature** ("Restoring Connectivity in Vascular Segmentation using a Learned Post-Processing Model", arXiv:2404.10506; "Retinal blood vessel segmentation by using the MS-LSDNet network and geometric skeleton reconnection method", ScienceDirect) | The general technique family — find ruptured/terminal points on separate fragments, test directional alignment between candidate pairs, connect the geometrically-best pairs. | Not ported code — the general idea (endpoint pairing + directional-ray test) informed the design of the metric-guided fragment-bridging novelty layer (`src/vesuvius_surface/postprocess/bridge.py`), implemented independently with this project's own geometry (nearest-surface-point pairing, not a skeleton graph — see `docs/decisions.md` decision 20 for why). |

## A correction worth stating plainly

**arunodhayan's public solution placed 3rd, not 1st.** The 1st-place postprocessing chain
(`first_place.py`) is reimplemented from a *separate, different* team's writeup — it has
nothing to do with arunodhayan. These two sources were briefly conflated once during this
project's own development (caught and corrected — see `docs/reproducibility_notes.md`). Kept
here as an explicit, permanent note so the distinction stays correct going forward.

## What is genuinely this project's own contribution

Stated explicitly, for contrast with everything above:

- **The metric-guided "unmerge" post-processing layer** (`src/vesuvius_surface/postprocess/unmerge.py`)
  — not from any public source. Targets a real, measured blind spot in the reimplemented
  1st-place chain (it never severs a bridge fused between two components; `voi_merge` sits
  flat across every stage of that chain in this project's own ablation).
- **The metric-guided fragment-bridging post-processing layer** (`src/vesuvius_surface/postprocess/bridge.py`)
  — the opposite failure mode from unmerge (components wrongly split apart, not wrongly
  fused), motivated by this project's own failure-case analysis and a real A2-vs-A3 Kaggle
  comparison. Informed by the general endpoint-pairing technique family from vessel-
  reconnection literature (see attribution row above), implemented independently with this
  project's own nearest-surface-point geometry, not ported code. Real, validated gain on top
  of the already-deployed, unconditional 1st-place pp stage: 0.5683 → 0.5691 (+0.0009,
  full 129-case LOSO, gated the same way unmerge is).
- **The last-layers-only cascade fine-tune strategy** (freeze all but the final decoder stage
  + deep-supervision heads, 0.07% of parameters trainable) — the one genuinely positive
  fine-tuning result of the project, arrived at only after three separate full-fine-tune
  attempts (STU-Net, the 5-way loss/architecture comparison, the full arunodhayan fine-tune)
  all regressed. This specific "freeze almost everything" strategy is this project's own
  response to that pattern, not copied from any of the sources above.
- **The contamination discovery and before/after-delta validation methodology** — recognizing
  that both arunodhayan's and m7's checkpoints were trained on 100% of available data (no
  clean holdout exists for either), and designing a delta-based comparison to salvage a valid
  *relative* measurement despite that.
- **The Kaggle test-set composition probe** — reverse-engineering the real hidden test set's
  71%/29% seen-scroll/novel-scroll composition via the competition's own discussion forum API,
  rather than assuming.
- **The failure-case visualization analysis** (`notebooks/02_failure_case_visualization.ipynb`)
  — an original comparison finding that 5 of the 10 worst-scoring cases overlap between two
  independently-trained models, suggesting a shared, model-independent difficulty rather than
  a model-specific weakness.
