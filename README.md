# Vesuvius Challenge — Surface Detection

3D segmentation of papyrus sheet surfaces in micro-CT scans of carbonized, rolled scrolls
(the [Vesuvius Challenge Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)
Kaggle competition), nnU-Net-based. This repo is a Senior ML Engineer take-home assignment
deliverable — see `experiment_summary.md` for the full, numbered history of every experiment,
and `research_log.md` for the narrative decision log.

## Results

| Model | Local LOSO (129 held-out) | Real Kaggle submission |
|---|---:|---:|
| From-scratch, 1000 epochs | 0.5597 | 0.50962 public / 0.51693 private |
| From-scratch, 700 epochs + skeleton-recall | 0.5671 (+pp: 0.5683) | pending — see `experiment_summary.md` |
| arunodhayan zero-shot (3rd place, unmodified) | 0.7198 | 0.58667 public / 0.62410 private |
| arunodhayan + last-layers fine-tune | 0.7248 (+pp: 0.7363) | not yet submitted |
| Skeleton-recall pipeline validation (partial checkpoint) | — | 0.48812 public / 0.49964 private |

"Local LOSO" is this project's own scroll-grouped held-out validation (scroll 26010, 129
cases), scored with the real leaderboard-equivalent metric — see `docs/reproducibility_notes.md`
for why local and real-leaderboard numbers don't match 1:1. "+pp" = with the 1st-place
postprocessing chain applied. Full numbered history, including 3 negative full-fine-tune
results that came before the positive one above, in `experiment_summary.md`.

**The headline finding isn't the highest number in that table.** Full fine-tuning on top of a
strong pretrained checkpoint regressed every single time it was tried (STU-Net, a 5-way
loss/architecture comparison, and the full arunodhayan fine-tune — see `experiment_summary.md`
Phase 3). Freezing everything except the final decoder stage and deep-supervision heads (0.07%
of parameters trainable) and fine-tuning *only that* is what actually worked. That pattern,
plus a genuinely novel post-processing layer (below), are this project's real contributions —
see `docs/attribution.md` for exactly what's original here versus adapted from public sources.

## Repo map

```
src/vesuvius_surface/    training / data / eda / postprocess / evaluation code (pip install -e .)
packages/vesuvius_evaluation/   the official scorer, its own installable package + own conda env
third_party/              arunodhayan's real fine-tuning driver, vendored verbatim, not ours
scripts/                  CLI entrypoints: data prep, training, inference, evaluation, downloads
notebooks/                EDA, failure-case analysis, real Kaggle submission notebooks
tests/                    unit/ (CI, no GPU/data needed) and functional/ (manual, needs both)
docs/                     attribution, reproducibility gaps, checkpoints, dataset schema, metric
configs/                  nnU-Net plan overrides, fine-tune config
```

`experiment_summary.md` and `research_log.md` are the two narrative documents — start there for
the full story; this README covers setup and reproduction mechanics.

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

Get the data and (optionally) a checkpoint:

```bash
export VESUVIUS_DATA_ROOT=/path/to/data
bash scripts/download_data.sh                                    # see docs/data.md
bash scripts/download_weights.sh <slug> checkpoints/<name>        # see docs/checkpoints.md
```

Smoke test (no GPU/data needed):

```bash
conda activate vesuvius
pip install -e ".[dev]"
pytest tests/unit
```

## Reproducing the experiments

Each phase below maps to `experiment_summary.md`'s numbered items. Reproducibility level is
stated honestly per item — "exact" means re-running produces the same class of result; "gap"
means a real, documented limitation exists (see `docs/reproducibility_notes.md` for the full
list, not summarized further here).

| Phase | What | Entry point | Reproducibility |
|---|---|---|---|
| 0 | EDA, metric discovery, test-set composition probe | `notebooks/01_dataset_overview.ipynb`, `docs/metric.md` | exact |
| 1 | Validation protocol (LOSO + contamination discovery) | `scripts/make_scroll_split.py`, `scripts/verify_split.py` | exact |
| 2 | Three zero-shot baselines (ours, arunodhayan's, m7's) | `scripts/nnunet_train_baseline.sh`; arunodhayan/m7 are downloaded checkpoints, see `docs/checkpoints.md` | exact for ours; zero-shot inference only for the other two |
| 3 | Full fine-tune attempts (all negative) | `src/vesuvius_surface/training/trainers/` (STU-Net, 5-way comparison); `third_party/arunodhayan_source/train.py` (arunodhayan full fine-tune) | exact for trainer-class results; **gap** for the arunodhayan driver — see `docs/reproducibility_notes.md` item 1 |
| 4 | The last-layers pivot (the one positive result) | `nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs`, `nnUNetTrainerSkeletonRecall_700epochs` | exact (seeded, not bit-exact-deterministic) |
| 5 | Postprocessing: 1st-place chain + unmerge novelty | `scripts/run_postprocess.py --method {first_place,unmerge}` | exact |
| 6 | Real Kaggle submissions | `notebooks/submissions/` | the notebooks are exact; the Kaggle-side scoring run itself is obviously not locally reproducible |

Scoring any of the above against ground truth:

```bash
conda activate vesuvius_eval
python scripts/evaluation/score_model.py --gt-dir $VESUVIUS_DATA_ROOT/train_labels \
    --pred-dir my_model=path/to/predictions \
    --splits-file $VESUVIUS_DATA_ROOT/../preprocessed/Dataset100_VesuviusSurface/splits_final.json --fold 0
```

## Known limitations

Full list with detail in `docs/reproducibility_notes.md`. Headline items: the arunodhayan
full-fine-tune result isn't reproducible from a clean script (the real driver was hardcoded,
vendored verbatim rather than presented as something it isn't); no ensembling/TTA-combination
script exists to reproduce how the real A/B ensemble predictions were combined; local LOSO
scores only validate the 71% of real grading that comes from scrolls also seen in training (the
other 29% is genuinely novel scrolls, discovered via the competition's own discussion forum,
not locally simulable).

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
stage conversion, LOSO-split leakage checks, the seeding utility) — runs in CI, no GPU or
dataset needed. `tests/functional/` — needs real data/checkpoints (`VESUVIUS_DATA_ROOT`,
`VESUVIUS_TEST_CHECKPOINT_DIR`), documented as manual-only, not run in CI. Real training,
real leaderboard-equivalent scoring, and real Kaggle submission runs are inherently
GPU/hours/real-account-scale and stay manual — `docs/reproducibility_notes.md` says exactly
which reported numbers came from which of these three tiers.

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

MIT for original code in this repository — see `LICENSE`. Code under `third_party/` and
checkpoints referenced in `docs/checkpoints.md` remain under their own original terms.
