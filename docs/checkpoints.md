# Checkpoints

No trained weights are committed to this repo (some are hundreds of MB to low GB — git is the
wrong place for them). Every checkpoint referenced by a real reported number in
`experiment_summary.md` is listed here with where it actually lives.

**Before this repo goes public, these Kaggle datasets need to be switched from private to
public** (same requirement as the repo itself — see the assignment brief's "make your
repository public before you give your presentation"), or reviewers won't be able to fetch
them. `scripts/download_weights.sh` assumes they're public.

## Public, third-party checkpoints (not ours — see docs/attribution.md)

| Checkpoint | Source | Used for |
|---|---|---|
| arunodhayan's cascade (2 fullres models + cascade refinement) | Kaggle datasets/models `arunodhayan/cascade`, `arunodhayan/cascade-updated` | Zero-shot reference (Phase 2, item 8); base for every fine-tune attempt in Phase 3/4 |
| `scrollprize/surface_m7_nnunet` | HuggingFace, Apache-2.0 | Zero-shot reference only (Phase 2, item 9) |
| STU-Net-B (TotalSegmentator-pretrained, 58M params) | HuggingFace `ziyanhuang/STU-Net`, shortest `base`-matching `.model` file (`huggingface_hub.hf_hub_download`) | Base for the fine-tune negative result (Phase 3, item 10) — see `scripts/finetune/run_finetuning_stunet_freeze_early.py` |

## This project's own trained checkpoints (Kaggle Datasets, private as of writing)

| Kaggle dataset slug | What it is | Result it produced |
|---|---|---|
| `vigneshk96/vesuvius-100epoch-checkpoint-v1` | 100-epoch from-scratch baseline | 0.5162 local LOSO, real submission 0.46426/0.46559 |
| `vigneshk96/vesuvius-1000epoch-checkpoint-v1` | 1000-epoch from-scratch (`nnUNetTrainerSeeded`) | Real submission 0.50962/0.51693; local LOSO 0.5597 |
| `vigneshk96/vesuvius-1000ep-partial-ep640best-checkpoint-v1` | Same run's `checkpoint_best` at epoch 639 — where EMA pseudo-dice plateaued and never improved again through epoch 999 | Confirmed via two real submissions scoring within 0.00004 of each other (same weights) |
| `vigneshk96/vesuvius-skelrecall700-checkpoint-v1` | 700-epoch from-scratch skeleton-recall (winner-extension of the Phase 3 5-way comparison's winner) | 0.5671 local LOSO (+1st-place pp: 0.5683); real submission (see below) |
| `vigneshk96/vesuvius-skelrecall700-partial-ep200-checkpoint-v1` | Same run at ~epoch 200 — used to pipeline-test the submission notebook end-to-end before the real checkpoint existed | Real submission public 0.48812 / private 0.49964 (validates the full inference+postprocessing pipeline works; not the final reported model) |

The last-layers-only cascade fine-tune checkpoint (Phase 4, item 14 — the one genuinely
positive fine-tuning result) and the unmerge-augmented models are not yet uploaded as separate
Kaggle datasets as of this writing; their predictions/scores exist locally
(`experiment_summary.md`) but the checkpoint itself should be uploaded following the same
pattern as the entries above before being cited as independently downloadable.

## Offline dependency bundles (Kaggle kernel `dataset_sources`, not checkpoints)

| Slug | What it is |
|---|---|
| `vigneshk96/vesuvius-nnunet-wheels-v3` | Offline pip wheel bundle for `nnunetv2==2.8.1` + deps, cp312-tagged for Kaggle's actual kernel Python (an earlier cp311 build failed — see git history) |

## Downloading

```bash
bash scripts/download_weights.sh <slug> <destination-dir>
```

Thin wrapper over `kaggle datasets download`; see that script for exact usage. Requires
`~/.kaggle/kaggle.json` credentials (`pip install kaggle` first).
