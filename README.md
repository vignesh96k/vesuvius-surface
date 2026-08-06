# Vesuvius Surface Detection

Research codebase for the [Vesuvius Challenge — Surface Detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection) competition (3D papyrus surface segmentation in micro-CT volumes).

## Layout

```text
src/data/     # I/O, schema, patching, PyTorch datasets / loaders
src/eda/      # EDA + modeling insights
src/utils/    # config, seed, logging, checkpoints
scripts/      # CLI entrypoints
configs/      # experiment YAML
notebooks/    # exploration notebooks
docs/         # dataset schema notes
```

Import packages as `data`, `eda`, and `utils` with `src/` on `PYTHONPATH` (scripts do this automatically).

## Documentation

| Document | Contents |
|---|---|
| [research_log.md](research_log.md) | Running narrative: decisions, evidence, corrections, open questions |
| [docs/metric.md](docs/metric.md) | Competition metric and why it drives the approach |
| [docs/dataset_schema.md](docs/dataset_schema.md) | Label encoding, paths, deprecated samples |

## Setup

```bash
conda env create -f environment.yml
conda activate vesuvius
```

Point `data.root` in `configs/config.yaml` at your Kaggle extract (or place it under `data/`). See [`docs/dataset_schema.md`](docs/dataset_schema.md).

## Validate dataset

```bash
python scripts/validate_dataset.py --data-root /path/to/vesuvius-challenge-surface-detection
```

Optional: `--max-volumes N` for a quick scan, `--out reports/` to write CSVs.

## Step 1 — Export to nnU-Net

```bash
python scripts/export_nnunet.py \
  --data-root /mnt/workspace/code/datasets/vesuvius-challenge-surface-detection \
  --output-root /mnt/workspace/code/nnUNet_raw \
  --mode symlink
```

This builds `Dataset100_VesuviusSurface/` with `dataset.json`, symlinked `imagesTr`/`labelsTr`, and `scroll_groups.json`.

## Step 2 — nnU-Net plan & preprocess

```bash
# After a successful full export:
python scripts/export_nnunet.py --mode symlink

# Install + plan (optional integrity check):
bash scripts/nnunet_setup_and_preprocess.sh --verify
# defaults to -c 3d_fullres (use --all to also preprocess 2d)
```

Or manually:

```bash
export nnUNet_raw=/mnt/workspace/code/nnUNet_raw
export nnUNet_preprocessed=/mnt/workspace/code/nnUNet_preprocessed
export nnUNet_results=/mnt/workspace/code/nnUNet_results
pip install nnunetv2
nnUNetv2_plan_and_preprocess -d 100 -c 3d_fullres --verify_dataset_integrity
```

## Step 3 — baseline train

```bash
# Default nnU-Net fold 0 split:
bash scripts/nnunet_train_baseline.sh

# Recommended: hold out one scroll from EDA (e.g. 26002):
bash scripts/nnunet_train_baseline.sh --scroll-val 26002
```

Or manually:

```bash
export nnUNet_raw=/mnt/workspace/code/nnUNet_raw
export nnUNet_preprocessed=/mnt/workspace/code/nnUNet_preprocessed
export nnUNet_results=/mnt/workspace/code/nnUNet_results
nnUNetv2_train 100 3d_fullres 0
```

## Reference baseline — public 1st-place checkpoint

`scrollprize/surface_m7_nnunet` is the nnU-Net component of the winning
ensemble, released under Apache-2.0. We use it as a reference score and as a
source of prediction maps for post-processing development — not as our own
trained model.

```bash
pip install huggingface_hub

# Inspect first (reports whether a splits_final.json ships with it):
python scripts/fetch_pretrained_m7.py

# Then expose it as an nnUNet_results tree:
python scripts/fetch_pretrained_m7.py --install
```

### Leakage-free evaluation subset

The checkpoint is `fold_0` of a 786-case dataset using our case ids. nnU-Net
generates folds deterministically (`KFold(n_splits=5, shuffle=True,
random_state=12345)` over sorted case ids), so the cases it never trained on
can be recovered:

```bash
python scripts/nnunet_folds.py \
  --dataset-dir /mnt/workspace/code/nnUNet_raw/Dataset100_VesuviusSurface \
  --fold 0 \
  --out reports/m7_holdout.json
```

Prefer an authoritative split file when the snapshot ships one:

```bash
python scripts/nnunet_folds.py --splits-json /path/to/splits_final.json
```

Any score for the public checkpoint should be reported on this subset only.

### How far this is actually verified

The snapshot ships no `splits_final.json`, so membership is **inferred**, not
read. Be precise about what is established:

| Claim | Status |
|---|---|
| Checkpoint is `fold_0` | Stated on the model card |
| Same dataset, 786 cases, same labels / `.tif` | Confirmed via `dataset.json` |
| Our sorted case order equals theirs | Testable — `scripts/verify_split.py` |
| They used the *default* seeded split | **Assumed**, not verifiable from the snapshot |

Two checks are available.

**Structural.** `dataset_fingerprint.json` stores per-case geometry in sorted
case-id order. If their sequence matches ours, the K-fold assignment is
identical:

```bash
python scripts/verify_split.py
```

**Empirical.** A model scores higher on data it trained on. Build a sample of
the cases we claim were in training and compare against the holdout — if the
two score the same, our split assumption is probably wrong:

```bash
python scripts/make_subset.py --manifest reports/m7_holdout.json \
  --split train --sample 20 --output /mnt/workspace/code/subsets/m7_trainprobe
```

Until both agree, treat holdout scores as a **relative** basis for comparing
our own configurations, not as a leaderboard estimate.

The reconstructed fold 0 is also unbalanced across scrolls — scroll 44430 gets
zero validation cases and 34117 takes half the subset — so report per-scroll
metrics rather than a single pooled number.

### Build the holdout subset

```bash
python scripts/make_subset.py \
  --manifest reports/m7_holdout.json \
  --output /mnt/workspace/code/subsets/m7_holdout
```

## Step 4 — inference

```bash
bash scripts/nnunet_predict.sh \
  --input /mnt/workspace/code/nnUNet_raw/Dataset100_VesuviusSurface/imagesTs \
  --output /mnt/workspace/code/predictions/m7_test \
  --plans nnUNetResEncUNetLPlans
```

Published checkpoints often ship only `checkpoint_best.pth` while nnU-Net
defaults to `checkpoint_final.pth`; the wrapper detects which exists and
passes `-chk` accordingly (override with `--checkpoint NAME`).

Probability maps are saved by default (`--save_probabilities`) so thresholds
and post-processing can be tuned offline rather than through leaderboard
submissions.

## Step 5 — local scoring

The competition score is **not** volumetric Dice:

```text
Score = 0.30 * TopoScore + 0.35 * SurfaceDice@2.0 + 0.35 * VOI_score
```

Topology and instance structure carry 0.65 of the weight. See
[docs/metric.md](docs/metric.md) for the full definition and why it drives the
post-processing strategy.

Install the official implementation (needs kaggle CLI credentials, git, cmake
and a C++ toolchain for the Betti-Matching-3D submodule):

```bash
bash scripts/setup_metric.sh
```

## Credits

- nnU-Net v2 — [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- Official metric implementation — Kaggle dataset `sohier/vesuvius-metric-resources`
  (wraps Betti-Matching-3D)
- Pretrained `surface_m7_nnunet` weights — [scrollprize on Hugging Face](https://huggingface.co/scrollprize/surface_m7_nnunet), from the
  [1st-place solution](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su)
  by Tony Li, OzanM., Yiheng Wang and PaulG
- Post-processing design follows that writeup (component filtering, per-sheet
  closing, height-map patching, 1-voxel hole plugging, `binary_fill_holes`)
