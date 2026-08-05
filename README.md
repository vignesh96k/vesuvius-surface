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
