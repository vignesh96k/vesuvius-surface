# Vesuvius Challenge — Surface Detection Dataset Schema

Competition: [vesuvius-challenge-surface-detection](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection)

This repository targets **3D papyrus surface segmentation** in micro-CT volumes.
It does **not** use the older Ink Detection fragment layout (`surface_volume/`, `inklabels.png`).

## Directory layout

Place the Kaggle extract under `data/` (or point `data.root` in config):

```text
data/
  train.csv
  test.csv
  train_images/
    <id>.tif          # 3D CT volume
    ...
  train_labels/
    <id>.tif          # 3D label volume (same id / shape as image)
    ...
  test_images/
    <id>.tif
    ...
```

Deprecated folders from the competition zip (`deprecated_train_images`, etc.) are ignored.

## Metadata (`train.csv` / `test.csv`)

| Column | Meaning |
|--------|---------|
| `id` | Unique volume identifier; must match the TIFF stem under `*_images/` / `train_labels/` |
| `scroll_id` | Source scroll; many volumes share a scroll. Prefer **scroll-level** holdout for validation so patches from the same scroll do not leak across splits |

## Image volumes

- Format: multi-page / 3D TIFF (`.tif` / `.tiff`)
- Content: micro-CT intensity (typically `uint8` / `uint16`)
- Shape: `(D, H, W)` — **not fixed** across the dataset
- One channel (grayscale)

## Label volumes

Encoded integers:

| Value | Name | Role |
|------:|------|------|
| `0` | background | Non-surface |
| `1` | surface / foreground | Positive class (papyrus surface) |
| `2` | unlabeled / ignore | Do **not** supervise or score these voxels |

Most masks cover only a subset of the volume; class `2` (and unlabeled regions) are common.

Constants in code: `data.schema.LABEL_BG`, `LABEL_SURFACE`, `LABEL_IGNORE`.

## Identity rules

1. Every `id` in `train.csv` must have `train_images/<id>.tif` and `train_labels/<id>.tif`.
2. Image and label shapes must match exactly.
3. Label voxels must be in `{0, 1, 2}` only.
4. Test rows need images only (no public labels for the hidden test set).

Run validation:

```bash
python scripts/validate_dataset.py --data-root data
```

## Download

```bash
pip install kaggle
# Credentials: %USERPROFILE%\.kaggle\kaggle.json (Windows) or ~/.kaggle/kaggle.json
kaggle competitions download -c vesuvius-challenge-surface-detection -p data/
# Unzip so train_images/, train_labels/, train.csv sit directly under data/
```

Approximate size: **~27 GB**.

### Local path (this workspace)

```text
C:\Users\vigne\Downloads\vesuvius-challenge-surface-detection
```

Set `data.root` in `configs/config.yaml` to your extract (default on the training host:
`/mnt/workspace/code/datasets/vesuvius-challenge-surface-detection`).

## Pipeline outputs (for later modeling)

| API | Returns |
|-----|---------|
| `SurfaceVolumeDataset` | Full volume + metadata (EDA / nnU-Net export prep) |
| `SurfacePatchDataset` | 3D patches: `image` `(1, D, H, W)`, `label` `(D, H, W)` |

Ignore index `2` is excluded from valid-patch filtering and should be ignored in loss/metrics later.

## Mapping toward nnU-Net (later)

| This repo | nnU-Net |
|-----------|---------|
| `train_images/<id>.tif` | `imagesTr/<id>_0000.nii.gz` (or TIFF via custom IO) |
| `train_labels/<id>.tif` | `labelsTr/<id>.nii.gz` |
| label `2` | nnU-Net **ignore** label |
| `scroll_id` | fold grouping / custom split JSON |

No nnU-Net conversion is implemented in this pass; case IDs and ignore semantics are intentionally aligned.
