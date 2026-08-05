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
