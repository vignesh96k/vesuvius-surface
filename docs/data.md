# Dataset

The competition dataset (Kaggle "Vesuvius Challenge Surface Detection") is not committed to
this repo — see `docs/dataset_schema.md` for its full layout, column definitions, and label
encoding.

## Getting it

```bash
export VESUVIUS_DATA_ROOT=/path/to/where/you/want/it
bash scripts/download_data.sh
```

Requires `~/.kaggle/kaggle.json` credentials and that you've accepted the competition's rules
on Kaggle (competition data downloads are gated on this, unlike public Datasets).

## `VESUVIUS_DATA_ROOT`

Every script/notebook in this repo that needs the raw dataset reads this environment variable
rather than a hardcoded path — set it once per shell session:

```bash
export VESUVIUS_DATA_ROOT=/path/to/vesuvius-challenge-surface-detection
```

If unset, scripts fall back to a repo-local `data/` directory. Notebooks fall back the same
way — see `notebooks/01_dataset_overview.ipynb` cell 2 for the exact resolution order.

## What you actually get

786 usable volumes (806 listed in `train.csv`; 20 are competition-retired samples under
`deprecated_train_images/`, not a broken download — see `docs/dataset_schema.md`). ~320×320×320
micro-CT volumes, labels `{background: 0, surface: 1, ignore: 2}`, 6 scrolls with very uneven
per-scroll case counts. Median papyrus sheet thickness measured directly at 2 voxels via 3D
distance transform (`research_log.md` §2) — this number is why the competition's own
`surface_tolerance=2.0` scoring parameter matters so much (§5).
