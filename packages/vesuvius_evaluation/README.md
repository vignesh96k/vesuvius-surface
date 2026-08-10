# evaluation

Standalone scorer for the Vesuvius Surface Detection competition metric
(`Score = 0.30*TopoScore + 0.35*SurfaceDice + 0.35*VOI`). Sibling to `baselinerun/`, not
inside it -- this scores any pipeline's `predictions_tiff/` output (or any two directories of
matching-named `.tif` volumes), so it stays reusable rather than tied to one pipeline's code.

## Layout

```
evaluation/
├── pyproject.toml
├── requirements.txt              pinned deps for the official topometrics package
├── scripts/
│   ├── install_topometrics.sh    one-time: fetch + build + install the official scorer
│   └── evaluate.py                CLI: score a predictions dir against a ground-truth dir
├── src/evaluation/
│   ├── official_score.py         wraps topometrics.leaderboard (the real organizer scorer)
│   └── approx_score.py           fast pure-Python approximation, for quick iteration
└── docs/
    ├── setup.md                  env setup, why a dedicated conda env
    └── metric_notes.md           what each sub-metric measures, sources
```

## Quick start

```bash
conda create -n vesuvius_eval python=3.11 -y
conda activate vesuvius_eval
bash scripts/install_topometrics.sh   # one-time, ~139MB download + C++ build
pip install -e . --no-deps

python3 scripts/evaluate.py \
    --pred-dir /path/to/predictions_tiff \
    --gt-dir   /mnt/workspace/code/datasets/vesuvius-challenge-surface-detection/train_labels
```

See `docs/setup.md` for the full explanation, and `docs/metric_notes.md` for what's actually
being computed.

**Important**: this must run in its own `vesuvius_eval` conda env, never inside `vesuvius`
(the training env) -- the official package's pinned dependency versions are older than what
`vesuvius` already has, and installing them there risks downgrading numpy from 2.x to 1.x
under torch/nnU-Net's already-validated setup.
