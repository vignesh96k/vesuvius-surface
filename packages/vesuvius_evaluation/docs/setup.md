# Setup

## 1. Create the dedicated conda env

The official scorer's pinned dependency versions (numpy==1.26.4, scipy==1.15.3, etc.) are
*older* than what the `vesuvius` training env already has (numpy 2.4.4 there). Installing
them into `vesuvius` would force a downgrade, most importantly numpy 2.x -> 1.x, a
major-version C-ABI break that risks silently breaking torch/nnU-Net's compiled extensions in
that env. Evaluation never needs to run in the same process as training (it only reads .tif
files), so it gets its own env instead:

```bash
conda create -n vesuvius_eval python=3.11 -y
conda activate vesuvius_eval
```

## 2. Install the official metric package

```bash
conda activate vesuvius_eval
bash scripts/install_topometrics.sh
```

This downloads `sohier/vesuvius-metric-resources` (~139MB, via the Kaggle API -- needs
`~/.kaggle/kaggle.json` credentials) into `/mnt/workspace/code/cache/topological-metrics-kaggle/`,
builds the Betti-Matching-3D C++ extension (needs `g++`/`make`/`cmake` -- already present on
this machine), and installs `topometrics-3d` in editable mode. One-time step; safe to re-run
(skips the download if the cache dir already exists).

## 3. Install this package

```bash
pip install -e . --no-deps
```

## 4. Verify

```bash
python3 -c "import topometrics.leaderboard; import vesuvius_evaluation.official_score; import vesuvius_evaluation.approx_score; print('ok')"
```

## Why two backends?

- **official** (`vesuvius_evaluation.official_score`, wraps `topometrics.leaderboard.compute_leaderboard_score`):
  the actual organizer scoring code from `sohier/vesuvius-2025-metric-demo`. Numerically
  matches the leaderboard. Slow -- roughly 1 minute per 320^3 volume measured locally, which
  lines up with the competition's own "expect to wait several hours" warning for a full
  submission.
- **approx** (`vesuvius_evaluation.approx_score`, ported from
  `jirkaborovec/replicate-lb-score-topology-aware-3d-surface-seg`): same formula and weights,
  but a different (simpler, non-official) Betti-number algorithm, so TopoScore -- and
  therefore the combined score -- will not exactly match `official`. Useful for fast
  iteration only; do not treat its numbers as leaderboard-equivalent.

Both were verified against real data during setup: scoring a ground-truth volume against
itself gives 1.0 on both backends; scoring an all-zero prediction gives 0.0 on both.
