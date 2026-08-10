#!/usr/bin/env bash
# Fetches and builds the official Vesuvius competition metric package (topometrics-3d,
# https://www.kaggle.com/datasets/sohier/vesuvius-metric-resources) for LOCAL use.
#
# The official demo notebook (sohier/vesuvius-2025-metric-demo) does this on Kaggle via an
# offline `uv pip install --no-index --find-links=wheels` against pre-fetched wheels, because
# Kaggle submission notebooks run with internet disabled. We have real internet access, so we
# just `pip install` the same pinned versions directly from PyPI instead -- same end result,
# no offline-wheel indirection needed. The C++ build step (Betti-Matching-3D) is unchanged.
#
# Prerequisites: `conda activate vesuvius_eval` (see requirements.txt for why this must be a
# dedicated env, not the `vesuvius` training env) and a working Kaggle API token.
set -euo pipefail

CACHE_DIR="${VESUVIUS_EVAL_CACHE:-/mnt/workspace/code/cache}"
RESOURCE_DIR="$CACHE_DIR/topological-metrics-kaggle"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "vesuvius_eval" ]; then
    echo "ERROR: activate the vesuvius_eval conda env first (conda activate vesuvius_eval)." >&2
    exit 1
fi

echo "[1/4] Installing pinned Python dependencies..."
pip install --quiet -r "$(dirname "$0")/../requirements.txt"

if [ ! -d "$RESOURCE_DIR" ]; then
    echo "[2/4] Downloading sohier/vesuvius-metric-resources (~139MB)..."
    mkdir -p "$CACHE_DIR"
    kaggle datasets download sohier/vesuvius-metric-resources -p "$CACHE_DIR" --unzip
else
    echo "[2/4] $RESOURCE_DIR already present, skipping download."
fi

echo "[3/4] Building Betti-Matching-3D C++ extension..."
cd "$RESOURCE_DIR"
chmod +x scripts/setup_submodules.sh scripts/build_betti.sh
make build-betti

echo "[4/4] Installing topometrics-3d (editable)..."
pip install -e . --no-deps --no-build-isolation

python3 -c "import topometrics.leaderboard; print('topometrics import OK:', topometrics.leaderboard.__file__)"
echo "Done."
