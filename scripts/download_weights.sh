#!/usr/bin/env bash
# Downloads a checkpoint (or wheel bundle) from a Kaggle Dataset.
#
# Usage: bash scripts/download_weights.sh <kaggle-dataset-slug> <destination-dir>
# Example: bash scripts/download_weights.sh vigneshk96/vesuvius-1000epoch-checkpoint-v1 checkpoints/1000ep
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/download_weights.sh <kaggle-dataset-slug> <destination-dir>"
  exit 1
fi

SLUG="$1"
DEST="$2"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not found. pip install kaggle, then place credentials at"
  echo "  ~/.kaggle/kaggle.json (chmod 600), or export KAGGLE_USERNAME / KAGGLE_KEY."
  exit 1
fi

mkdir -p "$DEST"
echo "Downloading $SLUG -> $DEST"
kaggle datasets download "$SLUG" -p "$DEST" --unzip
echo "Done."
