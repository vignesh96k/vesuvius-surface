#!/usr/bin/env bash
# Step 3 — baseline nnU-Net train (3d_fullres, fold 0)
# Usage:
#   bash scripts/nnunet_train_baseline.sh
#   bash scripts/nnunet_train_baseline.sh --scroll-val 26002

set -euo pipefail

NNUNET_RAW="${nnUNet_raw:-/mnt/workspace/code/nnUNet_raw}"
NNUNET_PREPROCESSED="${nnUNet_preprocessed:-/mnt/workspace/code/nnUNet_preprocessed}"
NNUNET_RESULTS="${nnUNet_results:-/mnt/workspace/code/nnUNet_results}"
DATASET_ID="${DATASET_ID:-100}"
CONFIG="${CONFIG:-3d_fullres}"
FOLD="${FOLD:-0}"
DATASET_DIR="$NNUNET_PREPROCESSED/Dataset$(printf '%03d' "$DATASET_ID")_VesuviusSurface"
RAW_DIR="$NNUNET_RAW/Dataset$(printf '%03d' "$DATASET_ID")_VesuviusSurface"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

VAL_SCROLLS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scroll-val)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        VAL_SCROLLS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: bash scripts/nnunet_train_baseline.sh [--scroll-val SCROLL_ID ...]"
      exit 1
      ;;
  esac
done

echo "nnUNet_raw         = $nnUNet_raw"
echo "nnUNet_preprocessed= $nnUNet_preprocessed"
echo "nnUNet_results     = $nnUNet_results"
echo "train              = dataset $DATASET_ID | $CONFIG | fold $FOLD"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: missing preprocessed dataset at $DATASET_DIR"
  echo "Run step 2 first: bash scripts/nnunet_setup_and_preprocess.sh --verify"
  exit 1
fi

if [[ ${#VAL_SCROLLS[@]} -gt 0 ]]; then
  if [[ ! -f "$RAW_DIR/scroll_groups.json" ]]; then
    echo "ERROR: $RAW_DIR/scroll_groups.json not found (re-run export_nnunet.py)"
    exit 1
  fi
  echo "Installing scroll holdout split (val scrolls: ${VAL_SCROLLS[*]})..."
  python - "$RAW_DIR" "$DATASET_DIR" "${VAL_SCROLLS[@]}" <<'PY'
import sys
from pathlib import Path
from vesuvius_surface.data.nnunet_export import write_scroll_holdout_split

raw = Path(sys.argv[1])
pre = Path(sys.argv[2])
val_scrolls = sys.argv[3:]
out = write_scroll_holdout_split(raw / "scroll_groups.json", pre / "splits_final.json", val_scrolls)
print("Wrote", out)
PY
fi

echo "Starting training..."
echo "Tips: nvidia-smi | watch -n 10; checkpoints under \$nnUNet_results"
nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD"
