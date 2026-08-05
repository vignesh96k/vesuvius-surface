#!/usr/bin/env bash
# Step 2 — env + plan/preprocess for Dataset100_VesuviusSurface
# Usage:
#   bash scripts/nnunet_setup_and_preprocess.sh
#   bash scripts/nnunet_setup_and_preprocess.sh --verify

set -euo pipefail

NNUNET_RAW="${nnUNet_raw:-/mnt/workspace/code/nnUNet_raw}"
NNUNET_PREPROCESSED="${nnUNet_preprocessed:-/mnt/workspace/code/nnUNet_preprocessed}"
NNUNET_RESULTS="${nnUNet_results:-/mnt/workspace/code/nnUNet_results}"
DATASET_ID="${DATASET_ID:-100}"

export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "nnUNet_raw         = $nnUNet_raw"
echo "nnUNet_preprocessed= $nnUNet_preprocessed"
echo "nnUNet_results     = $nnUNet_results"
echo "dataset_id         = $DATASET_ID"

if [[ ! -f "$nnUNet_raw/Dataset$(printf '%03d' "$DATASET_ID")_VesuviusSurface/dataset.json" ]]; then
  echo "ERROR: raw dataset not found. Run step 1 first:"
  echo "  python scripts/export_nnunet.py --mode symlink"
  exit 1
fi

if ! python -c "import nnunetv2" 2>/dev/null; then
  echo "Installing nnunetv2 into current env..."
  pip install nnunetv2
fi

VERIFY_FLAG=()
if [[ "${1:-}" == "--verify" ]]; then
  VERIFY_FLAG=(--verify_dataset_integrity)
fi

echo "Running nnUNetv2_plan_and_preprocess ..."
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" "${VERIFY_FLAG[@]}"

echo
echo "Done. Next (step 3 — short baseline train):"
echo "  nnUNetv2_train $DATASET_ID 3d_fullres 0"
echo "  # or with scroll split: copy splits_final.json into"
echo "  #   \$nnUNet_preprocessed/Dataset$(printf '%03d' "$DATASET_ID")_VesuviusSurface/"
echo "  # before training."
