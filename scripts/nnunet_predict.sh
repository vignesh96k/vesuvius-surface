#!/usr/bin/env bash
# Step 4 — inference with a trained (or public) nnU-Net checkpoint.
#
# Saves softmax probabilities so thresholds and post-processing can be tuned
# offline instead of by burning leaderboard submissions.
#
# Usage:
#   bash scripts/nnunet_predict.sh --input DIR --output DIR
#   bash scripts/nnunet_predict.sh --input DIR --output DIR --plans nnUNetResEncUNetLPlans

set -euo pipefail

NNUNET_RAW="${nnUNet_raw:-/mnt/workspace/code/nnUNet_raw}"
NNUNET_PREPROCESSED="${nnUNet_preprocessed:-/mnt/workspace/code/nnUNet_preprocessed}"
NNUNET_RESULTS="${nnUNet_results:-/mnt/workspace/code/nnUNet_results}"
DATASET_ID="${DATASET_ID:-100}"
CONFIG="${CONFIG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
PLANS="${PLANS:-nnUNetPlans}"
FOLD="${FOLD:-0}"
CHECKPOINT="${CHECKPOINT:-}"
INPUT=""
OUTPUT=""
SAVE_PROBS=1

export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --trainer) TRAINER="$2"; shift 2 ;;
    --plans) PLANS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --fold) FOLD="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --no-probabilities) SAVE_PROBS=0; shift ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: bash scripts/nnunet_predict.sh --input DIR --output DIR [--trainer T] [--plans P] [--config C] [--fold F] [--checkpoint NAME] [--no-probabilities]"
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  echo "ERROR: --input and --output are required"
  exit 1
fi

if [[ ! -d "$INPUT" ]]; then
  echo "ERROR: input dir not found: $INPUT"
  exit 1
fi

MODEL_DIR="$NNUNET_RESULTS/Dataset$(printf '%03d' "$DATASET_ID")_VesuviusSurface/${TRAINER}__${PLANS}__${CONFIG}"
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model dir not found: $MODEL_DIR"
  echo "Available:"
  ls -1 "$NNUNET_RESULTS/Dataset$(printf '%03d' "$DATASET_ID")_VesuviusSurface" 2>/dev/null || echo "  (none)"
  exit 1
fi

# Published checkpoints often ship only checkpoint_best.pth, while nnU-Net
# defaults to checkpoint_final.pth. Pick whichever is actually present.
FOLD_DIR="$MODEL_DIR/fold_$FOLD"
if [[ -z "$CHECKPOINT" ]]; then
  if [[ -f "$FOLD_DIR/checkpoint_final.pth" ]]; then
    CHECKPOINT="checkpoint_final.pth"
  elif [[ -f "$FOLD_DIR/checkpoint_best.pth" ]]; then
    CHECKPOINT="checkpoint_best.pth"
  else
    echo "ERROR: no checkpoint found in $FOLD_DIR"
    echo "Available:"
    ls -1 "$FOLD_DIR" 2>/dev/null || echo "  (fold dir missing)"
    exit 1
  fi
fi

if [[ ! -f "$FOLD_DIR/$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $FOLD_DIR/$CHECKPOINT"
  echo "Available:"
  ls -1 "$FOLD_DIR" 2>/dev/null || echo "  (fold dir missing)"
  exit 1
fi

mkdir -p "$OUTPUT"

echo "model      = $MODEL_DIR"
echo "fold       = $FOLD"
echo "checkpoint = $CHECKPOINT"
echo "input      = $INPUT"
echo "output     = $OUTPUT"

PROB_FLAG=()
if [[ "$SAVE_PROBS" -eq 1 ]]; then
  PROB_FLAG=(--save_probabilities)
fi

nnUNetv2_predict \
  -i "$INPUT" \
  -o "$OUTPUT" \
  -d "$DATASET_ID" \
  -c "$CONFIG" \
  -f "$FOLD" \
  -tr "$TRAINER" \
  -p "$PLANS" \
  -chk "$CHECKPOINT" \
  "${PROB_FLAG[@]}"

echo
echo "Done. Probability maps (.npz) enable offline threshold + post-processing tuning."
