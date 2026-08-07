#!/usr/bin/env bash
# Launch Stage 1 (Skeleton Recall) or Stage 2a (affinity auxiliary) training.
#
# Prerequisites (in THIS env — do not use the m7 scoring env if it differs):
#   1. authored split (hold out scroll 26010 as the 129-case val set):
#        python scripts/make_scroll_split.py --mode holdout-scroll --val-scroll 26010
#   2. registered trainers:
#        export PYTHONPATH=/mnt/workspace/code/vesuvius-surface/src:$PYTHONPATH
#        python scripts/register_nnunet_trainers.py
#   3. Do NOT initialise from the m7 checkpoint (unverifiable split).
#      From-scratch or STU-Net weights only.
#
# Usage:
#   bash scripts/nnunet_train_topology.sh --stage skelrecall
#   bash scripts/nnunet_train_topology.sh --stage affinity --fold 0
#   bash scripts/nnunet_train_topology.sh --stage skelrecall --trainer nnUNetTrainerSkeletonRecall_w2

set -euo pipefail

NNUNET_RAW="${nnUNet_raw:-/mnt/workspace/code/nnUNet_raw}"
NNUNET_PREPROCESSED="${nnUNet_preprocessed:-/mnt/workspace/code/nnUNet_preprocessed}"
# Separate results tree so topology experiments never collide with m7.
NNUNET_RESULTS="${nnUNet_results:-/mnt/workspace/code/nnUNet_results_topology}"
DATASET_ID="${DATASET_ID:-100}"
DATASET_NAME="Dataset$(printf '%03d' "$DATASET_ID")_VesuviusSurface"
CONFIG="${CONFIG:-3d_fullres}"
PLANS="${PLANS:-nnUNetPlans}"
FOLD="${FOLD:-0}"
STAGE=""
TRAINER=""
PRETRAINED=""
DRY_RUN=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --trainer) TRAINER="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --plans) PLANS="$2"; shift 2 ;;
    --fold) FOLD="$2"; shift 2 ;;
    --pretrained) PRETRAINED="$2"; shift 2 ;;
    --results) NNUNET_RESULTS="$2"; export nnUNet_results="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: bash scripts/nnunet_train_topology.sh --stage skelrecall|affinity [--trainer T] [--fold F] [--pretrained PATH] [--dry-run]"
      exit 1
      ;;
  esac
done

case "$STAGE" in
  skelrecall|skeleton|1)
    STAGE="skelrecall"
    TRAINER="${TRAINER:-nnUNetTrainerSkeletonRecall}"
    ;;
  affinity|2|2a)
    STAGE="affinity"
    TRAINER="${TRAINER:-nnUNetTrainerAffinity}"
    ;;
  "")
    echo "ERROR: --stage skelrecall|affinity is required"
    exit 1
    ;;
  *)
    echo "ERROR: unknown stage '$STAGE' (use skelrecall or affinity)"
    exit 1
    ;;
esac

SPLIT_FILE="$NNUNET_PREPROCESSED/$DATASET_NAME/splits_final.json"
if [[ ! -f "$SPLIT_FILE" ]]; then
  echo "ERROR: no authored split at $SPLIT_FILE"
  echo "Run: python scripts/make_scroll_split.py --mode stratified"
  exit 1
fi

if [[ -n "$PRETRAINED" && "$PRETRAINED" == *"surface_m7"* ]]; then
  echo "ERROR: refusing to initialise from the m7 checkpoint."
  echo "It inherits an unverifiable split and would recontaminate the holdout."
  echo "Use STU-Net weights or omit --pretrained for from-scratch."
  exit 1
fi

echo "stage      : $STAGE"
echo "trainer    : $TRAINER"
echo "dataset    : $DATASET_NAME"
echo "config     : $CONFIG"
echo "plans      : $PLANS"
echo "fold       : $FOLD"
echo "split      : $SPLIT_FILE"
echo "results    : $NNUNET_RESULTS"
echo "PYTHONPATH : $PYTHONPATH"
echo

python - <<PY
from training.trainers import ${TRAINER}
print(f"resolved    : {${TRAINER}}")
PY

CMD=(nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD" -tr "$TRAINER" -p "$PLANS")
if [[ -n "$PRETRAINED" ]]; then
  if [[ ! -f "$PRETRAINED" ]]; then
    echo "ERROR: pretrained weights not found: $PRETRAINED"
    exit 1
  fi
  CMD+=(-pretrained_weights "$PRETRAINED")
  echo "pretrained  : $PRETRAINED"
fi

echo
echo "${CMD[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "(dry run; nothing launched)"
  exit 0
fi

echo
"${CMD[@]}"
