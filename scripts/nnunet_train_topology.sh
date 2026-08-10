#!/usr/bin/env bash
# Track A — topology training from scratch on 3d_lowres.
#
# Skeleton Recall and/or affinity auxiliary. NOT the m7 track. NOT STU-Net.
#
# Prerequisites (vesuvius / stock nnunetv2 env):
#   1. lowres plans + preprocess:
#        python scripts/make_lowres_plans.py
#        nnUNetv2_preprocess -d 100 -c 3d_lowres -plans_name nnUNetPlans
#   2. authored split (hold out scroll 26010):
#        python scripts/make_scroll_split.py --mode holdout-scroll --val-scroll 26010
#   3. registered trainers:
#        export PYTHONPATH=/mnt/workspace/code/vesuvius-surface/src:$PYTHONPATH
#        python scripts/register_nnunet_trainers.py
#
# Usage:
#   bash scripts/nnunet_train_topology.sh --stage skelrecall
#   bash scripts/nnunet_train_topology.sh --stage affinity --fold 0
#   bash scripts/nnunet_train_topology.sh --stage skelrecall --trainer nnUNetTrainerSkeletonRecall_w2

set -euo pipefail

NNUNET_RAW="${nnUNet_raw:-/mnt/workspace/code/nnUNet_raw}"
NNUNET_PREPROCESSED="${nnUNet_preprocessed:-/mnt/workspace/code/nnUNet_preprocessed}"
# Separate results tree so topology experiments never collide with m7 / STU-Net.
NNUNET_RESULTS="${nnUNet_results:-/mnt/workspace/code/nnUNet_results_topology}"
DATASET_ID="${DATASET_ID:-100}"
DATASET_NAME="Dataset$(printf '%03d' "$DATASET_ID")_VesuviusSurface"
CONFIG="${CONFIG:-3d_lowres}"
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
      echo "Usage: bash scripts/nnunet_train_topology.sh --stage skelrecall|affinity [--trainer T] [--fold F] [--dry-run]"
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

PLANS_FILE="$NNUNET_PREPROCESSED/$DATASET_NAME/${PLANS}.json"
if [[ ! -f "$PLANS_FILE" ]]; then
  echo "ERROR: plans not found: $PLANS_FILE"
  echo "Run scripts/nnunet_setup_and_preprocess.sh then scripts/make_lowres_plans.py"
  exit 1
fi

if ! python -c "
import json,sys
p=json.load(open('$PLANS_FILE'))
sys.exit(0 if '$CONFIG' in p.get('configurations',{}) else 1)
"; then
  echo "ERROR: config '$CONFIG' not in $PLANS_FILE"
  echo "Author it with:"
  echo "  python scripts/make_lowres_plans.py"
  echo "  nnUNetv2_preprocess -d $DATASET_ID -c $CONFIG -plans_name $PLANS"
  exit 1
fi

DATA_ID=$(python -c "
import json
p=json.load(open('$PLANS_FILE'))
c=p['configurations']['$CONFIG']
print(c.get('data_identifier',''))
")
if [[ -n "$DATA_ID" && ! -d "$NNUNET_PREPROCESSED/$DATASET_NAME/$DATA_ID" ]]; then
  echo "ERROR: preprocessed data for $CONFIG not found:"
  echo "  $NNUNET_PREPROCESSED/$DATASET_NAME/$DATA_ID"
  echo "Run:"
  echo "  nnUNetv2_preprocess -d $DATASET_ID -c $CONFIG -plans_name $PLANS"
  exit 1
fi

SPLIT_FILE="$NNUNET_PREPROCESSED/$DATASET_NAME/splits_final.json"
if [[ ! -f "$SPLIT_FILE" ]]; then
  echo "ERROR: no authored split at $SPLIT_FILE"
  echo "Run: python scripts/make_scroll_split.py --mode holdout-scroll --val-scroll 26010"
  exit 1
fi

if [[ -n "$PRETRAINED" ]]; then
  echo "ERROR: Track A is from-scratch only. Do not pass --pretrained."
  echo "STU-Net is Track B (scripts/stunet_finetune.sh). m7 is not used here."
  exit 1
fi

echo "track      : A (topology, from scratch)"
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
from vesuvius_surface.training.trainers import ${TRAINER}
print(f"resolved    : {${TRAINER}}")
PY

CMD=(nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD" -tr "$TRAINER" -p "$PLANS")

echo
echo "${CMD[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "(dry run; nothing launched)"
  exit 0
fi

echo
"${CMD[@]}"
