#!/usr/bin/env bash
# Fine-tune the published m7 checkpoint at a larger patch size.
#
# This is the winning solution's own method: their best ensemble is mostly
# fine-tuned models, and they measured a 192-patch fine-tune at 250 epochs as
# matching a 4000-epoch from-scratch run on private LB (0.614 vs 0.613).
#
# Prerequisite:
#   python scripts/make_finetune_plans.py --patch-size 192
#
# Usage:
#   bash scripts/nnunet_finetune.sh --config 3d_fullres_192
#   bash scripts/nnunet_finetune.sh --config 3d_fullres_192 --epochs 100 --dry-run

set -euo pipefail

NNUNET_RAW="${nnUNet_raw:-/mnt/workspace/code/nnUNet_raw}"
NNUNET_PREPROCESSED="${nnUNet_preprocessed:-/mnt/workspace/code/nnUNet_preprocessed}"
NNUNET_RESULTS="${nnUNet_results:-/mnt/workspace/code/nnUNet_results}"
DATASET_ID="${DATASET_ID:-100}"
DATASET_NAME="Dataset$(printf '%03d' "$DATASET_ID")_VesuviusSurface"
CONFIG="${CONFIG:-3d_fullres_192}"
PLANS="${PLANS:-nnUNetResEncUNetLPlans}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-250}"
PRETRAINED="${PRETRAINED:-/mnt/workspace/code/pretrained/surface_m7_nnunet/fold_0/checkpoint_best.pth}"
DRY_RUN=0

export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --plans) PLANS="$2"; shift 2 ;;
    --fold) FOLD="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --pretrained) PRETRAINED="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: bash scripts/nnunet_finetune.sh [--config C] [--plans P] [--fold F] [--epochs N] [--pretrained PATH] [--dry-run]"
      exit 1
      ;;
  esac
done

PLANS_FILE="$NNUNET_PREPROCESSED/$DATASET_NAME/${PLANS}.json"
if [[ ! -f "$PLANS_FILE" ]]; then
  echo "ERROR: plans not found: $PLANS_FILE"
  echo "Run: python scripts/make_finetune_plans.py --patch-size 192"
  exit 1
fi

if ! python -c "
import json,sys
p=json.load(open('$PLANS_FILE'))
sys.exit(0 if '$CONFIG' in p.get('configurations',{}) else 1)
"; then
  echo "ERROR: config '$CONFIG' not present in $PLANS_FILE"
  echo "Available:"
  python -c "
import json
p=json.load(open('$PLANS_FILE'))
print('  ' + '\n  '.join(sorted(p.get('configurations',{}))))
"
  exit 1
fi

if [[ ! -f "$PRETRAINED" ]]; then
  echo "ERROR: pretrained weights not found: $PRETRAINED"
  echo "Run scripts/fetch_pretrained_m7.py first."
  exit 1
fi

# nnU-Net ships trainer variants for common training lengths; use one when it
# matches so we inherit the standard schedule rather than patching the trainer.
TRAINER="nnUNetTrainer"
if [[ "$EPOCHS" != "1000" ]]; then
  CANDIDATE="nnUNetTrainer_${EPOCHS}epochs"
  if python -c "
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2, os
r = recursive_find_python_class(os.path.join(nnunetv2.__path__[0],'training','nnUNetTrainer'),'$CANDIDATE','nnunetv2.training.nnUNetTrainer')
raise SystemExit(0 if r is not None else 1)
" 2>/dev/null; then
    TRAINER="$CANDIDATE"
  else
    echo "NOTE: no trainer variant '$CANDIDATE'; using $TRAINER (1000 epochs)."
    echo "      Stop early and use checkpoint_best.pth, or pick a supported"
    echo "      length (nnUNetTrainer_250epochs, _100epochs, _50epochs, ...)."
  fi
fi

# Preserving fold 0 keeps our existing holdout meaningful across runs.
echo "dataset     : $DATASET_NAME"
echo "plans       : $PLANS"
echo "config      : $CONFIG"
echo "fold        : $FOLD"
echo "trainer     : $TRAINER"
echo "pretrained  : $PRETRAINED"
echo

CMD=(nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD"
     -tr "$TRAINER" -p "$PLANS" -pretrained_weights "$PRETRAINED")

echo "${CMD[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "(dry run; nothing launched)"
  exit 0
fi

echo
echo "Training starts now. nnU-Net prints per-epoch times to"
echo "  $NNUNET_RESULTS/$DATASET_NAME/${TRAINER}__${PLANS}__${CONFIG}/fold_$FOLD/training_log_*.txt"
echo "Read the first few epochs and multiply by $EPOCHS before walking away."
echo

"${CMD[@]}"
