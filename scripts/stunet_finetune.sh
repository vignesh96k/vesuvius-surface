#!/usr/bin/env bash
# SUPERSEDED: the real Phase 3 item-10 result did not use this script or STU-Net's own
# vendored nnU-Net-2.2 fork runner below -- it used the project's own installed nnunetv2
# via scripts/finetune/run_finetuning_stunet_freeze_early.py (see README.md step 9). Kept
# for reference only.
#
# Fine-tune STU-Net on our dataset against a split we authored.
#
# Run this inside the isolated STU-Net env created by scripts/setup_stunet.sh.
#
# Prerequisites:
#   bash scripts/setup_stunet.sh --model base
#   python scripts/make_scroll_split.py --mode stratified
#
# Usage:
#   bash scripts/stunet_finetune.sh --model base --weights /path/to/base.pth
#   bash scripts/stunet_finetune.sh --model base --weights ... --dry-run

set -euo pipefail

NNUNET_RAW="${nnUNet_raw:-/mnt/workspace/code/nnUNet_raw}"
NNUNET_PREPROCESSED="${nnUNet_preprocessed:-/mnt/workspace/code/nnUNet_preprocessed}"
NNUNET_RESULTS="${nnUNet_results:-/mnt/workspace/code/nnUNet_results_stunet}"
STUNET_ROOT="${STUNET_ROOT:-/mnt/workspace/code/stunet}"
DATASET_ID="${DATASET_ID:-100}"
DATASET_NAME="Dataset$(printf '%03d' "$DATASET_ID")_VesuviusSurface"
CONFIG="${CONFIG:-3d_fullres}"
FOLD="${FOLD:-0}"
MODEL="${MODEL:-base}"
WEIGHTS="${WEIGHTS:-}"
DRY_RUN=0

# Results go to a separate tree so STU-Net runs cannot collide with the m7
# pipeline's nnUNet_results.
export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --fold) FOLD="$2"; shift 2 ;;
    --results) NNUNET_RESULTS="$2"; export nnUNet_results="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: bash scripts/stunet_finetune.sh --model base --weights PATH [--config C] [--fold F] [--dry-run]"
      exit 1
      ;;
  esac
done

case "$MODEL" in
  small|base|large|huge) ;;
  *) echo "ERROR: --model must be small, base, large or huge"; exit 1 ;;
esac

TRAINER="STUNetTrainer_${MODEL}_ft"

if [[ -z "$WEIGHTS" ]]; then
  echo "ERROR: --weights is required (from scripts/setup_stunet.sh)"
  echo "Looked for candidates under $STUNET_ROOT/weights:"
  ls -1 "$STUNET_ROOT/weights" 2>/dev/null || echo "  (none)"
  exit 1
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "ERROR: weights not found: $WEIGHTS"
  exit 1
fi

SPLIT_FILE="$NNUNET_PREPROCESSED/$DATASET_NAME/splits_final.json"
if [[ ! -f "$SPLIT_FILE" ]]; then
  echo "ERROR: no splits_final.json at $SPLIT_FILE"
  echo
  echo "Without an authored split nnU-Net generates its own, which reintroduces"
  echo "exactly the ambiguity we are trying to remove. Run:"
  echo "  python scripts/make_scroll_split.py --mode stratified"
  exit 1
fi

RUNNER="$STUNET_ROOT/STU-Net/nnUNet-2.2/nnunetv2/run/run_finetuning_stunet.py"
if [[ ! -f "$RUNNER" ]]; then
  echo "ERROR: STU-Net fine-tuning runner not found: $RUNNER"
  echo "Run scripts/setup_stunet.sh first."
  exit 1
fi

echo "dataset    : $DATASET_NAME"
echo "config     : $CONFIG"
echo "fold       : $FOLD"
echo "trainer    : $TRAINER"
echo "weights    : $WEIGHTS"
echo "split      : $SPLIT_FILE (authored)"
echo "results    : $NNUNET_RESULTS"
echo

python - "$SPLIT_FILE" "$FOLD" <<'PY'
import json, sys
splits = json.load(open(sys.argv[1]))
fold = int(sys.argv[2])
if fold >= len(splits):
    print(f"ERROR: fold {fold} but split file has {len(splits)} folds")
    raise SystemExit(1)
s = splits[fold]
print(f"fold {fold}: {len(s['train'])} train / {len(s['val'])} val")
overlap = set(s["train"]) & set(s["val"])
if overlap:
    print(f"ERROR: {len(overlap)} case(s) in both train and val: {sorted(overlap)[:5]}")
    raise SystemExit(1)
print("no train/val overlap")
PY

CMD=(python "$RUNNER" "$DATASET_NAME" "$CONFIG" "$FOLD"
     -pretrained_weights "$WEIGHTS" -tr "$TRAINER")

echo
echo "${CMD[*]}"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "(dry run; nothing launched)"
  exit 0
fi

echo
echo "The segmentation head will not transfer — TotalSegmentator has 104 classes"
echo "and we have 3 — so nnU-Net reinitialises it. Encoder/decoder weights load."
echo "Check the first epoch times in the training log before walking away."
echo

"${CMD[@]}"
