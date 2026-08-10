#!/bin/bash
# Generic checkpoint-history archiver for any nnU-Net training run.
#
# nnU-Net only ever keeps ONE "latest" checkpoint (overwritten every save_every=50 epochs)
# and ONE "best" checkpoint (overwritten whenever EMA pseudo-dice improves) -- no history is
# kept by default. This watches a fold's output dir and, whenever either file's mtime
# changes, copies it into a checkpoint_archive/ subdir under a name that encodes its REAL
# epoch number -- read directly from the checkpoint's own 'current_epoch' field via
# torch.load, not inferred from log timestamps (nnUNetTrainer.save_checkpoint() stores
# 'current_epoch': self.current_epoch + 1 in every checkpoint dict -- verified from source).
#
# Usage: archive_checkpoints.sh <fold_output_dir> <training_pid> <label>
# Runs until <training_pid> exits, then does one final check (also picks up
# checkpoint_final.pth if present) before exiting itself.
set -uo pipefail
FOLD_DIR="$1"
TRAIN_PID="$2"
LABEL="${3:-run}"
ARCHIVE_DIR="$FOLD_DIR/checkpoint_archive"
mkdir -p "$ARCHIVE_DIR"

# Activates the training conda env if one isn't already active. Assumes `conda` is already
# on PATH / initialized in the calling shell (e.g. via `conda init`) -- no hardcoded
# installation path, unlike this script's original scratchpad version.
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  conda activate vesuvius
fi

get_epoch() {
  python3 -c "
import torch
try:
    ck = torch.load('$1', map_location='cpu', weights_only=False)
    print(ck.get('current_epoch', 'unknown'))
except Exception:
    print('error')
" 2>/dev/null
}

archive_if_changed() {
  local src="$1" tag="$2"
  if [ ! -f "$src" ]; then return; fi
  local mtime
  mtime=$(stat -c %Y "$src" 2>/dev/null || echo 0)
  local state_file="$ARCHIVE_DIR/.last_${tag}_mtime"
  local last
  last=$(cat "$state_file" 2>/dev/null || echo 0)
  if [ "$mtime" != "$last" ]; then
    local epoch
    epoch=$(get_epoch "$src")
    local dest="$ARCHIVE_DIR/checkpoint_${tag}_epoch${epoch}_${mtime}.pth"
    if [ ! -f "$dest" ]; then
      cp "$src" "$dest"
      echo "$(date '+%Y-%m-%d %H:%M:%S') [$LABEL] archived $tag (epoch $epoch) -> $(basename "$dest")"
    fi
    echo "$mtime" > "$state_file"
  fi
}

echo "$(date '+%Y-%m-%d %H:%M:%S') [$LABEL] archiver started, watching $FOLD_DIR (training pid $TRAIN_PID)"
while true; do
  archive_if_changed "$FOLD_DIR/checkpoint_latest.pth" "latest"
  archive_if_changed "$FOLD_DIR/checkpoint_best.pth" "best"
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$LABEL] training pid $TRAIN_PID no longer running -- final check then exit"
    sleep 5
    archive_if_changed "$FOLD_DIR/checkpoint_latest.pth" "latest"
    archive_if_changed "$FOLD_DIR/checkpoint_best.pth" "best"
    archive_if_changed "$FOLD_DIR/checkpoint_final.pth" "final"
    break
  fi
  sleep 60
done
echo "$(date '+%Y-%m-%d %H:%M:%S') [$LABEL] archiver exiting"
