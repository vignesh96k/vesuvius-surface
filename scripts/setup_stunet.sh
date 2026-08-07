#!/usr/bin/env bash
# Install STU-Net for fine-tuning on a split we control.
#
# Why STU-Net rather than the published m7 checkpoint: STU-Net is pretrained on
# TotalSegmentator (human anatomy CT), so it has provably never seen a Vesuvius
# volume. Fine-tuning it against our own split therefore yields a holdout that
# is genuinely clean — which m7 cannot give us, because its split was never
# recorded. It also starts from pretrained weights rather than scratch, and the
# paper reports successful CT -> MRI and CT -> PET transfer, so CT -> micro-CT
# is at least plausible.
#
# IMPORTANT: STU-Net ships its own nnU-Net 2.2 fork. Installing it into the
# active environment would replace the nnunetv2 that our m7 pipeline depends
# on, so this installs into a separate conda env by default.
#
# Usage:
#   bash scripts/setup_stunet.sh                  # env stunet, model base
#   bash scripts/setup_stunet.sh --model large
#   bash scripts/setup_stunet.sh --same-env       # skip isolation (not advised)

set -euo pipefail

STUNET_ROOT="${STUNET_ROOT:-/mnt/workspace/code/stunet}"
ENV_NAME="${ENV_NAME:-stunet}"
MODEL="${MODEL:-base}"
SAME_ENV=0
REPO="https://github.com/Ziyan-Huang/STU-Net.git"
HF_REPO="ziyanhuang/STU-Net"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --root) STUNET_ROOT="$2"; shift 2 ;;
    --same-env) SAME_ENV=1; shift ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: bash scripts/setup_stunet.sh [--model small|base|large|huge] [--env NAME] [--root DIR] [--same-env]"
      exit 1
      ;;
  esac
done

case "$MODEL" in
  small) PARAMS="14.6M"  ;;
  base)  PARAMS="58.3M"  ;;
  large) PARAMS="440M"   ;;
  huge)  PARAMS="1.46B"  ;;
  *) echo "ERROR: --model must be small, base, large or huge"; exit 1 ;;
esac

echo "model     : STU-Net-$MODEL ($PARAMS params)"
echo "root      : $STUNET_ROOT"
if [[ "$SAME_ENV" -eq 1 ]]; then
  echo "env       : (current) — WARNING: this replaces nnunetv2 in place"
else
  echo "env       : $ENV_NAME (isolated from the m7 pipeline)"
fi
echo

mkdir -p "$STUNET_ROOT"

if [[ ! -d "$STUNET_ROOT/STU-Net" ]]; then
  echo "== cloning STU-Net"
  git clone --depth 1 "$REPO" "$STUNET_ROOT/STU-Net"
else
  echo "== STU-Net already cloned"
fi

FORK_DIR="$STUNET_ROOT/STU-Net/nnUNet-2.2"
if [[ ! -d "$FORK_DIR" ]]; then
  echo "ERROR: expected nnU-Net v2 fork at $FORK_DIR"
  echo "Repository layout may have changed. Contents:"
  ls -1 "$STUNET_ROOT/STU-Net"
  exit 1
fi

if [[ "$SAME_ENV" -eq 0 ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found; re-run with --same-env or install conda"
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "== creating conda env $ENV_NAME (python 3.11)"
    conda create -y -n "$ENV_NAME" python=3.11
  fi
  conda activate "$ENV_NAME"
  echo "== active python: $(which python)"
fi

echo "== installing STU-Net's nnU-Net fork (editable)"
pip install -e "$FORK_DIR"

echo "== installing helpers"
pip install huggingface_hub tifffile tqdm

WEIGHTS_DIR="$STUNET_ROOT/weights"
mkdir -p "$WEIGHTS_DIR"

echo "== fetching STU-Net-$MODEL weights from $HF_REPO"
python - "$HF_REPO" "$MODEL" "$WEIGHTS_DIR" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import list_repo_files, hf_hub_download

repo, model, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])

files = list_repo_files(repo)
matches = [f for f in files if model in f.lower() and f.endswith((".pth", ".model", ".pt"))]
if not matches:
    matches = [f for f in files if model in f.lower()]
if not matches:
    print(f"Could not find a '{model}' checkpoint in {repo}.")
    print("Available files:")
    for f in files:
        print("   ", f)
    raise SystemExit(1)

# Prefer the shortest name; variants tend to be suffixed.
target = sorted(matches, key=len)[0]
path = hf_hub_download(repo_id=repo, filename=target, local_dir=str(dest))
print(f"WEIGHTS={path}")
PY

echo
echo "Setup complete."
echo
echo "Next:"
echo "  1. Author a known split (run in the ORIGINAL env, it needs our src/):"
echo "       python scripts/make_scroll_split.py --mode stratified --dry-run"
echo "  2. Fine-tune:"
echo "       bash scripts/stunet_finetune.sh --model $MODEL --weights $WEIGHTS_DIR/<file>"
echo
echo "Reminder: this env is separate. Use the original env for the m7 pipeline"
echo "and scoring, and this one only for STU-Net training."
