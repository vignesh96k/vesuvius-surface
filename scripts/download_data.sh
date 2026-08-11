#!/usr/bin/env bash
# Downloads the competition dataset into $VESUVIUS_DATA_ROOT via the Kaggle API.
# See docs/data.md. Requires ~/.kaggle/kaggle.json and that you've accepted the
# competition's rules on the Kaggle website first (competition downloads are gated on this).
set -euo pipefail

if [[ -z "${VESUVIUS_DATA_ROOT:-}" ]]; then
  echo "ERROR: set VESUVIUS_DATA_ROOT first, e.g.:"
  echo "  export VESUVIUS_DATA_ROOT=\$PWD/data"
  exit 1
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not found. pip install kaggle, then place credentials at"
  echo "  ~/.kaggle/kaggle.json (chmod 600), or export KAGGLE_USERNAME / KAGGLE_KEY."
  exit 1
fi

mkdir -p "$VESUVIUS_DATA_ROOT"
echo "Downloading vesuvius-challenge-surface-detection -> $VESUVIUS_DATA_ROOT"
kaggle competitions download -c vesuvius-challenge-surface-detection -p "$VESUVIUS_DATA_ROOT"

ZIP="$VESUVIUS_DATA_ROOT/vesuvius-challenge-surface-detection.zip"
if [[ -f "$ZIP" ]]; then
  echo "Unzipping..."
  unzip -q -o "$ZIP" -d "$VESUVIUS_DATA_ROOT"
  rm "$ZIP"
fi

echo "Done. See docs/data.md for the expected layout."
