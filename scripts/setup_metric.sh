#!/usr/bin/env bash
# Install the official competition metric (topometrics) so scoring can be done
# locally instead of by spending leaderboard submissions.
#
# Score = 0.30*TopoScore + 0.35*SurfaceDice@tau + 0.35*VOI_score
#
# Source: Kaggle dataset sohier/vesuvius-metric-resources. The TopoScore term
# needs the Betti-Matching-3D C++ submodule compiled, so this needs network
# access, git, cmake and a C++ toolchain.
#
# Usage:
#   bash scripts/setup_metric.sh                 # download, build, install, verify
#   bash scripts/setup_metric.sh --inspect-only  # just dump the installed API

set -euo pipefail

METRIC_ROOT="${METRIC_ROOT:-/mnt/workspace/code/metric}"
DATASET="sohier/vesuvius-metric-resources"
INSPECT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --metric-root) METRIC_ROOT="$2"; shift 2 ;;
    --inspect-only) INSPECT_ONLY=1; shift ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: bash scripts/setup_metric.sh [--metric-root DIR] [--inspect-only]"
      exit 1
      ;;
  esac
done

dump_api() {
  echo
  echo "=== installed metric API ==="
  python - <<'PY'
import importlib
import inspect
import pkgutil

CANDIDATES = ["topometrics", "topological_metrics", "topological_metrics_kaggle"]

mod = None
for name in CANDIDATES:
    try:
        mod = importlib.import_module(name)
        print(f"import OK: {name}")
        break
    except ImportError:
        continue

if mod is None:
    print("Could not import any of:", ", ".join(CANDIDATES))
    print("Installed top-level modules matching 'topo'/'metric':")
    for m in pkgutil.iter_modules():
        if "topo" in m.name.lower() or "metric" in m.name.lower():
            print("   ", m.name)
    raise SystemExit(1)

print("version   :", getattr(mod, "__version__", "n/a"))
print("file      :", getattr(mod, "__file__", "n/a"))

if hasattr(mod, "__path__"):
    print("\nsubmodules:")
    for m in pkgutil.iter_modules(mod.__path__):
        print("   ", m.name)

print("\npublic callables:")
for name in sorted(dir(mod)):
    if name.startswith("_"):
        continue
    obj = getattr(mod, name)
    if not callable(obj):
        continue
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "(?)"
    print(f"    {name}{sig}")
PY
  echo "=== end API ==="
}

if [[ "$INSPECT_ONLY" -eq 1 ]]; then
  dump_api
  exit 0
fi

for tool in git cmake make python; do
  command -v "$tool" >/dev/null 2>&1 || { echo "ERROR: '$tool' not found on PATH"; exit 1; }
done

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not found. Install it and add API credentials:"
  echo "  pip install kaggle"
  echo "  # then place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)"
  exit 1
fi

if [[ ! -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ]]; then
  echo "ERROR: kaggle.json not found. Download it from your Kaggle account page"
  echo "       (Settings -> API -> Create New Token) and save to ~/.kaggle/kaggle.json"
  exit 1
fi

mkdir -p "$METRIC_ROOT"
cd "$METRIC_ROOT"

echo "== downloading $DATASET -> $METRIC_ROOT"
kaggle datasets download "$DATASET" --unzip

PKG_DIR="$(find "$METRIC_ROOT" -maxdepth 3 -type d -name 'topological-metrics-kaggle' | head -n 1)"
if [[ -z "$PKG_DIR" ]]; then
  echo "ERROR: could not find topological-metrics-kaggle under $METRIC_ROOT"
  echo "Contents:"
  ls -1 "$METRIC_ROOT"
  exit 1
fi

echo "== package dir: $PKG_DIR"
cd "$PKG_DIR"

if [[ -f requirements.txt ]]; then
  echo "== installing requirements"
  pip install -r requirements.txt
fi

echo "== building Betti-Matching-3D (C++)"
[[ -f scripts/setup_submodules.sh ]] && chmod +x scripts/setup_submodules.sh
[[ -f scripts/build_betti.sh ]] && chmod +x scripts/build_betti.sh
make build-betti

echo "== installing package"
pip install -e . --no-deps --no-index --no-build-isolation -v

dump_api

echo
echo "Metric installed. Paste the API block above so the eval harness can be"
echo "wired against the real function signatures."
