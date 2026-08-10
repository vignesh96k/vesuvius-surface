#!/usr/bin/env python3
"""Thin wrapper around nnU-Net's own training entry point.

Historical note: an earlier version of this file (`run_training_wrapper.py`, repo root)
existed specifically to work around a namespace collision -- this package used to be a bare
top-level `training` module, which shared its name with nnunetv2's own internal
`nnunetv2.training` subpackage, and nnU-Net's trainer-discovery mechanism could resolve a
bare `import training` to the wrong one depending on import order. Renaming this package to
`vesuvius_surface.training` (this file's own location) makes that collision structurally
impossible -- `vesuvius_surface.training` cannot be confused with `nnunetv2.training` -- so
the workaround itself is gone, not just better-documented. See docs/reproducibility_notes.md.

This file now only exists as a convenience for invoking nnU-Net's real training entry point
without a prior `pip install -e .` (it puts `src/` on `sys.path` first). If this package is
installed (`pip install -e .`), just use the real `nnUNetv2_train` CLI directly instead.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nnunetv2.run.run_training import run_training_entry  # noqa: E402

if __name__ == "__main__":
    run_training_entry()
