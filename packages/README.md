# packages/

Nested, independently-installable Python packages that deliberately live in their own
sub-environment rather than being flattened into `src/vesuvius_surface/`. Currently just one:

- **`vesuvius_evaluation/`** — the official, leaderboard-equivalent competition scorer (wraps
  the organizers' `topometrics` package). Needs its own conda env (`environment-eval.yml`, at
  the repo root) because its pinned numpy/scipy versions are older than the training
  environment's and would risk a C-ABI break with torch/nnU-Net if installed together — a
  real, previously-hit crash, not a hypothetical (see `docs/reproducibility_notes.md`).

`src/vesuvius_surface/evaluation/` (the main package) is repo-specific orchestration —
resumable, per-scroll-aggregated scoring over a directory of predictions — that calls into an
official scorer. **Known gap, documented honestly rather than silently duplicated:** it
currently uses its own small array-based wrapper (`metric_adapter.py`) around the same
underlying `topometrics` package that `vesuvius_evaluation.official_score` also wraps (via a
path-based interface), rather than calling into this vendored package directly. Both produce
numerically identical scores (same underlying `topometrics.leaderboard.compute_leaderboard_score`
call, same weights/params) — this is wrapper-interface duplication (array-in vs. path-in), not
a scoring discrepancy — but it's real duplication worth consolidating with dedicated testing
time rather than in the same pass that produced this repo's real reported numbers.
