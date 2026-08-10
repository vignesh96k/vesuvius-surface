# Decisions — what was chosen, what else was on the table, and why

This project's rubric requires being able to explain, for any line of code or model choice,
why it's there and what else was considered. That reasoning already exists throughout this
repo — in docstrings, commit messages, `research_log.md`, and `presentation_notes.md` — but
it's scattered. This doc consolidates the decisions an evaluator is most likely to ask about
into one place, each in the same shape: **Context** (the problem), **Alternatives considered**
(what else was real, not hypothetical), **Decision**, **Why**, and **Source** (where the full
account lives, so nothing here is asserted without a paper trail).

One entry at the end is deliberately different: it names a real default with **no** recorded
justification, rather than inventing one to fill the shape. That's intentional — see the note
there.

---

## 1. Validation split: leave-one-scroll-out (LOSO), not stratified k-fold or a random split

**Context.** Needed a validation split that would actually predict generalization, not just
measure "another region of a familiar scroll."

**Alternatives considered** (all three are real, implemented modes in
`scripts/make_scroll_split.py`, not hypothetical):
- `stratified` — every fold's validation set gets a proportional share of every scroll.
  Guarantees coverage, but train/val share scrolls, so it measures a within-scroll region,
  not held-out generalization.
- `scroll-holdout` — leave-one-scroll-out across *all* scrolls. Correct in principle, but
  small scrolls (44430 has 16 volumes, 53997 has 13) make some folds too tiny to trust.
- `holdout-scroll` — hold out *one* named scroll (26010, 129 volumes) as a single validation
  fold, train on everything else.

**Decision:** `holdout-scroll`, scroll 26010.

**Why:** Backed by real literature on same-patient/same-scan leakage in medical imaging
(Yagis et al. 2021 *Sci Rep*; Varoquaux & Cheplygina 2022 *npj Digital Medicine*) — same-scroll
cases are likely spatially correlated, so a stratified split's "good" numbers would be
optimistic. Scroll 26010 gives 129 held-out volumes, large enough to trust a single number
without needing 5 separate folds. Stratified k-fold wasn't discarded — it was run *separately*,
specifically to sanity-check whether the single LOSO number was a lucky/unlucky draw (see
decision 2).

**Source:** `scripts/make_scroll_split.py` docstring; `presentation_notes.md` line 36-38.

---

## 2. Also ran stratified k-fold — not to find a better model, but to audit the split itself

**Context.** A single LOSO number (0.5162) could plausibly be a lucky or unlucky draw of
which scroll got held out.

**Alternative considered:** Trust the single LOSO number and move on — it's the methodologically
correct split, so re-checking it could look like second-guessing a sound decision.

**Decision:** Ran 3 independent stratified 80/20 folds as a cross-check, not as a candidate
replacement methodology.

**Why:** Convergent evidence beats a single point estimate. All 3 folds landed within ~0.01 of
each other (0.5162 / 0.5079 / 0.5051) — real evidence the split wasn't a fluke, not assumed.
When asked to cut this from 5 folds to 3 for time, agreed: each stratified fold is already an
independent, valid split, so running fewer is a legitimate compute trade-off, not a methodology
change.

**Source:** `presentation_notes.md` line 60-69.

---

## 3. Baseline: train from scratch, not fine-tune a public checkpoint

**Context.** Could have skipped a from-scratch baseline entirely and made "fine-tune
arunodhayan's checkpoint" the project's rung-1 baseline — less work, likely a higher number
immediately.

**Alternative considered:** Exactly that — fine-tune the public checkpoint *as* the baseline.

**Decision:** Keep from-scratch training and fine-tuning as separate rungs.

**Why:** Blending "our own model" and "someone else's pretrained model" into one number would
blur attribution — no way to tell how much of a score is our decisions versus the base model's.
This decision predates (and independently motivates) the leakage discovery in decision 4, which
gave it a second, stronger reason: arunodhayan's and m7's checkpoints were both trained on
100% of the data, so no local score against either is a clean generalization estimate anyway —
a from-scratch baseline trained against our own authored split is the only way to get one.

**Source:** `presentation_notes.md` line 18-22 (original attribution reasoning); line 85-92
(the leakage discovery that reinforced it); `scripts/verify_split.py`.

---

## 4. Baseline config: `3d_lowres`, not `3d_cascade_fullres`

**Context.** nnU-Net offers `2d`, `3d_lowres`, `3d_fullres`, and `3d_cascade_fullres`
configurations; the cascade config is what the strongest public checkpoints use.

**Alternative considered:** Start the baseline directly on `3d_cascade_fullres`, matching the
architecture of the strongest reference solutions.

**Decision:** `3d_lowres` for rung 1.

**Why:** Cascade is a two-stage, more complex setup. A baseline's job is to be a clean floor
reference, not to already be competitive — cascade complexity belongs in a later rung (it
appears at decision 8, fine-tuning arunodhayan's actual cascade checkpoint), not mixed into
the first, simplest measurement.

**Source:** `presentation_notes.md` line 14-16.

---

## 5. Seed RNG state, but don't force full `cudnn` determinism

**Context.** Stock nnU-Net sets no random seed anywhere — verified directly from
`nnUNetTrainer.__init__` and the `nnUNetv2_train` CLI arg list, not assumed. Re-running the
same config could give a different result, with no way to tell whether a change in score was
real or noise.

**Alternatives considered:**
- Leave it unseeded (nnU-Net's actual default) — rejected, since it makes ablations meaningless
  (can't tell if a comparison's delta is signal or seed noise).
- Force full bit-exact determinism (`cudnn.deterministic=True`, `cudnn.benchmark=False`) —
  rejected as unnecessary cost: this class of run only needs *run-to-run comparability*
  (same init, same augmentation order), not bit-identical reproduction, and forcing full
  determinism has a real, measurable training-speed cost.

**Decision:** `nnUNetTrainerSeeded` seeds Python/NumPy/PyTorch RNG state at trainer
construction; `cudnn.benchmark=True` is left at nnU-Net's own speed default.

**Source:** `src/vesuvius_surface/training/trainers/nnUNetTrainerSeeded.py` docstring;
`presentation_notes.md` line 24-30.

---

## 6. Fine-tuning comparisons: "before/after delta" methodology, not absolute scores

**Context.** arunodhayan's and m7's checkpoints were both confirmed trained on 100% of the
dataset ("we abandoned the traditional K-Fold cross-validation... trained directly on the
entire dataset" — their own writeup), so any local score against either checkpoint (zero-shot
*or* fine-tuned) is inflated by leakage. This looked like it might invalidate every local
number for the entire fine-tuning track.

**Alternatives considered:**
- Report absolute local scores anyway, with a caveat — rejected: a caveat doesn't fix an
  invalid comparison, it just discloses it.
- Abandon local validation for this track entirely, wait for real Kaggle submissions only —
  rejected: too slow to iterate on, and real submissions are a scarce, rate-limited resource.
- Switch to STU-Net (see decision 7) for *every* fine-tuning question — rejected: STU-Net is
  architecturally different, so it can't answer "does fine-tuning arunodhayan's specific
  checkpoint help," only "does fine-tuning in general help on a leak-free target."

**Decision:** Score the checkpoint before and after fine-tuning on the *same* held-out set,
never fine-tuning on the held-out portion. Report the delta, not either absolute number.

**Why:** Contamination is identical in both the before and after conditions, so it cancels out
of the delta — the delta isolates whether fine-tuning helped, even though neither number alone
predicts real leaderboard performance. Real Kaggle submissions remain the actual arbiter across
every rung, since the hidden test set is equally unseen to everyone regardless of pretraining.

**Source:** `presentation_notes.md` line 94-99.

---

## 7. STU-Net as the one genuinely leak-free fine-tuning comparison

**Context.** Every available public Vesuvius checkpoint (arunodhayan, m7) has the all-data
contamination problem from decision 6. Wanted at least one fine-tuning data point with a truly
clean holdout, not just a delta-corrected one.

**Alternative considered:** Accept that no leak-free comparison is possible with Vesuvius-domain
checkpoints and rely on decision 6's delta methodology exclusively.

**Decision:** Fine-tune STU-Net-B, a TotalSegmentator-pretrained checkpoint, specifically
*because* it has provably never seen a Vesuvius volume.

**Why:** A genuinely clean holdout is worth having even from an architecturally-unrelated
starting point — it answers a different, complementary question ("does full fine-tuning onto
an out-of-domain but strong pretrained checkpoint work at all") without any contamination
caveat needed. Result was a clear negative (0.4629 vs. 0.5575 best baseline) — real evidence
that full fine-tuning was not the shortcut it looked like, from an unimpeachable source.

**Source:** `docs/checkpoints.md`; `experiment_summary.md` Phase 3 item 10.

---

## 8. 1000-epoch backbone run — approved for a *different* reason than it was first proposed

**Context.** User proposed a 1000-epoch from-scratch run.

**Alternative considered:** Argued against it initially — fine-tuning a pretrained checkpoint
looked like a faster path to a good score than training a longer backbone from scratch.

**Decision:** Ran it anyway, once the *reasoning* changed.

**Why:** The user's justification wasn't "faster to a good score" (the thing already argued
against) — it was "we need something we can fully, honestly validate, given the fine-tuning
contamination problem" (decision 6). That's a genuinely different question with a different
right answer, not the same idea recycled with more insistence. Worth recording as an example
of updating a position for the right reason, not just because asked twice.

**Source:** `presentation_notes.md` line 77-83.

---

## 9. Skeleton-recall picked via a 5-way empirical comparison, not chosen a priori

**Context.** Several auxiliary-loss/architecture candidates were plausible for improving
topology-sensitive segmentation: skeleton-recall, clDice+ScheduleFree, an affinity auxiliary
head, a highpass input channel, and a Laplacian-pyramid channel.

**Alternatives considered:** All 5, trained head-to-head at 100 epochs on the identical
657/129 LOSO split — not argued about in the abstract.

**Decision:** Skeleton-recall (0.5307), by a real margin over clDice+ScheduleFree (0.5285),
affinity (0.5226), highpass-only (0.5204), and Laplacian (0.5122).

**Why:** An empirical bake-off removes the need to argue from intuition about which topology
loss "should" work best. Note: clDice was also isolated from the RAdamScheduleFree optimizer
separately (a "clDice loss, stock SGD" control), specifically to avoid crediting the wrong
ingredient for the combined variant's gain over baseline — a real methodological step, not
assumed necessary.

**Source:** `experiment_summary.md` Phase 3 item 11; `presentation_notes.md` line 166-170.

---

## 10. The central pivot: freeze everything but the last layers, don't fully fine-tune

**Context.** This is the project's single most important modeling decision. Full fine-tuning
was the obvious first thing to try on top of a strong pretrained checkpoint.

**Alternatives considered, all real, all tried before this one, all negative:**
- STU-Net fine-tune, early encoder layers frozen (leak-free target): 0.4629 vs. 0.5575 baseline.
- The 5-way loss/architecture comparison retried as a full cascade fine-tune.
- Full (unfrozen) arunodhayan fine-tune — highpass input + skeleton-recall + affinity loss,
  applied to both the ensemble and the cascade: cascade 0.7198→0.5208, ensemble 0.7029→0.5172,
  unambiguously negative across every component metric.

**Decision:** Freeze everything except the final decoder stage and deep-supervision heads
(0.07% of parameters trainable), fine-tune only that, for 10 epochs.

**Why:** Three independent negative results (different architectures, different loss recipes,
different starting checkpoints) converge on the same conclusion: full fine-tuning on top of a
strong pretrained checkpoint is a real risk, not a free win — it's easy to destroy what the
checkpoint already does well before the new signal has a chance to help. Freezing almost
everything limits how much can go wrong. Result: **the project's only genuinely positive
fine-tuning result** (0.7248 vs. 0.7198 zero-shot, +0.0050 on the full 129-case LOSO).

**Note on reproducibility:** the full arunodhayan fine-tune is fully reproducible — it reuses
`nnUNetTrainerSkeletonRecallAffinity`, an already-existing, tested trainer class, via
`-pretrained_weights` on a highpass-augmented dataset. `README.md` step 9 condenses this
negative result to one line; the real commands (dataset prep + the two `nnUNetv2_train` calls)
live in `docs/reproducibility_notes.md`'s entry on this experiment.

**Source:** `experiment_summary.md` Phase 3/4; `README.md` "Results" and step 9;
`src/vesuvius_surface/training/trainers/nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs.py`.

---

## 11. Metric-guided unmerge: nearest-seed Voronoi partition, not a full watershed

**Context.** The 1st-place postprocessing control chain repairs holes *inside* a connected
component but never severs a bridge fused *between* two components (confirmed via ablation:
`voi_merge` sits essentially flat across every control stage). Needed a way to partition a
merged component back into its real pieces once a bridge is detected.

**Alternative considered:** A full watershed transform, seeded by the eroded pieces — the more
standard, more general tool for exactly this kind of partition problem.

**Decision:** Euclidean nearest-seed assignment via `distance_transform_edt(...,
return_indices=True)` — a Voronoi tessellation seeded by the eroded pieces.

**Why:** "The simplest valid stand-in for a full watershed, and it needs nothing beyond scipy"
(already a hard dependency here) — for two-seed splits on locally-thin sheet geometry, the
extra machinery a full watershed brings (elevation functions, flooding order, ridge handling)
buys nothing a straight nearest-seed assignment doesn't already give, and it's simpler to
reason about and to write a unit test against.

**Source:** `src/vesuvius_surface/postprocess/unmerge.py` module docstring.

---

## 12. Unmerge erosion radius: 1, not 2

**Context.** Bridge detection erodes each connected component by a ball of radius
`erosion_radius`; a thin neck should vanish under erosion while the two masses it joins
survive as separate seeds. The radius controls how aggressive that erosion is.

**Alternative considered:** `erosion_radius=2` — tried first, not assumed suboptimal in
advance.

**Decision:** `erosion_radius=1`.

**Why:** Measured, not guessed: a real distance-transform check on control masks showed even
the largest healthy component's half-thickness maxes out around 2.0 voxels (median 1.0) — these
are inherently thin sheets, not blobs. `erosion_radius=2` erodes a healthy sheet away entirely
(zero candidates found across 5 real cases). `erosion_radius=1` is the largest radius that
still leaves real sheet material as seeds while still stripping sub-voxel-scale bridges, and it
immediately finds real candidates (9 in one case from scroll 35360, the same scroll flagged
separately for merge skew).

**Source:** `src/vesuvius_surface/postprocess/unmerge.py`, `UnmergeConfig` docstring.

---

## 13. Unmerge accept/reject gate: per volume, not per cut

**Context.** A volume can contain multiple candidate merge bridges. The metric-improvement
gate ("accept a cut only if the official metric improves") could in principle be applied to
each cut individually, or to the whole volume's cut-set at once.

**Alternative considered:** Score each candidate cut independently within a volume, accepting
or rejecting each on its own merit — the more granular, in-principle-more-correct approach.

**Decision:** Score the whole volume once, with every candidate cut applied at once; accept or
reject the entire cut-set together.

**Why:** Cost, stated plainly rather than hidden: scoring one volume already costs ~60-90s
(Betti-matching dominates), so per-cut scoring inside a volume with multiple candidates is not
affordable at the scale this project runs at. Per-volume is "the coarsest granularity that is
still a direct, honest reading of 'accept a cut only if the official metric improves on that
volume'" — a real trade-off, disclosed as one, not presented as free.

**Source:** `src/vesuvius_surface/postprocess/unmerge.py` module docstring.

---

## 14. Two separate conda environments, not one shared environment

**Context.** Training/inference (nnU-Net, torch) and the official scorer (`topometrics`) have
different pinned dependency requirements.

**Alternative considered:** One shared environment, installing the scorer's dependencies
alongside nnU-Net's — simpler setup, one fewer thing to document.

**Decision:** `environment-train.yml` and `environment-eval.yml`, kept deliberately separate.

**Why:** Not a style preference — a real, already-hit failure mode. The scorer's pinned
numpy/scipy versions are older than what a modern torch build needs; force-installing them into
the training environment risks a numpy 2.x↔1.x-class C-ABI break that can silently corrupt
torch/nnU-Net's own compiled extensions. Evaluation only ever reads `.tif` files off disk — it
never needs to run in the same process as training, so there's no cost to keeping them apart.

**Source:** `docs/reproducibility_notes.md`, "Why two conda environments, not one";
`packages/vesuvius_evaluation/docs/setup.md`.

---

## 15. Package rename (`vesuvius_surface.training`), not a `sys.path` workaround

**Context.** The repo's own top-level `training` package shared its name with nnU-Net's
internal `nnunetv2/training/` subpackage. nnU-Net's trainer-discovery mechanism temporarily
inserts its own root onto `sys.path[0]` (ahead of this project's `PYTHONPATH`) before importing
candidate trainer modules, so `import training.trainers` could silently resolve to nnU-Net's
*internal* package instead of this project's own.

**Alternatives considered, both tried, both real, both eventually rejected:**
- Pre-import `training.trainers` in a thin wrapper script before nnU-Net's entry point runs, so
  Python's import cache serves the correct module afterward. Worked, but *only* for invocations
  that went through that specific wrapper — a plain, unwrapped `nnUNetv2_train` call for an
  unrelated trainer crashed on the identical collision, because nnU-Net's discovery mechanism
  scans and imports every file in its trainer directory regardless of which one is requested.
- A `sitecustomize.py` startup hook in the conda environment, pre-importing `training.trainers`
  before *any* code runs. Fixed the gap above — verified against both a stock trainer and a
  project trainer — but is still a workaround for a name collision that could resurface with a
  new entry point later.

**Decision:** Rename the package to `vesuvius_surface.training`.

**Why:** A different top-level name can't collide with `nnunetv2.training` regardless of import
order, entry point, or future code paths — the fix is structural, not procedural. This was
judged the cheapest possible moment to do it, since a packaging/import-path pass was already
underway for other reasons.

**Source:** `presentation_notes.md` line 238-254; `README.md` "Packaging notes"; git history on
`src/vesuvius_surface/training/run_training.py`.

---

## 16. Cite arunodhayan's real driver's recipe directly; don't vendor the full script

**Context.** The actual checkpoint fine-tuning driver that produced arunodhayan's checkpoints
in his own solution is a 1140-line, hardcoded, notebook-derived script with no CLI or config
file. What this project actually needed from it was narrow: his real loss/optimizer recipe
(DC+CE + 0.2·clDice, RAdamScheduleFree, no LR schedule), to replicate as a controlled
ablation (`nnUNetTrainerSeeded_ClDice_ScheduleFree`).

**Alternatives considered, both tried in sequence, not just discussed:**
- Clean it up into a readable, config-driven script and present that as "how this was
  produced" — rejected outright: this project already has one real lesson about the cost of
  presenting a reconstruction as evidence (see Why, below), so this was never seriously on the
  table.
- Vendor the full script verbatim, unmodified, in `third_party/` — the first real decision:
  gives an evaluator the actual code to check any claim about it against, not just this
  project's word. Implemented, lived in the repo for a while.

**Decision:** Removed the full vendored script; cite the specific recipe values directly in
`docs/attribution.md` instead, with no code file in this repo standing in for arunodhayan's
own work.

**Why:** The full-vendoring choice was reconsidered once its actual use became clear: only a
handful of hyperparameter values were ever read from it, everything else in the 1140 lines
(mount-waiting, environment setup, notebook boilerplate) was never used for anything. Keeping
someone else's entire pipeline physically in this repo for that narrow a purpose was
disproportionate, and sits uncomfortably close to the assignment's own explicit rule against
presenting someone else's work as this project's own — even when clearly labeled as vendored,
its mere physical presence invites that reading. The underlying lesson that motivated vendoring
in the first place still holds and still applies: an earlier version of this README claimed
the public `scrollprize/surface_m7_nnunet` checkpoint was genuinely `fold_0` of a real
cross-validation split, backed by an elaborate, well-built reconstruction of which cases it
never saw — checked directly against the checkpoint's own embedded metadata (`fold='all'`),
and against the actual 1st-place team's own writeup, and the reconstruction was simply wrong.
A well-built reconstruction had been mistaken for evidence. That's still why the clean rewrite
in `src/` is explicitly marked unverified against the original rather than presented as a
faithful reproduction — the fix for *that* risk was never the vendored file itself, it was
never fabricating what wasn't directly read or verified.

**Source:** `presentation_notes.md` line 134-144 (the m7 `fold='all'` incident);
`docs/reproducibility_notes.md`; `docs/attribution.md`.

---

## 17. `score_model.py --workers 8` — an empirically-found ceiling, not a round number

**Context.** The official scorer is CPU-bound and slow (~60-90s/volume); parallelizing across
cases is the obvious speedup.

**Alternatives considered, both tried, both real failures:**
- 16 workers: each spawned its own ~25-thread BLAS pool (`OMP_NUM_THREADS` uncapped before
  numpy import) — 400 threads competing for 22 cores. Result: 8.5 minutes, zero completed
  cases.
- 20 workers (after fixing the thread-oversubscription bug above): the topology/Betti-matching
  computation needs up to ~5GB RAM per case, not obvious from a purely CPU-bound framing — 20
  concurrent workers exceeded the box's 117GB, and the kernel's OOM-killer cascaded, taking
  down 5+ workers within 4 seconds and the terminal multiplexer session itself (same cgroup).

**Decision:** 8 workers, with `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/
`NUMEXPR_NUM_THREADS` pinned to 1 *before* numpy is first imported in the process.

**Why:** ~40GB worst-case memory at 8 workers, verified against real per-worker RSS rather than
assumed safe this time — a number arrived at by hitting two different real failure modes first,
not chosen a priori. This same OOM class recurred once more later in the project (parallelizing
`postprocess/unmerge.py`'s scoring path) and was caught the same way: real per-worker RSS
measurement before trusting a new worker count.

**Source:** `presentation_notes.md` line 270-285; `scripts/evaluation/score_model.py` `--workers`
help text.

---

## 18. Metric wrapper: pin every parameter explicitly, don't trust the package's own defaults

**Context.** `evaluation/metric_adapter.py`'s `score_pair()` used to call
`compute_leaderboard_score(prediction, label, **overrides)` with no explicit parameters, on the
stated theory that the package's own defaults already matched the leaderboard.

**Alternative considered:** Exactly that — trust the package defaults, since it's one fewer
place a hardcoded value could drift from the package's own evolving API.

**Decision:** Pin every parameter explicitly in a `DEFAULT_METRIC_KWARGS` dict, matching
`scripts/evaluation/score_model.py`'s own long-standing explicit call exactly.

**Why:** The "trust the defaults" assumption was never actually checked against what
`score_model.py` (which produced every real reported number in this project) explicitly passes
— and they differed on exactly one parameter (`voi_alpha`: package default `1.0`, versus `0.3`
used everywhere numbers were actually reported). This was a real, discovered bug: since
`postprocess/unmerge.py`'s own accept/reject gate calls `score_pair()` directly, every accept/
reject decision the unmerge novelty layer had ever made was gated against a metric that didn't
match what this project reports everywhere else. Verified by scoring the same real case both
ways before concluding anything (0.5204 at the old default vs. 0.6182 at the explicit value,
which matched a number already on record). One source of truth now, not two that can silently
diverge.

**Source:** `src/vesuvius_surface/evaluation/metric_adapter.py` module docstring and git history
on that file (the commit fixing this).

---

## 19. arunodhayan-line Kaggle submission: layer our postprocessing on top of his, don't replace it

**Context.** Submitting the fine-tuned arunodhayan checkpoint (decision 10) to Kaggle required
a real inference notebook. The forked notebook already contains arunodhayan's own
postprocessing (inverse-EDT dilation + Hessian ridge detection) — a different technique from
this project's own 1st-place-writeup postprocessing chain (`first_place.py`).

**Alternative considered:** Replace his postprocessing step with `first_place.py` entirely,
matching exactly how the local LOSO number for this model (0.7363) was computed.

**Decision:** Keep his postprocessing step untouched; add `first_place.py` as an additional
step layered on top of its output, not a replacement.

**Why:** Explicit, direct instruction — don't rewrite what's already his and working; add this
project's own contribution on top rather than erase part of the vendored pipeline. Matches this
project's general stance on third-party code (decision 16): adapt around it, don't rewrite it
and call the result the same thing.

**Source:** This session's direct instruction ("dont change arunodhayan's pp add our method on
top"); the pushed kernel `vigneshk96/vesuvius-cascade-lastlayers-1st-pp`.

---

## What this doc deliberately does *not* do: `min_score_delta=0.0`

`UnmergeConfig.min_score_delta` defaults to `0.0` — a cut is accepted on *any* non-negative
score improvement, however small. The docstring explains what raising it would do ("require a
real margin before keeping a cut"), but there is no recorded evidence trail for why `0.0`
specifically was the shipped value rather than, say, `0.005` to require a real margin. Unlike
every entry above, this one doesn't have an alternative-considered story to tell — and rather
than manufacture one to fill the shape of this document, it's named here plainly as a real gap.
If asked directly: this is a default that hasn't been tuned or justified beyond "accept any
non-negative signal," and it's exactly the kind of thing this project's own logging discipline
(see `presentation_notes.md`) would normally have caught being set — worth flagging honestly
rather than padding this document past what the repo can actually back up.
