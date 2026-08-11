# Presentation raw material

Running log of pushbacks, caveats, problems (and how we solved them), and ideas that came up
during the actual work — kept specifically so there's real, concrete material to mine for
slides, rather than reconstructing the story from memory afterward. Chronological. Each entry
tagged by type. Maintained continuously as things happen, not just backfilled once.

Tags: `[PUSHBACK]` a disagreement/redirect between human and AI (either direction) `[CAVEAT]`
an explicit limitation we stated `[PROBLEM]` something broke or was wrong, and how it got
fixed `[IDEA]` a genuinely novel decision or approach, not copied from the public notebook

---

- `[PUSHBACK]` Proposed `3d_cascade_fullres` as the baseline config early on. Reasoning:
  cascade is a two-stage, more complex setup — keeping the baseline as simple `3d_lowres`
  gives a clean floor reference; cascade complexity belongs in a later rung, not rung 1.

- `[PUSHBACK]` User asked whether fine-tuning arunodhayan's public solution could just *be*
  the baseline, rather than training from scratch. Argued to keep them as separate rungs —
  mixing "our own model" and "someone else's pretrained model" into one number would blur
  attribution (can't tell how much of the score is our decisions vs. his). User agreed: "yes
  this makes sense."

- `[PROBLEM]` Stock nnU-Net sets no random seed anywhere — verified directly from source
  (`nnUNetTrainer.__init__`, the `nnUNetv2_train` CLI arg list), not assumed. Fix: custom
  `nnUNetTrainerSeeded` class seeding random/numpy/torch, loaded via nnU-Net's own
  `nnUNet_extTrainer` external-trainer mechanism rather than patching the installed package.
  Deliberately left `cudnn.benchmark=True` alone (nnU-Net's own speed default) rather than
  forcing full bit-exact determinism, which would cost real training time for a baseline that
  only needs run-to-run comparability.

- `[PROBLEM]` User asked why val_loss (and later train_loss) showed negative numbers. Traced
  to source: nnU-Net's Dice loss returns `-dice_coefficient`, not `1 - dice`, so the compound
  loss is literally `CE − Dice`. Not a bug — expected as Dice climbs toward 1.

- `[IDEA]` Chose scroll-grouped leave-one-scroll-out over a random/case-level split, backed by
  real literature on leakage in medical imaging (Yagis et al. 2021 *Sci Rep*; Varoquaux &
  Cheplygina 2022 *npj Digital Medicine*) — same-scroll cases are likely spatially correlated.

- `[IDEA]` Reverse-engineered the real hidden test set's actual composition by searching the
  competition's own Kaggle discussion forum via the API (`kaggle competitions topics`) rather
  than guessing. Found a competitor's "Probing Results" thread: 71% of test samples come from
  scroll IDs also in the training pool, 29% from genuinely novel scrolls. This directly shaped
  the validation design and revealed a concrete, honest limitation (see next entry).

- `[CAVEAT]` Because of the 71%/29% split above, our held-out score can only ever validate the
  easier "seen scroll, new cases" majority of grading — there's no local way to simulate the
  29% "genuinely novel scroll" portion. Stated explicitly rather than implied.

- `[CAVEAT]` nnU-Net's own pseudo-dice is a biased proxy for checkpoint selection — computed
  on a foreground-oversampled (33%) patch stream, not the true class balance a full volume has.
  Verified from source (`oversample_foreground_percent` applied to both train *and* val
  dataloaders). Real scoring always goes through the actual `topometrics` official scorer.

- `[PROBLEM]` The official scorer was too slow to iterate with — ~60s/volume, ~2hr for the
  129-case baseline. Fix: rewrote `score_directory()` to parallelize via
  `ProcessPoolExecutor` (each case's score is independent). Cut the 129-case run to ~8-10 min,
  and the full 5-fold total (786 cases) from ~13hr sequential to well under an hour.

- `[IDEA]` Ran stratified k-fold specifically to test whether the *validation methodology*
  itself was trustworthy — not to find a better model, but to check whether the single LOSO
  number (0.5162) was representative or just a lucky/unlucky draw of which scroll got held
  out. Result: 3 independent stratified folds (0.5079 / 0.5051 / 0.5084) landed within 0.0033
  of each other, and within 0.0111 of the LOSO number itself (0.5162) — real convergent
  evidence, not assumed.

- `[PUSHBACK]` Proposed reducing the k-fold run from 5 folds to 3 to cut compute time. Argued
  this doesn't need regenerating anything — each of the 5 stratified folds is already an
  independent, valid 80/20 split; just running fewer of them is a legitimate compute/time
  trade-off, not a methodology change.

- `[PUSHBACK]` User asked directly: "is 100 epochs enough to determine this?" Checked the real
  training curve rather than defend the shortcut — EMA pseudo-dice was still climbing right up
  to the final epoch (0.5314→0.5510 in the last ~20 epochs). Answer was an honest "no, and
  here's what that means for how much we can trust this specific comparison" rather than
  a reassurance.

- `[PUSHBACK]` User proposed a 1000-epoch backbone run. Initially argued against training
  longer from scratch (fine-tuning a pretrained checkpoint is a faster path to a good score).
  When the user's *reasoning* changed — not "faster to a good score" but "we need something
  we can fully, honestly validate, given the fine-tuning contamination problem" — agreed,
  because that's a genuinely different question with a different right answer, not the same
  idea recycled. Good example of updating a position for the right reason, not just because
  asked twice.

- `[PROBLEM]` User raised a sharp concern: arunodhayan's checkpoint was trained on "all
  available data" (his own team's writeup: *"we abandoned the traditional K-Fold
  cross-validation ... trained directly on the entire dataset"*), so it's already seen every
  case in our local held-out sets — any local validation of a fine-tuned version would be
  contaminated. Confirmed this was real, then searched 11 top-10-through-88th-place solution
  threads for alternative checkpoints with public weights — found none; arunodhayan is the
  only practical fine-tuning candidate with real weights available, so switching base models
  doesn't escape the problem.

- `[IDEA]` Designed a "before/after delta" methodology to salvage a valid comparison despite
  the contamination: score arunodhayan's original checkpoint and our fine-tuned version on the
  *same* held-out set (never fine-tuning on the held-out portion ourselves). Contamination is
  identical in both conditions, so the delta isolates whether fine-tuning helped — even though
  neither number alone predicts real leaderboard performance. Real Kaggle submissions remain
  the only fair arbiter across rungs, since the hidden test set is equally unseen to everyone.

- `[PUSHBACK]` Asked whether we should broaden the weight-availability search to the top 25
  solutions. Checked empirically first (6 more threads, 12th-88th place) rather than guess —
  still zero weight links found. Recommended against expanding further: 11 threads pointing
  the same direction is a real signal, not sampling noise, and continuing would likely just
  cost time for the same negative result.

- `[IDEA]` User's own insight: arunodhayan's *standalone lowres* checkpoint (separate from the
  cascade model) turned out — verified directly from the checkpoint's embedded metadata, not
  assumed — to be the exact same architecture/config/plans as our own baseline
  (`3d_lowres`, `nnUNetResEncUNetMPlans`, same 786-case dataset), and at a similar epoch count
  (997) to our planned 1000-epoch run. That makes it a much cleaner comparison than the
  cascade model: same everything except training-data scope. Zero-shot inference + scoring on
  our LOSO held-out set launched from this.

- `[PUSHBACK]` Reorganized project structure: kept `baselinerun/` (rung 1) and created a
  separate `finetune/` (rung 2/3) rather than merging them — different starting point
  (pretrained vs. random init), different nnU-Net config (`3d_cascade_fullres` vs
  `3d_lowres`), different source lineage. User also asked for output folders to be "clearly
  demarcated" — gave `finetune/` its own output root (`outputs/finetune_run/`) separate from
  baselinerun's (`outputs/training_run/`), while still sharing the expensive, dataset-level
  preprocessing tree (no reason to duplicate large preprocessed volumes just for folder
  hygiene).

- `[PROBLEM]` Kaggle kernel push silently created a *new* kernel under a title-derived slug
  instead of versioning the existing one (title/id mismatch warning turned out not to be
  harmless this time). Traced via `kaggle kernels status` on both slugs to find which one was
  actually live.

- `[PROBLEM]` `kaggle competitions submit -f submission.zip` gave a 400 on this code
  competition. Fix: code competitions require `-k <kernel> -v <version>`, not a raw file
  upload — found via `--help`, then had to determine the *correct* version number empirically
  since the push output's stated version didn't match the naive assumption (fresh kernel = v1).

- `[PROBLEM]` `vesuvius-surface/README.md` claims the public 1st-place `scrollprize/
  surface_m7_nnunet` checkpoint is genuinely `fold_0` of a real cross-validation split (with an
  elaborate, well-built reconstruction of which cases it never saw). Checked directly rather
  than trust the README: the checkpoint's own embedded metadata says `fold='all'`, and the
  actual 1st-place team's own writeup independently confirms *"we used all available data for
  training"* for their nnU-Net baseline. The "fold_0" label appears to be a documentation
  artifact, not a real split. This flipped an earlier assumption (that m7 would be a cleaner
  fine-tuning target than arunodhayan because of "better split info") — it isn't; both have the
  same all-data contamination problem. Good example of verifying a documented claim against
  the actual binary/data rather than trusting the README, and of a plan changing because the
  evidence changed, not because of a preference.

- `[PROBLEM]` Corrected my own earlier claim (from the top-10-solutions weight search) that no
  top solution shares public weights. The actual 1st-place checkpoint is public
  (`scrollprize/surface_m7_nnunet` on HuggingFace, Apache-2.0) — I'd missed it because I was
  searching Kaggle discussion *thread text* for links, and it was only referenced via the
  writeup/model card, not the forum post itself. Worth stating plainly rather than quietly
  fixing it — an honest record of a real gap in the research, corrected once found.

- `[PROBLEM]` Told the user, when summarizing the validation conclusion, that LOSO "only
  measures the 71% in-distribution portion" of real grading and can't capture the 29%
  novel-scroll portion — backwards, and a direct contradiction of an earlier, correct
  statement in the same conversation (LOSO holds out an entire scroll, so it's actually
  analogous to the *harder* OOD case; the stratified k-fold splits are what test the 71%
  case). Caught by the user asking "isn't our held-out scroll novel too?", not self-caught.
  Once corrected, the follow-on claim ("predicted the local-vs-real gap's direction from this
  mechanism, then confirmed it") didn't hold up either — the corrected mechanism actually
  predicts the *opposite* direction from what was observed, so the honest conclusion is
  "local overestimates real, for reasons not yet cleanly attributed," not a validated
  forecast. Good, concrete material for the presentation's limits/communication section, and
  a real example of admitting an overclaim rather than quietly softening it.

- `[IDEA]` Isolated clDice's effect from the RAdamScheduleFree optimizer's effect. The earlier
  combined ablation (clDice loss + ScheduleFree together) showed a real gain over baseline
  (+0.0123 official score, full 129-case LOSO) but couldn't attribute it to either ingredient
  specifically. Built a "clDice loss, stock SGD" variant as a genuine control to separate the
  two before committing a full training budget to either.

- `[IDEA]` Built a checkpoint-history archiver that labels snapshots by the real epoch number
  read directly from each checkpoint's own embedded `current_epoch` field (verified from
  `nnUNetTrainer.save_checkpoint` source) rather than inferring it from log timestamps. nnU-Net
  itself only ever keeps a "best" and a "latest" checkpoint — no history — so without this,
  questions like "did the model actually improve between epoch 400 and 700" are unanswerable
  after the fact.

- `[IDEA]` Discovered, via direct timestamp analysis of the original 1000-epoch run's training
  log, that `checkpoint_best.pth` froze at epoch 639 (EMA pseudo-dice) and never improved again
  through epoch 999 — 360 epochs, over a third of the run, with zero measurable gain. Confirmed
  two independent ways: local EMA tracking never beat the epoch-639 value again, and two real
  Kaggle submissions (the "epoch ~640 snapshot" and the "final 1000-epoch" checkpoint) scored
  within 0.00004 of each other on both public and private leaderboards — because they were
  literally the same weights. Directly motivated switching every subsequent long run's epoch
  budget from 1000 down to ~700 (or ~350 for faster-converging ScheduleFree variants), a
  real, evidence-backed efficiency decision rather than a guess.

- `[PUSHBACK]` User corrected an initial ~60-70s/epoch estimate for a planned 1000-epoch run to
  ~30s/epoch. Checked the actual original run's log rather than defend the estimate: confirmed
  ~34-35s/epoch for the stock CE+Dice config — the higher number had been unknowingly carried
  over from the clDice ablation's timing, which has real, separately-confirmed overhead from
  the clDice loss's own skeletonization computation (~69s/epoch), not representative of
  stock-loss runs. Precision here mattered: it changed the day's whole GPU scheduling plan.

- `[PUSHBACK]` User: "cant we use /dev/vdb" during a disk-full incident. Correct catch — the
  root filesystem (255GB, /dev/vda4) had filled to 100%, but a much larger, nearly-empty
  second volume (/dev/vdb, 1TB, mounted at /mnt/workspace) was sitting unused because the
  nnU-Net scratch trees had been placed under /tmp by original convention. Migrated the whole
  ~150GB tree across volumes rather than just deleting things to survive — the more durable fix.

- `[PROBLEM]` Root disk filled to 100%, breaking command output and killing an in-flight
  preprocessing job. Root cause was compound: nnU-Net's own scratch data living on the small
  system volume, plus a checkpoint archiver (see above) that had no retention policy and
  accumulated 41GB from copying a full 1.2GB checkpoint on nearly every epoch of a 100-epoch
  run. Fixed by deleting disposable derived caches (never the only copy of anything — verified
  explicitly when asked), trimming the archiver to 8 representative snapshots, and migrating
  the scratch tree to the larger volume.

- `[PROBLEM]` The disk migration silently broke a dataset's raw-data symlinks: they pointed at
  absolute paths under the now-deleted `/tmp` tree (a symlink-to-symlink chain), and `rsync -a`
  preserves symlink targets verbatim rather than rewriting them. Surfaced as an opaque "not all
  training cases have a label file" error on the preprocessing retry. Fixed by re-pointing every
  case's symlinks directly at the original source data (one hop, matching how the rest of the
  pipeline already did it) instead of through an intermediate copy — more robust than what broke.

- `[PROBLEM]` nnU-Net's cascade training silently expects previous-stage (lowres) predictions
  covering the *entire* training set, in a directory keyed on the exact trainer class name that
  will train the cascade — but nnU-Net's own automatic export only ever covers the validation
  split. No existing full-coverage set from a well-converged checkpoint existed locally.
  Rather than spend real GPU hours regenerating one cleanly, merged what was already on disk:
  kept 129 real high-quality predictions from the best checkpoint untouched, filled the
  remaining 657 cases from a weak 5-epoch fallback model's predictions. A disclosed, real
  quality inconsistency in the cascade's coarse-hint channel, not a hidden shortcut.

- `[CAVEAT]` The Nelder-Mead ensemble-weight search over arunodhayan's checkpoints was launched,
  then killed mid-run once it was realized the 129-case LOSO set isn't actually held out
  relative to those specific checkpoints (trained on `FOLD="all"`) — the same contamination
  category already documented for arunodhayan's checkpoint generally, just re-surfacing in a
  new context (weight *search*, not just scoring) that made it worse: fitting a hyperparameter
  to data the models had memorized, not just reporting an inflated number. Caught and stopped
  before the result was ever reported as valid.

- `[IDEA]` Laplacian-pyramid high-pass sub-band as a second input channel (raw CT intensity
  plus `volume - gaussian_blur(volume)`), directly extending the user's own prior published
  work (M-SCQALE) rather than borrowing a technique from arunodhayan's public solution.

- `[PROBLEM]` Smoke-testing vesuvius-surface's genuinely-never-executed `nnUNetTrainerSkeletonRecall`/`nnUNetTrainerAffinity` surfaced two real, independent bugs before either could waste a 100-epoch run: (1) a namespace collision -- the repo's own top-level `training` package shares its name with nnU-Net's internal `nnunetv2/training/` subpackage, and nnU-Net's trainer-discovery mechanism temporarily inserts its own root onto `sys.path[0]` (ahead of our PYTHONPATH) before importing candidate trainer modules, so `import training.trainers` inside their own registration shim silently resolved to nnU-Net's *internal* package instead of theirs (`ModuleNotFoundError: No module named 'training.trainers'`). Fixed by pre-importing our `training.trainers` package in a thin wrapper script before nnU-Net's own entry point runs -- Python's import cache then serves the correct module on every later lookup, since it never re-walks sys.path for an already-cached name. (2) Both trainers declare `__init__(self, *args, **kwargs)` instead of matching nnU-Net's real explicit signature -- breaks nnU-Net's own `self.my_init_kwargs` bookkeeping, which introspects `inspect.signature(self.__init__)` and indexes the *caller* frame's locals() by those parameter names (`KeyError: 'args'`). Fixed by giving both trainers (and our own derived `_100epochs`/combined subclasses, which had copied the same pattern) explicit signatures matching nnU-Net's actual `__init__`. Also caught and fixed a smaller, unrelated bug in their own `register_nnunet_trainers.py`: a `textwrap.dedent()` call that silently produces invalid (inconsistently-indented) Python when the source lines don't share a common prefix, breaking the registration shim it generates. Concrete vindication of insisting on a real smoke test before committing GPU time to untested code, twice over -- neither bug was hypothetical.

- `[PROBLEM]` The namespace-collision fix (pre-importing `training.trainers` in a wrapper
  script) only protected invocations that went through that specific wrapper. Highpass-only
  training -- launched via a plain, unwrapped `nnUNetv2_train` call for a completely
  unrelated stock trainer -- crashed on the *same* collision anyway, because nnU-Net's
  discovery mechanism scans and imports every file in its trainer directory regardless of
  which trainer is being searched for, so the broken shim file was reached either way. Worse,
  the wrapping shell script didn't check that specific command's exit code, so it printed
  "DONE." and exited normally despite the training never actually starting -- a silent
  failure that would have let the master continuation chain proceed to score a model that
  was never trained. Fixed properly this time: installed a `sitecustomize.py` in the conda
  environment (a standard Python startup hook) that pre-imports `training.trainers` before
  *any* code runs in that environment, making the collision structurally impossible
  regardless of entry point -- verified against both a stock trainer and a vesuvius-surface
  one. Also added an explicit check for the real completion marker in the log (not just
  "did the wrapper process exit") before letting the chain proceed to the next stage.

- `[PROBLEM]` Fine-tuned two separate checkpoints (arunodhayan's two fullres ensemble
  members, "A" and "B") back-to-back using the identical trainer name, dataset, config, and
  `nnUNet_results` path for both -- nnU-Net keys its output directory on exactly those four
  things, so B's fine-tune silently overwrote A's completed checkpoint at the same path the
  moment it started saving its own. A's result (73+ minutes of real training, already
  confirmed working via its own validation pseudo-dice) was gone by the time this was caught
  -- the chain went straight from "A done" into "B starting" with no step in between to back
  A's checkpoint up. Fixed two ways: immediately copied B's checkpoint to a separate,
  uniquely-named location before anything else could touch it, then re-ran A into an
  isolated `nnUNet_results` root (`nnUNet_results_ensembleA_ft`, not the shared default) so
  the same collision can't happen again for any future fine-tune in this pipeline. Concrete
  example of a mistake that cost real, non-recoverable compute time (not just a bug caught
  before it ran) -- worth naming plainly rather than glossing over.

- `[PROBLEM]` Tried to parallelize the final official-metric scoring pass (129 cases x 4
  conditions) two different ways, hit a real bug each time. First: 16 `multiprocessing.Pool`
  workers each spawned their own ~25-thread BLAS pool (400 threads vs 22 cores) because
  `OMP_NUM_THREADS` wasn't capped before importing numpy -- the exact mistake already
  documented in our own engineering-learnings file from earlier the same night, made again in
  a new script. Impact: 8.5 minutes produced zero completed cases. Fixed and bumped workers to
  20 for extra speed -- which triggered a worse failure: the topology/Betti-matching
  computation turned out to need up to ~5GB RAM per case (not obvious from the CPU-bound
  framing), so 20 concurrent workers blew through the box's 117GB and the kernel OOM-killer
  cascaded, killing 5+ workers within 4 seconds AND the tmux server itself (same cgroup),
  which took every tmux session down at once. Recovered cleanly: confirmed via `ps aux` that
  all other paused background jobs survived as orphaned processes (nothing else was lost),
  restarted tmux, dropped workers to 8 (~40GB worst case), verified real per-worker RSS this
  time before trusting it. Good concrete example for both a "parallelization has real, easy-
  to-miss failure modes beyond just wall-clock" slide and a "documented lessons don't apply
  themselves -- still have to check" slide.

---

## How to use this for slides

Not every line needs its own slide. Good candidates for "novelty" slides: the test-set
probing research, the seeded-trainer fix, the parallelized scorer, the contamination
discovery + before/after-delta design. Good candidates for "communication/limits" slides: the
71/29 caveat, the 100-epoch convergence caveat, the pseudo-dice-vs-official-metric distinction.
The pushback entries are the strongest evidence for the AI-disclosure section specifically —
they show redirection and revised reasoning, not blind agreement.
