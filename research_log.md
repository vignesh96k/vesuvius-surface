# Research log

Vesuvius Challenge — Surface Detection. A running narrative of what we tried,
what the evidence said, and what we changed our minds about. Commits give the
*what*; this file is the *why*.

---

## 1. Getting the repository to run

The package directories (`src/data`, `src/eda`, `src/utils`) did not match the
names being imported (`datasets`, `analysis`, `utils`), so nothing imported
cleanly. `scripts/validate_dataset.py` failed with `TypeError: 'module' object
is not callable` — it was importing the *module* `validate_dataset` and calling
it as a function.

Fixed by aligning every import to the real directory names and putting `src/`
on `sys.path` in the entry-point scripts, rather than renaming directories.

`probe_volume()` was loading entire TIFF volumes just to read their shape. It
now reads TIFF series metadata, with `include_extrema=True` to opt back into a
full load when min/max are actually needed.

**Gotcha worth recording:** `.gitignore` contained `data/`, which silently
excluded `src/data/`. A new module was committed locally, pushed, and then
`ModuleNotFoundError` on the remote box. Changed to `/data/` so only the
top-level data directory is ignored.

## 2. What the dataset actually is

`train.csv` lists 806 ids but only 786 have files. The other 20 sit under
`deprecated_train_images/` — competition-retired samples, not a broken
download. The validator reports these as warnings rather than errors.

| Property | Value |
|---|---|
| Volumes | 786 usable (806 listed) |
| Shape | ~320×320×320 (not H×W×65) |
| Labels | 0 background, 1 surface, 2 ignore |
| Scrolls | 26002, 26010, 34117, 35360, 44430, 53997 |

Scroll sizes are very uneven: 34117 has 376 volumes, 44430 has 16, 53997 has
13. This imbalance comes back to bite us repeatedly.

On a representative volume the class split was 37% background, 4.9% surface,
58% ignore. The positive class is rare and most of the volume is unlabelled.

A 3D distance transform put median papyrus sheet thickness at **2 voxels**
(mean 2.33, p90 2.83). This number turns out to matter enormously later.

## 3. Choosing an approach

Target: reproduce the [1st-place solution][1st], which is an nnU-Net ensemble.
Rationale — nnU-Net self-configures, the winners published both their method
and their weights, and the remaining gains they documented came from
post-processing rather than architecture.

Built the export to nnU-Net v2 raw format with `--mode symlink`, so the
~100 GB of volumes are not duplicated.

`nnUNetv2_plan_and_preprocess` preprocesses every configuration it generates,
which meant sitting through 2D preprocessing we had no intention of using. The
setup script now passes `-c 3d_fullres` by default; `--all` restores the old
behaviour. The `no spacing file found, assuming (1,1,1)` warning is expected —
this is micro-CT with no physical spacing metadata, and isotropic spacing is
the right assumption.

Resulting plan: patch 128³, batch 2, median image size 320×314×314.

## 4. The published checkpoint

Fetched `scrollprize/surface_m7_nnunet` and installed it into an
`nnUNet_results` layout. Its `dataset.json` reports 786 training cases, labels
`{background: 0, surface: 1, ignore: 2}` and `.tif` file ending — all identical
to our export, which is good indirect evidence that our case list matches
theirs.

Two practical snags:

- The snapshot ships `checkpoint_best.pth`; `nnUNetv2_predict` defaults to
  `checkpoint_final.pth`. The wrapper now detects which exists.
- No `splits_final.json`. Which fold-0 validation set the checkpoint never saw
  is therefore **not recorded anywhere**. See section 7.

## 5. The metric is not Dice — the pivotal finding

We had been assuming a Dice-like objective and had sketched an evaluation
around per-scroll Dice plus hole counts. That was wrong. The competition scores:

```text
Score = 0.30 * TopoScore + 0.35 * SurfaceDice@2.0 + 0.35 * VOI_score
```

- **SurfaceDice@2.0** — boundary proximity, forgiving to 2 voxels.
- **VOI_score** — instance split/merge over connected components.
- **TopoScore** — Betti matching over components, tunnels, cavities.

Volumetric Dice appears nowhere. Roughly two thirds of the weight is
topological, and the geometric third tolerates 2 voxels of error — the same
order as the sheet thickness we measured in section 2. A change that improves
overlap while fusing two sheets can *lower* the score.

We installed the official implementation (Kaggle dataset
`sohier/vesuvius-metric-resources`, requiring a C++ Betti-Matching-3D build)
rather than reimplementing it. Reading its actual signature corrected two
things we had written down from the competition prose:

| Parameter | We assumed | Package default |
|---|---|---|
| `voi_alpha` | 0.3 | **1.0** |
| `voi_connectivity` | 6 | **26** |
| ignore handling | ours to solve | **`ignore_label=2`, built in** |

The third correction deleted code. We had built a `neutralize` / `background`
switch to decide how to handle unlabelled voxels; the metric handles them
natively, so raw labels are passed straight through and the switch is gone.

We call `compute_leaderboard_score` with its own defaults and override nothing,
so local numbers stay in parity with the leaderboard scorer.

**Why having this matters:** the winners stated they had *no validation set*
and tuned thresholds by spending submissions, later admitting they overfit
public LB and that 0.35–0.40 would have been better than the 0.20/0.26 they
shipped. An offline scorer is the capability they lacked.

## 6. First measurements

Three holdout volumes, published checkpoint, no post-processing:

| Term | Score | Weighted | Ceiling | Headroom |
|---|---:|---:|---:|---:|
| SurfaceDice | 0.9852 | 0.3448 | 0.35 | **0.005** |
| VOI | 0.3226 | 0.1129 | 0.35 | **0.237** |
| TopoScore | 0.4372 | 0.1312 | 0.30 | **0.169** |
| **Composite** | | **0.5889** | 1.00 | 0.411 |

The parts sum exactly to the composite, confirming the harness reads the
report correctly. For sanity: the winners reported 0.572 public / 0.596 private
for this checkpoint without post-processing, so 0.5889 on different volumes is
the right neighbourhood.

**Geometry is solved; topology is the entire remaining game.** SurfaceDice sits
at 98.5% of its ceiling. 99% of available points are in VOI and TopoScore. This
is the 2-voxel-tolerance effect predicted in section 2 — a sheet riddled with
pinholes still scores near-perfect on position, while every pinhole is a tunnel
to TopoScore and enough of them sever a component for VOI.

It also explains the winners' ablation, where every gain was topological:
remove small components, plug holes, patch large holes, binary closing, fill
holes, 0.572 → 0.606. Threshold tuning was worth "1e-3 level" by comparison.

Scoring costs ~63 s/volume.

## 7. Trying to verify the holdout — and failing honestly

We had been calling the 158 fold-0 cases "leakage-free". We never established
that. We *reconstructed* what nnU-Net's default seed would produce over our
sorted case ids; nothing in that computation touched the checkpoint. The model
card documents the checkpoint as fold 0, but not which split defined fold 0.

Two attempts to close the gap:

**Structural.** Compare `dataset_fingerprint.json` case ordering. First run
compared our raw shapes against their post-crop shapes — apples to oranges,
0/786 matches with a uniform (320,320,320) vs (320,314,314) offset. A uniform
offset actually argues *for* a matching case list; scattered mismatches would
argue against. Rewritten to compare our own `plan_and_preprocess` fingerprint
against theirs, which is like-for-like. *Not yet run.*

**Empirical.** Score 20 volumes we claim were *in* training; a model should
score higher on data it has seen. Comparing scrolls present in both samples:

| Scroll | Claimed-train | Claimed-holdout | Δ |
|---|---:|---:|---:|
| 34117 | 0.6034 (n=11) | 0.6244 (n=2) | −0.021 |
| 35360 | 0.5309 (n=7) | 0.5179 (n=1) | +0.013 |

Two hundredths, opposite directions, tiny samples. No detectable gap. This is
equally consistent with "the model doesn't overfit measurably" and "our split
is meaningless" — if the split were wrong, both samples would be random
mixtures and would score alike, which is what we see.

**Conclusion: the holdout is unverified.** It should not be described as
leakage-free, and its scores are not a leaderboard estimate. They remain valid
for comparing *our own configurations* against each other, which is what
threshold and post-processing tuning needs.

## 8. The blind spot that changed the plan

The train probe produced the first ever measurement of scroll 44430:

| Scroll | Composite | TopoScore | SurfaceDice |
|---|---:|---:|---:|
| 26010 | 0.6388 | 0.4757 | 0.9931 |
| 34117 | 0.6034 | 0.4764 | 0.9858 |
| 35360 | 0.5309 | 0.2751 | 0.9673 |
| 44430 | **0.4413** | **0.1324** | **0.8860** |

44430 is far the worst — TopoScore 0.13 against ~0.48 for the good scrolls, and
the only scroll where even SurfaceDice degrades. The per-scroll spread is 0.20.

And 44430 has **zero cases in our 158-volume holdout**, while 34117 — one of
the easier scrolls — supplies half of it. So the holdout is both optimistic and
blind to our worst failure mode. Scoring all 158 would cost 2.8 hours to
produce a number that hides the thing most worth fixing.

This reframes what the evaluation set is for. For tuning, we compare
configurations on a *fixed* set of predictions, so train/validation membership
largely cancels out. What cannot cancel is unrepresentativeness. **Scroll
coverage is worth more to us than fold purity.**

Also visible: 35360 has a merge skew (merge 1.28 vs split 1.05), meaning fused
sheets specifically, which calls for different post-processing than fragment
removal.

## 9. Architecture: considered and rejected

**Question: why not fine-tune a pretrained Swin UNETR 3D instead?**

The direct comparison has been run rigorously. Isensee et al.,
[nnU-Net Revisited][rev] (MICCAI 2024), benchmarked SwinUNETR, nnFormer, CoTr
and Mamba variants against properly-configured CNNs under identical conditions.
SwinUNETR scored below the *original* nnU-Net on all six datasets — by up to
4.8 Dice — while using ~70% more VRAM and ~67% more training time. Their
conclusion was that many published transformer wins came from weak nnU-Net
baselines. Our checkpoint is `nnUNetResEncUNetLPlans`, the ResEnc L preset that
paper recommends as the fair baseline, so m7 is already the strong CNN config.

The stronger argument is that architecture aims at the wrong bottleneck.
SurfaceDice contributes 0.3448 of a possible 0.35 (section 6), so a model with
5 more Dice points would buy at most **0.005** of composite. All 0.41 of
headroom is topological.

There is also direct counter-evidence: in the ScrollPrize repository a
MedNeXt-L model beat the production baseline on SurfaceDice and VOI but lost
badly on TopoScore (0.022 vs 0.084) because smoother predictions generate extra
spurious handles and cavities. Smoother output helps geometry and hurts
topology — the exact trade we do not want.

Deferred, not dismissed: architectural diversity genuinely helps an *ensemble*.
The winners fused four related nnU-Net variants; a structurally different model
would decorrelate more. Worth revisiting if the topology work lands early.

**Better version of the same instinct:** target the loss, not the encoder.
clDice, skeleton recall and Betti-matching losses optimise connectivity and
hole structure directly — the 65% of the score that is actually broken.

## 10. CT foundation models

**Question: is there an "ImageNet for CT" we can fine-tune like a ResNet?**

Yes, several: STU-Net (TotalSegmentator, >100k annotations, 14.6M–1.46B params),
CT-FM (148k CT scans, self-supervised), TAP-CT (105k volumes, DINOv2-style 3D
ViT), MONAI's Swin UNETR SSL (~5k volumes), Models Genesis.

Initially argued against all of them: they are pretrained on human anatomy at
millimetre resolution with Hounsfield semantics, while ours is micro-CT of
carbonised papyrus — thin sheets, no anatomy, micron scale. And we already had
a checkpoint pretrained on this exact task, which strictly dominates generic
pretraining.

That reasoning was overturned by section 11.

## 11. The contamination problem — why we pivoted to STU-Net

m7's model card records the *fold* it trained on but never the *split* that
defined that fold. Consequently **no volume is provably outside its training
set**, and every number we have produced from m7 is unverifiable.

What this does and does not invalidate:

| Experiment | Valid? |
|---|---|
| m7's absolute score, generalisation claims | No |
| "did our fine-tune improve things" | No |
| Comparing post-processing variants on *fixed* m7 predictions | Yes — contamination inflates the baseline but does not flip the sign of an improvement; bias is conservative |

The trap we nearly walked into: **fine-tuning from m7 inherits its
contamination.** A model initialised from m7 has effectively seen whatever m7
saw, so it cannot produce a clean holdout no matter what split we author. The
`finetune_setup` scripts remain useful for chasing score, but not for
measurement.

Training from scratch would be clean but the winners needed 4000 epochs to
reach 0.587 — out of budget.

**STU-Net resolves the dilemma.** It is pretrained on TotalSegmentator, so it
has provably never seen a Vesuvius volume; fine-tuning it against a split we
author yields a genuinely clean holdout, while still starting from pretrained
weights rather than scratch. The paper reports successful CT→MRI and CT→PET
transfer, so CT→micro-CT is at least plausible. It is nnU-Net-based, so it
plugs into the existing pipeline, and being architecturally distinct it doubles
as an ensemble-diversity candidate.

This reverses the section 10 conclusion. Generic pretraining is worth less than
task-specific pretraining *on quality* — but it is worth more *on
measurability*, and measurability was the binding constraint.

Operational note: STU-Net ships its own nnU-Net 2.2 fork, which would replace
the `nnunetv2` the m7 pipeline depends on. `scripts/setup_stunet.sh` therefore
installs into a separate conda env and writes to a separate results tree.

**Deliverable split.** Score and measurement are now separate deliverables:

- m7 + post-processing → leaderboard score, described honestly as building on
  published weights with an unverifiable split.
- STU-Net fine-tuned on our authored split → every measured claim
  (post-processing ablation, threshold sensitivity, per-scroll behaviour). It
  will score lower; its job is to be measurable, not to win.

`scripts/make_scroll_split.py` writes the authored split, in `stratified` mode
(every fold's validation set gets a proportional share of every scroll, fixing
the 44430 blind spot) or `scroll-holdout` mode (leave-one-scroll-out, for a
genuine generalisation number).

## 12. Corrections made along the way

Recorded because they shaped the work:

| Believed | Actually | Cost |
|---|---|---|
| Metric is Dice-like | Topological composite | Would have optimised the wrong thing |
| `voi_alpha` 0.3, connectivity 6 | 1.0 and 26 | Docs only |
| We must handle ignore ourselves | Metric handles it | Deleted a module |
| Holdout is leakage-free | Unverified | Claim withdrawn |
| Raw vs post-crop shapes comparable | They are not | One inconclusive run |
| Generic CT pretraining not worth it | Right on quality, wrong on measurability | Reversed; see §11 |
| Fine-tuning m7 would give a clean holdout | It inherits m7's exposure | Caught before spending GPU days |

[rev]: https://arxiv.org/abs/2404.09556

## Open questions

1. **Is this a Kaggle code competition?** If inference must run inside a
   notebook under a time limit, the local pipeline is not what gets scored and
   packaging is separate work. Unresolved, and the highest-risk unknown.
2. Does the like-for-like fingerprint comparison match? Not yet run.
3. Does nnU-Net's 3-class output need binarising before scoring, or does the
   metric handle predicted 2s? A probe is written but not yet run.
4. Why is 44430 so much harder? 16 volumes only; needs visual inspection.

## Next

Build a scroll-balanced evaluation bench (~40 volumes, ~40 min per experiment),
then work the topological post-processing chain, then tune threshold and
post-processing jointly against the local scorer.

[1st]: https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su
