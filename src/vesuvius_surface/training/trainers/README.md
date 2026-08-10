# Trainer classes -> experiments

Every class here is an `nnUNetTrainer` subclass, registered via `scripts/register_nnunet_trainers.py`
and run via `nnUNetv2_train ... -tr <ClassName>` (or `src/vesuvius_surface/training/run_training.py`
directly). This table maps each one to the numbered experiment it produced in
`experiment_summary.md` (phases/items refer to that file), so a result number can always be traced
back to the exact trainer that produced it.

| Class | Experiment | Result |
|---|---|---|
| `nnUNetTrainerSkeletonRecall` (base) | Stage 1 loss definition (research_log.md §13) — DC+CE + skeleton-recall auxiliary loss, ported from MIC-DKFZ/Skeleton-Recall (Kirchhoff et al., ECCV 2024) | — |
| `nnUNetTrainerSkeletonRecall_100epochs` | Phase 3, item 11 — 100-epoch 5-way candidate comparison | **winner**, 0.5307 |
| `nnUNetTrainerSkeletonRecall_20epochs` | Phase 3, item 13 — diagnostic isolating the arunodhayan-finetune regression's cause (raw CT, no highpass/affinity) | 0.5768 |
| `nnUNetTrainerSkeletonRecall_700epochs` | Phase 4, item 15 — from-scratch winner-extension of the Phase 3 winner | 0.5671 (LOSO), real Kaggle submission (see docs/checkpoints.md) |
| `nnUNetTrainerAffinity` (base) | Stage 2a auxiliary-head definition (research_log.md §13) — long-range affinity head on the full-resolution decoder stage, discarded at inference | — |
| `nnUNetTrainerAffinity_100epochs` | Phase 3, item 11 — 100-epoch 5-way candidate comparison | 0.5226 (lost to skeleton-recall) |
| `nnUNetTrainerAffinity_700epochs` | Prepared as the winner-extension trainer *if* affinity had won the 100-epoch comparison (per its own docstring: "if it wins the decision point") | **not run** — affinity lost (0.5226 vs 0.5307) |
| `nnUNetTrainerSkeletonRecallAffinity` | Combined Stage 1 + Stage 2a, Track A design (research_log.md §13's "two tracks" table) — from-scratch combination, architecturally distinct from the separate combined recipe used in the arunodhayan full-finetune attempt (Phase 3, item 12, which lived in `finetune/`, not here) | exploratory; not tied to a specific `experiment_summary.md` item — see research_log.md §13 for the design rationale |
| `nnUNetTrainerSkeletonRecallAffinity_1epoch` | Smoke test for the combined trainer above (forward/backward/checkpoint sanity only) | not a real result |
| `nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs` | Phase 4, item 14 — last-layers-only (0.07% of params trainable) fine-tune of arunodhayan's real cascade checkpoint. **The one genuinely positive fine-tuning result of the whole project.** | 0.7248 vs 0.7198 zero-shot (+0.0050), full 129-case LOSO |
| `nnUNetTrainerSkeletonRecallCascadeLastLayers_1epoch` | Smoke test for the trainer above (verifies the frozen-backbone + cascade + skeleton-recall combination actually trains before committing to the real 10-epoch run) | not a real result |

For the *negative*-result trainers (STU-Net fine-tune, the full arunodhayan highpass+skeleton-recall
+affinity fine-tune, Phase 3 items 10 and 12) see `nnUNetTrainerSeeded` and the arunodhayan cascade
driver — those live under `training/finetune/` and `third_party/arunodhayan_source/` respectively
(vendored from `baselinerun/` and `finetune/`), not in this directory, since they use a genuinely
different training entrypoint (not nnU-Net's own `-tr` trainer-class mechanism for the STU-Net case).
