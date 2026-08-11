# Trainer classes -> experiments

Every class here is an `nnUNetTrainer` subclass, registered via `scripts/register_nnunet_trainers.py`
and run via `nnUNetv2_train ... -tr <ClassName>` (or `src/vesuvius_surface/training/run_training.py`
directly). This table maps each one to the experiment it produced, so a result number can
always be traced back to the exact trainer that produced it.

| Class | Experiment | Result |
|---|---|---|
| `nnUNetTrainerSkeletonRecall` (base) | Stage 1 loss definition — DC+CE + skeleton-recall auxiliary loss, ported from MIC-DKFZ/Skeleton-Recall (Kirchhoff et al., ECCV 2024) | — |
| `nnUNetTrainerSkeletonRecall_100epochs` | 100-epoch 5-way candidate comparison | **winner**, 0.5307 |
| `nnUNetTrainerSkeletonRecall_20epochs` | Diagnostic isolating the arunodhayan-finetune regression's cause (raw CT, no highpass/affinity) | 0.5768 |
| `nnUNetTrainerSkeletonRecall_700epochs` | From-scratch winner-extension of the 100-epoch winner | 0.5671 (LOSO), real Kaggle submission |
| `nnUNetTrainerAffinity` (base) | Stage 2a auxiliary-head definition — long-range affinity head on the full-resolution decoder stage, discarded at inference | — |
| `nnUNetTrainerAffinity_100epochs` | 100-epoch 5-way candidate comparison | 0.5226 (lost to skeleton-recall) |
| `nnUNetTrainerAffinity_700epochs` | Prepared as the winner-extension trainer *if* affinity had won the 100-epoch comparison (per its own docstring: "if it wins the decision point") | **not run** — affinity lost (0.5226 vs 0.5307) |
| `nnUNetTrainerSkeletonRecallAffinity` | Combined Stage 1 + Stage 2a, Track A design — from-scratch combination. Also the *same* trainer class used for the full arunodhayan fine-tune: confirmed directly from the real checkpoints' own embedded `trainer_name` metadata, not assumed. That fine-tune adds a highpass input channel (`scripts/data_prep/highpass.py`) via a separate preprocessed dataset (`Dataset102_VesuviusSurfaceHighpassOnly`) and `-pretrained_weights`, rather than anything in this class itself. | exploratory (from-scratch use); fine-tune use, negative: ensemble 0.7029→0.5172, cascade 0.7198→0.5208 |
| `nnUNetTrainerSkeletonRecallAffinity_1epoch` | Smoke test for the combined trainer above (forward/backward/checkpoint sanity only) | not a real result |
| `nnUNetTrainerSkeletonRecallCascadeLastLayers_10epochs` | Phase 4, item 14 — last-layers-only (0.07% of params trainable) fine-tune of arunodhayan's real cascade checkpoint. **The one genuinely positive fine-tuning result of the whole project.** | 0.7248 vs 0.7198 zero-shot (+0.0050), full 129-case LOSO |
| `nnUNetTrainerSkeletonRecallCascadeLastLayers_1epoch` | Smoke test for the trainer above (verifies the frozen-backbone + cascade + skeleton-recall combination actually trains before committing to the real 10-epoch run) | not a real result |
| `nnUNetTrainerSeeded` (base, vendored from `baselinerun/`) | Reproducibility fix: stock nnU-Net sets no RNG seed anywhere (verified from `nnUNetTrainer.__init__` and the `nnUNetv2_train` CLI arg list directly). Seeds torch/numpy/random via nnU-Net's own `nnUNet_extTrainer` external-trainer mechanism. Base class for every from-scratch baseline result in Phase 2, item 7. | — |
| `nnUNetTrainerSeeded_ClDice_ScheduleFree` (vendored from `baselinerun/`) | Phase 3, item 11 — 100-epoch 5-way candidate comparison. Replicates arunodhayan's exact loss+optimizer recipe (DC+CE + 0.2·clDice, RAdamScheduleFree lr=1e-3, no LR schedule — verified from his public `train.py`, not guessed) on our own leakage-free 657-case LOSO split, isolating loss+optimizer from his other changes (boosted rotation aug deliberately excluded). | 0.5285 (lost to skeleton-recall, 0.5307) |
| `nnUNetTrainerSeeded_ClDice_ScheduleFree_350epochs` | Prepared as the winner-extension trainer *if* clDice+ScheduleFree had won the 100-epoch comparison (350 not 700 epochs: ScheduleFree's constant-LR design converges faster, and clDice's ~69s/epoch overhead vs. ~35s/epoch stock means 350 epochs costs about the same wall-clock as 700 of a cheaper candidate) | **not run** — lost the 100-epoch comparison |
| `STUNetTrainer` (base, vendored from `baselinerun/`) | STU-Net (TotalSegmentator-pretrained, architecturally distinct from nnU-Net's own CNN) fine-tuning family — chosen specifically because it's provably leak-free (never seen a Vesuvius volume), unlike fine-tuning arunodhayan's or m7's checkpoints. Contains a real upstream bug fix (wrong positional-arg order) documented in-file. | — |
| `STUNetTrainer_base_ft_30epochs` | Phase 3, item 10 — 30-epoch fine-tune of STU-Net-B (58M params). The two shallowest encoder stages (`conv_blocks_context.0`, `.1`) are frozen at load time by `scripts/finetune/run_finetuning_stunet_freeze_early.py`, which wraps `nnUNetv2_train`'s own `-pretrained_weights` loading — the freezing is not in this class itself. | **negative**, 0.4629 vs. 0.5575 best_baseline (n=129 full LOSO); ruled out |

`nnUNetTrainerSeeded`/`STUNetTrainer`/their variants above were vendored from `baselinerun/`,
a separate from-scratch training pipeline. The arunodhayan full cascade fine-tune (the other
negative full-fine-tune result) reuses `nnUNetTrainerSkeletonRecallAffinity` above via
`-pretrained_weights`, not a separate trainer class — trainer attribution was confirmed from
a real checkpoint's own embedded metadata, not assumed.
