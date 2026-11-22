# NEC-04C: preregistered matched source-parity Looper curriculum pilot

**Status:** prospective six-root calibration-world training protocol, issue [#296](https://github.com/MoralityLabAI/LoopedTransformers/issues/296). The complete registration, original world-root cohort, score selection and source SHA receipt are committed to the branch before any new native optimizer update. Training and evaluation run from a single immutable GitHub commit and its preflight artifacts.

## Parent and purpose

The original NEC-04B six-root native 12-loop pilot failed its frozen native-headroom target; [merged #295](https://github.com/MoralityLabAI/LoopedTransformers/pull/295) and successful Actions run 44992324562 preserve the original 48 native checkpoints. The read-only NEC-04C historic branch audit in [merged #297](https://github.com/MoralityLabAI/LoopedTransformers/pull/297) independently reproduced the NEC04B 900-step composed parity error **1,139/2,272 correct (50.13%)** and the alternate-branch score **1,655/2,336 correct (70.85%)**, using only archived training and calibration-world validation cases. Parity remained at chance across all four original budgets.

This next study asks whether the **original 32D, twelve-visit PEF07 Looper** can learn the same context-selected five-focal Boolean parity teacher with a task-aligned source-only sampling curriculum.

## Immutable root cohort and task

Original NEC04A **calibration WORLD roots 6..11** are newly untouched by NEC04B model optimization. They are precisely two each of the symbolic easy/medium/hard classes, with all original world, program and context SHA256 commitments preserved from NEC04A run 44983509977, source corpus SHA256 82c80643d3c99871913f5228f1a1cde1eec4484787c1784872b508349793e9e0. Every world retains the first 80 whole source context groups, 32 focal masks per group, giving 2,560 original source histories. The 24 previously allocated **owned developmental contexts in these six CALIBRATION worlds** give 768 validation histories, scored only after both arms' weights and full optimizer states are frozen. The 24 reserved contexts per calibration world remain unopened.

These owned validation-context groups belong to existing calibration roots. The separate original 24 developmental WORLD roots and 16 confirmation WORLD roots remain uninstantiated. No old NEC03Z sealed source group or NEC03P recruited-world model is scored.

Both arms use the exact original NEC04A composed Boolean program, preserving the originally committed outer MUX and five-bit XOR parity branch. No intermediate synthetic target or new task label is introduced. An independent scalar oracle verifies all original source and owned validation labels before training.

## Two equal-budget model arms

Same model initialization within root: the original NEC04B SHA-derived native composed Looper seed formula applied to root6..11, replaying **identical initial weights** between arms. Each receives one continuous AdamW (LR 0.003, no weight decay, gradient clipping 1.0), batch64, CPU Torch 2.10.0, 12 recurrent network updates per inference and **900 optimizer updates**.

- Vanilla: 900 steps of uniformly sampled source rows, with replacement and a fixed Torch generator, matching the original NEC04B training recipe.
- Parity curriculum: exactly **240 steps** with all 64 examples sampled with replacement from original source contexts whose MUX selector equals 1, using their original composed truth labels. Then **660 steps** of original full-task labels under source-only selector-balanced minibatches with 32 parity and 32 alternate rows each, shuffled deterministically. No optimizer reset or model architecture change.

Save per-arm native weights and full optimizer states after **240 and 900** updates. All four original model files and their SHA256 manifest must be frozen before owned validation scoring. Retain the complete source/validation logits and original model weights in root artifacts. An independent native replay checks every byte SHA, model hash, optimizer step count, native logits, original scalar labels and conditional branch correctness.

## Frozen primary endpoint and stop

At exactly 900 optimizer steps, compute the paired within-world difference in validation accuracy on the original context MUX parity branch, curriculum minus vanilla. The primary summary is its **macro mean across six independently seeded calibration world roots**. Require improvement **at least +10 percentage points** with at least **four of six** roots strictly positive. The predeclared retention floor is macro mean curriculum-minus-vanilla alternate-branch validation accuracy **at least −5 percentage points**.

Publish every root and arm, including failing arms, the 240-step exploratory checkpoints, class priors, balanced accuracy and complete source-versus-validation scores. Original parity target labels are exactly balanced in every complete 32-mask context block, so constant-class parity baseline is 50%. Per-root uncertainty is descriptive at n=6. Pooled masks/contexts are within-root correlated and cannot substitute for six independent worlds.

Passing this training-curriculum pilot only motivates a **new preregistered native-headroom/SAE reconstruction and physical-negative-control qualification protocol**. It does not by itself reopen the old NEC04B headroom-band failure or justify an SAE necessity claim. If the conditions fail, record STOP_PARITY_CURRICULUM_CALIBRATION_PILOT with the complete frozen denominator.

All old and new protected world-root cohorts remain sealed and NEC03P's earlier 69/88 source-qualified recruits continue to have zero causally measured sham-root outcomes. A valid functional and physically inert null plus at least 59 independently eligible causal sham worlds remain prerequisites for a later 95/95 directional tolerance claim. Throughout, **FSI-01: REDESIGN_OR_PILOT_ONLY**.

## Reproducibility

The structured registered contract is NEC04C_TRAIN_LOCK.json. The dedicated workflow is .github/workflows/nec04c-curriculum.yml. Its preflight uploads exact training source/owned validation oracles, the lock SHA256, all source-code SHA256 hashes and the original GitHub checkout commit SHA **before launching the six root matrix jobs**. Every root job and independent auditor rejects any code, root or teacher SHA discrepancy. The final pure-standard-library aggregator requires all six original independent root audits before scoring the frozen pilot gates.
