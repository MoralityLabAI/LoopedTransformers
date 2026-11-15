# RAM-01: CPU/RAM training on the FC-12 Lean gym

Registered 2026-11-14 before RAM-01 model fitting. This is an exploratory training pilot. FC-12 and FC-13 scores and source commits stay unchanged.

## Source and hardware

Use the exact training artifact from LoopedTransformers run 43547762837, artifact 14095156819. ZIP SHA256: d91a12d725ca0a611b4de308c80365bfd3888d88449b03b2c5f07137a1c2025a. Verify the 51-state file SHA256 403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0 and semantic candidate-effect SHA256 bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec. There are 961 concrete scripts on 27 original MathCan theorems. Keep the original Lean 4.35.0 source revision c84e2025334e9a967c1297425a490f98d5b3d189.

Train on CPU with two PyTorch threads, no CUDA. The interactive sandbox has a 4 GiB cgroup memory ceiling and four CPU cores of quota. Log actual peak process RSS, elapsed training time, optimizer steps, parameter count and checkpoint hashes. The installed local runtime is PyTorch 2.10.0+cpu; CI uses the same PyTorch release. Cross-machine bitwise equality is not assumed.

## Four small candidate scorers

All policies consume only compiler-printed local context and goal text, the concrete candidate script, and its registered family. Filenames, theorem names, source proof commands, future states and compiler outcomes are never input features.

The shared token embedding has 2048 buckets. Encode context and conclusion separately, each capped at 96 tokens, plus 24 action tokens; use masked means. Combine these with a learned family embedding. Configurations are flat64 (width 64, feed-forward control), loop32 (width 32, four shared residual passes), loop64 (width 64, four shared passes), loop128 (width 128, four shared passes). Two fixed seeds, 17 and 29, for each. Each head predicts four compiler outcomes: failed/unchanged, single-goal progress, multi-goal progress, and scratch closure. Scratch closure is a supervised label only.

Train each configuration for 400 AdamW updates, batch size 64, learning rate 0.002, weight decay 0.01, gradient norm cap 1.0. Use weighted cross-entropy with class weights proportional to inverse square-root counts, normalized to mean one. No early stopping or hyperparameter choice from evaluation. Rank candidate scripts by 2*P(closure)+P(single-goal progress); multi-goal probability is retained in the outcome model and earns no search utility while the controller cannot manage branches. Native search uses the original twelve-family/context argument order.

## Development split and final fits

For an initial within-corpus source-module check, hold out every state and probe from DeletionMemory and TOECosmology (12 states). Fit on the other five modules (39 states). Report all eight models, with cached-probe top-1 and top-3 progress, executable progress, and closure counts, including the native ordering and a training-only frequency comparator. These are action-prediction metrics, not complete theorem proofs. Keep all four mandatory parameterized action families in the training outcome table and report their observed support. Never split candidates or prefixes of the same theorem across this source-module split.

Then train the same eight final configurations on all 51 FC-12 training states. Keep all final checkpoints. Development scores do not select or change these fits.

## Compiler regression

Evaluate every final policy on the already published 16-case FC-13 original-theorem regression suite. Its previous outcomes are known; this is a post-hoc regression comparison, not a new pristine holdout or a replacement FC-13 result. Keep the pinned source selector e8a780254b5cf41d479095f4f22bcfc4aa76ed72. Both learned and fixed-native policies use the identical context-generated candidate grammar, a 20-second deadline per theorem, at most 12 actual Lean subprocess calls, and at most three goal advances. Count scratch and final compiler processes against the same budget. Explicitly abstain at multiple goals. Accept a theorem only after a separate full-source compile of the selected script without sorry, before the deadline; print and audit the theorem's axiom dependencies. Cache no compiler verdicts across policies. Archive failures and exact candidate source hashes. Any rerun or implementation repair after viewing scores is labeled separately.

No language-model pretraining, unseen mathematical discovery or general Lean competence is claimed from this small policy-training exercise.
