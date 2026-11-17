# COE-09 registration: label alignment and norm-matched dark-fiber causal tests

**Date:** 2026-11-15 (Asia/Manila). **Source branch:** `MoralityLabAI/LoopedTransformers@a7104071c05b9cac4aa01bae25c29f2fd5c381c0` (current main at registration). All native TRM evaluation settings fixed before downloading or observing COE-09 held-out activations. COE-07 and COE-08 remain unchanged.

## Motivation

The K9 code separation score in COE-07 is invariant under permutation of the nine digit labels. COE-08 showed a truth-conditioned oracle telemetry gain compatible with its shuffle null. Both warrant a *label alignment* test on label-free latent activations and a norm-matched causal perturbation.

## Source locks

- Model: `gaoxin492/TinyRecursiveModels-Sudoku-Extreme-att` `step_21700`; SHA-256 `932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3`, verified before deserialization by COE-07's native loader.
- Benchmark: original 256-row LoopScope-02 bundle `59459ae0bb60b75db5be017e5f41adead391f2abb0b8fb043cd331c0ee043811` (its own canonical bundle digest). Source package `mechinterp/loopscope02/results/benchmark_bundle.json` is immutable.
- COE-07 selection: first board for each digit 1..9 in each of four strata; 36 boards. Fit: `S1_naked1` and `S3_prop3plus`. Held out: `S2_prop2`, `S4_stalled`. Eight recurrent ACT visits. For each of `visible`, `dark`, `full`, `scratch`, all four *existing* COE-07 readout heads are preserved exactly.
- Input: `native_activations.npz` plus `receipt.json` from the COE-07 SHA-pinned run. Every activation tensor and the `heldout_codes` list is independently validated by SHA-256 against the original receipt. A missing or failed pin is a hard stop.

## Registered label-alignment outcome

For each frozen head, encode all fit and heldout states using its original threshold. Each digit has two fit-stratum prototypes. Decode every heldout board by the smallest distance to its fit prototypes, tie-broken by smallest digit, with five fixed distances: Hamming, minimum cyclic Hamming, minimum cyclic-and-reversal Hamming, absolute positive-count difference, and final-bit difference. Head selection and thresholds are never tuned on heldout data.

One result comprises 18 heldout boards and their true target-digit accuracy, separately for S2 and S4. The conditional null consistently permutes the *training* digit-to-prototype association while leaving heldout label assignments unchanged (2,000 fixed-seed permutations per head). Report accuracy, group counts, shuffle mean, interval and upper-tail p-value. Report K9 edge-retention separately as a capacity diagnostic. Every report states the board count; the pair-level 72 K9 edges must never be used as a sample size for significance.

**Registered efficacy gate:** for the dark `cyclic_gain` head, a cyclic-invariant test accuracy of at least 8/18, one or more correct labels in each heldout stratum, and a conditional shuffle upper-tail p-value <= 0.05. This is a stringent pilot gate, not a claim of generalization to unseen base-model training samples or a mechanistic result.

## Registered causal outcome

Apply COE-07's fit-selected `cyclic_gain` direction after native visit **four**, only to the evaluator target cell. Continue to visit eight without modifying weights. Preset rotation angle `±0.05` rad in dark `z_H`, visible `z_H`, scratch `z_L`. Same-angle random tangent controls use seeds **303** and **317**, fixed independently of heldout activations. All states stay on the original norm sphere. A dark-sector intervention must preserve instantaneous head logits to `2e-4`, with relative vector-norm error < `2e-5`. Report clean correctness, clean-to-wrong and wrong-to-correct transitions, prediction flips, mean true-digit logit-margin change, and maximum geometry errors for each arm. Any geometry failure aborts the report.

An observed selected-versus-random difference is a small paired sample diagnostic, and must be replicated on independent puzzles and checkpoint variants before being called a causal mechanism. A comparison is meaningful only for models whose clean source accuracy passes the existing LoopScope admission gates. No hypothesis selection occurs on test outcomes.

## Executed synthetic control (separate provenance)

The optional COE-04 toy replay uses three frozen, locally trained 24D tied-attention checkpoints (seeds 2406-2408), and their previously selected one-bit heads. Four distractor contexts fit prototypes and four are held out; base models were trained on all 64 input pairs. This is **probe cross-context generalization on a model's training task**, not the native pretrained experiment. A fixed-seed label-shuffle control is reported at the board level. The frozen input hashes and independent receipt are recorded in `COE09_SYNTHETIC_RECEIPT.json`.

## Execution and promotion

Portable unit tests and the synthetic-source replay run locally. The native runner is published on an isolated research branch and executes in GitHub Actions when runners and model download are available. It will emit SHA-pinned native activations and raw causal receipts as workflow artifacts. The native outcome remains **unmeasured** until the complete receipt and its checks pass. Pretrained result and synthetic validation stay separated in all publications.
