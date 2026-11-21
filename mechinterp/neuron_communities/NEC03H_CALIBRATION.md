# NEC-03H | Opened-world 450-to-900 calibration and source-only fallback

**Research stage:** calibration on already examined NEC-03G development roots. Date: 2026-11-21. This is a new protocol for a separate, more permissive development experiment; it does not revise the earlier eight-root NEC-03G ITT mean of zero or its pinned EvalMath preflight.

## Why calibrate

The first fresh-function NEC-03G study trained 16 independent 32-D/12-visit PEF-style checkpoints across eight distinct four-input Boolean worlds. Eight root audits succeeded, while the frozen admission criteria accepted 0/8. Four roots had a teacher with less than 90% clean heldout accuracy after 450 updates. Five had at least one arm with no qualifying positive-phi multi-visit native-neuron community. One more failed the all-arms physical-signal gate. These gates overlap and the original results are preserved in [NEC-03G results](NEC03G_RESULTS.md).

## Frozen 900-step competence comparison

Reuse the exact original NEC-03G world generator and its SHA256 registry commitment. **Only dev-000 to dev-007** are admitted for this calibration. For each, train from scratch with the original native initialization seed and independent composed/additive teacher seeds; optimizer AdamW at LR=0.003, zero weight decay, batch=128, gradient norm clipping 1.0. Continue the deterministic original data-index stream through **900** updates. This deliberately recreates the first 450 steps rather than continuing from a checkpoint whose Adam optimizer moments were not preserved.

The original 450-step native clean-score receipt is recovered by immutable GitHub Actions artifact from [run 44870543545](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44870543545). The new 900-step frozen native checkpoints are given distinct filenames and SHA256 values. Compare native clean scores on each world's *previously opened* 48 development context quartets (768 correlated Boolean states), and record the 90% competence pass/fail for both teachers. Keep all 16 models and failed learners. Do not optimize the per-world training length.

The original 48 within-world **reserved** context groups remain untouched. No dev-008..015 worlds are trained by this calibration workflow. All 64 confirmation-world specs remain committed by hash; no confirmation model is trained or scored.

## Source-only site fallback

Before opening that world's existing development-evaluation groups, extract top-two memory-token GELU activation×gradient native occurrences at every recurrent visit across **80 complete parameter-training context quartets**. Build the original seeded Louvain positive-phi graph and remove ≥90%-frequency hubs.

The original first qualifying multi-visit graph community is retained when one exists, with the source graph selection status logged. When none qualifies, select exactly two non-hub GELU channels at the **third and fourth earliest focal-input positions**, ranked by frequency of appearance among source-only top-two attributions, with frequency at least two of eighty contexts. The scheduled visits depend on the frozen world spec, not the evaluation outcomes. If even this fallback lacks four eligible channels, record **no source site** and keep the root in the denominator. Source attribution rows, graph frequencies, hub exclusions and fallback decisions are stored before measuring clean development accuracy.

Every fallback candidate has four visit:channel occurrences across exactly two recurrent visits and fits the existing TopK16 source-policy capacity of at most eight selected native units at a given visit. The additive teacher receives its **own** source-only site selection and its own native control. Shared sites are not copied from the composed teacher.

## What the calibration can establish

The full public CPU runner tests source selection and teacher source ownership, executes new 900-step native weights on all eight already-opened roots, independently checks their checkpoint hashes, and compares clean accuracy and graph/fallback status with the original immutable 450-step receipts.

The measured outcomes decide whether the *fixed 900-step* schedule merits a separate NEC-03H-v2 development preregistration. Those later fresh development roots must be dev-008..015 with distinct Boolean compositions and focal positions, and the v2 source/causal recipe must be SHA256-frozen **before training them**.

No new SAE fits or causal effect inference are required in this opened-world calibration; its purpose is native training competence and native-site admission feasibility. The original NEC-03G negative ITT result remains unchanged.

**EvalMath FSI-01:** This calibration cannot support a mechanistic significance claim. The 95% upper-tolerance null-floor requirement of at least 59 independent development roots remains unmet, and confirmation stays prohibited until a new independently powered preregistration and explicit release.
