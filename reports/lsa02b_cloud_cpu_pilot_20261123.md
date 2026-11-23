# LSA-02B CPU-native training pilot: first cloud results

**Date:** 2026-11-23. **Status:** completed development-only CPU campaign, architecture gate **BLOCKED_INCOMPETENT_CHECKPOINT**. Source and training code are in [PR #321](https://github.com/MoralityLabAI/LoopedTransformers/pull/321), and the source SHA/artifacts were recorded in [Actions run 45167489504](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45167489504). The uploaded binary checkpoints and transcript bundles have fourteen-day artifact retention. Exact corpus and script hashes remain in their manifests.

## Training source

The real Lean 4.19 G03 server extracted **24 independent training roots**, yielding **124 shortest-continuation frames**. All 24 reference scripts compiled independently without additional axioms. **21/24** training-root reference proofs fit within eight tactics. Calibration and sealed splits were disjoint from training and neither supplied optimizer updates.

A frame's binary action label means that the action begins a **minimum-size continuation** under the Lean-reported AND/OR proof graph, rather than asserting one-step terminal closure. The teacher source was fully hashed before training. No model-generated proof labels were used as authority.

## Paired CPU fits

Each Plan B checkpoint used **57,475 unique parameters**, two recurrent passes, FP32 AdamW with 240 updates, and 32-wide hidden states. Same-seed AB and BA fits had identical initial weight hashes and root-sampling schedules:

| Seed | Order | Initial weight hash prefix | First sampled loss | Final sampled loss | Final-32 mean loss |
| --- | --- | --- | ---: | ---: | ---: |
| 101 | AB | `f1c977da` | 0.5837 | 0.1659 | 0.9619 |
| 101 | BA | `f1c977da` | 0.5460 | 0.1250 | 1.0835 |
| 103 | AB | `f326ed1b` | 0.9446 | 0.0271 | 0.9643 |
| 103 | BA | `f326ed1b` | 0.9771 | 0.0263 | 0.8989 |

Individual step losses are noisy and the two orders already differ at the first update. They provide evidence of finite optimization on Lean-derived frames, not of an architecture effect.

## Real Lean calibration (the same eight theorem roots per checkpoint)

| Trained checkpoint | AB full proofs | BA full proofs | Public-order `auto8` | Reference within-budget |
| --- | ---: | ---: | ---: | ---: |
| 101 / AB | 1/8 | 1/8 | 1/8 | 6/8 |
| 101 / BA | 1/8 | 1/8 | 1/8 | 6/8 |
| 103 / AB | 2/8 | 2/8 | 1/8 | 6/8 |
| 103 / BA | 2/8 | 2/8 | 1/8 | 6/8 |

All native closures passed the Lean kernel and independent standalone compilation. Replay used the saved model-free tactic transcripts and the original theorem suite source. Each checkpoint's source-verified training corpus and its full native root census were authenticated before the calibration set was scored.

The **frozen AB/BA intervention produced zero first-action changes** across all four cells. Across-seed and across-checkpoint rows repeat the same eight theorem roots; their 32 entries are correlated observations, not 32 independent roots.

At two seeds and eight roots, the scoped conclusion is that we have working Lean-supervised Plan B training with limited native proof competence. Both orderings made identical decisions after training on the current examples, which leaves the commutator-to-task-performance hypothesis open.

## Verification and boundaries

The following workflows completed green on PR #321:

- LSA-02B CPU-native checkpoint pilot, including 24 independently compiled training proofs, four checkpoint artifacts and four complete native calibration/replay jobs.
- PPT Gym CPU regression.
- PPT G03 native Lean verification.
- LSA-02B native paired Lean order smoke.

The pilot was expressly kept out of the previously stated competence gate, which requires at least 32 independent eligible calibration roots, a trained-order closure rate of at least 0.40, and +0.10 over a registered random-menu baseline. The present best trained-order rate is **0.25** on eight roots, below the gate. The result is labeled `BLOCKED_INCOMPETENT_CHECKPOINT`. It does not establish architecture equivalence, advantage, mechanistic mediation or transfer to genuine Mathlib proof tasks.

## Next development registration

Move to a *distinct* CPU-only developmental cohort with more unique training roots, a separate competence calibration split, and a preregistered input-feature audit. More optimizer steps alone may simply memorize the same 124 frames. A stronger next comparison should include the fraction of public states offering multiple candidate tactics and whether their required optimal actions differ across matched minimal pairs. If these are rare, the experiment primarily measures candidate-menu determinism.

Before any later gym, compare a compact public-state serialization with the existing raw JSON under matched token/byte budgets, verify that the model can read the relevant premise graph, and require an actual improvement in standalone Lean proof completion. Continue preserving parameter-sharing identities and exact CPU cost receipts.
