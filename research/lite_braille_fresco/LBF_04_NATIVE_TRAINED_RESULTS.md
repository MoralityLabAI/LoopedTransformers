# LBF-04: Native-trained PPT-B luminous fresco and competence gate

**Date:** 2026-11-23  
**Disposition:** `BLOCKED_INCOMPETENT_CHECKPOINT`  
**Evidence class:** real Lean 4.19 proof-source training, independently compiled and replayed native evaluation, development-only eight-root calibration.  
**Workflow:** [CI run 45180208697](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45180208697), **success**.  
**Artifact:** [source-trained model checkpoint, episode receipts, message probes, encoded fresco and SVG](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45180208697/artifacts/14821221260).

## Study

LBF-04 replaces the two-step synthetic checkpoint from LBF-03 with a genuinely source-verified PPT-B model. It uses the existing `ppt_gym.native_train collect` and `fit` code and the `native_order` real-Lean evaluator, preserving the source/control separation registered in [LSA-02B](../../docs/lsa02b_cpu_training_pilot.md).

The training corpus contained **24** independent train-only Lean roots and **124** public shortest-continuation training frames, each derived from a native Lean 4.19 action search and independently compiled shortest proof. All reference scripts passed standalone verification. Seed **103**, trained order **AB**, was selected for this development follow-up based on the previously exposed LSA-02B calibration campaign. Therefore this repetition is **not independent confirmation** of the earlier eight-root close rate. The original exposure and lineage limitations remain explicit.

The checkpoint used width 32, byte budget 384, two recurrent training passes, 240 CPU optimizer updates, learning rate 0.001, and two CPU threads. The model's parameter weights remained frozen during native evaluation and message probes.

## Observed results

| Metric | LBF-04 |
| --- | ---: |
| Native train lineages | 24 |
| Native train frames | 124 |
| Completed optimizer steps | 240 |
| Calibration roots (same frozen set as prior development) | 8 |
| AB standalone-certified complete proofs | 2 / 8 |
| BA standalone-certified complete proofs | 2 / 8 |
| Auto8 standalone-certified complete proofs | 1 / 8 |
| Reference proofs independently compiled | 8 / 8 |
| Reference proofs within common tactic budget | 6 / 8 |
| AB model calls / Lean tactic checks | 34 / 34 |
| BA model calls / Lean tactic checks | 34 / 34 |
| Exact native episode replay | Passed, all 24 arm episodes |
| LBF neural episodes rendered | 4 |
| LBF normal/zero/roll message-probe rows | 84 |
| Peak absolute candidate-logit displacement | **0.02437114715576172** |
| Changes in top-ranked tactic across message probes | **0** |
| Extra Lean calls from LBF adapter | 0 |

Checkpoint SHA-256: `05dda9f98e14cf7f1b51a22174ec0911b4d1bc1a2fbcef3cd0b8b05730c29709`. Workflow artifact ZIP SHA-256: `c0a92166564821105c21d2bb924e2e523d7db145b95f30e52ab7c304ead91303`.

The implementation checks predecision-only observation construction, independently replayed native input provenance, and first-pass message equivalence (the memory bank is zero at the beginning of recurrence). The observed message effect is an **output-level causal intervention on the frozen neural computation**. All four neural fresco traces and the per-pass candidate-logit and coda-state probes are stored as reproducible artifacts.

## Competence decision

The registered LSA-02B competence gate requires at least **32** independent eligible calibration lineages, trained-order close rate >= **0.40**, and >= **0.10** improvement over an exact candidate-uniform baseline. LBF-04 covered eight roots, achieved 0.25, and did not measure the candidate-uniform baseline. Its official disposition is **`BLOCKED_INCOMPETENT_CHECKPOINT`**; the locked `ordinary_test` and `ood_family` cohorts remain unopened.

The native proof reference reached 6/8 under the common tactic budget, while this checkpoint reached 2/8. There is still reachable headroom for model policy improvements. The absence of selected-action flips under message replacement shows that these interventions have not yet changed discrete decisions on the observed development frames; measuring a nonzero continuous logit displacement alone cannot establish a reasoning advantage.

## Proposed next development gate: LBF-05 typed public-state repair

A likely limitation is the raw byte-encoded public Lean state, which can truncate important candidate-prerequisite structure at the 384-byte bound. The existing [LSA-02C typed-state encoder](../../research/lean_cgrt_hrm400/ppt_gym/typed_state.py) was built specifically to preserve goal-relevant dependency information. It provides a bounded `typed` representation with its own source identity. Test a new paired **raw vs typed** 240-step fit, matching seed, optimizer, source teacher, parameter count, recurrence depth and training budget. Change only the declared public-input view. Reuse development calibration, keep both fresh checkpoints and full trace receipts, and never open sealed tests until an independent competence gate passes.

If typed repair improves verified root completion, analyze which candidate-relevant slots and message reads change using direct gated-middle instrumentation and time-shuffled/norm-matched shams. A sufficiently competent checkpoint and at least 32 newly preregistered calibration roots are prerequisites for any confirmatory LBF reading claim.
