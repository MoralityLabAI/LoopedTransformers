# OLR-METTA-03: paid evidence decision with a recurrent obligation reader

**Registration:** 2026-11-23, source-only and before implementation scoring.  
**Status:** new exploratory native Lean/Hyperon CPU pilot, not a confirmation test.

## Question and source boundary

Can a tied recurrent policy learn whether to spend **two additional native Lean checks** diagnosing a speculative guard, based on MeTTa-curated public witness claims, while the actual proof has two additional pending goals? An unfiltered source-claim control, one-pass cross-reader, direct all-input MLP, always-probe, fixed-r0, and an exact arithmetic **full-information** oracle are required. The latter is an upper bound because the model observation deliberately masks the public guard numerals.

Each theorem uses the pinned OLR-01D Lean 4.19 grammar: two alternatives to prove R, both requiring an arithmetic equality and the pending P/Q obligations. The verifier receives the complete theorem and executes actual tactics. The model receives **only a deliberately restricted public observation**: at most six noisy, synthetic evidence claims, their public source and revision statuses, MeTTa admission bits, and the six-check call allowance. Neither the truth bit, the arithmetic operands nor the reference script appear in those model features. Claims model fallible external testimony and have no Lean proof authority.

## Frozen experiment contract

- Generator seed **20261129**, disjoint integer operand bands: train 17–30, development 17–30 with disjoint (a,b,offset) triplets, OOD 31–44. Source roots are paired by underlying (a,b,offset); both placements of the valid equality are kept together.
- Initial pilot **24 training pairs, 8 development pairs, 8 OOD pairs**; each pair has two proof worlds. Fixed source evidence modes: high-signal, weak, conflicting and stale/alias. One public evidence claim carries a vote for r0 or r1 and declared strength. Duplicate source IDs and stale revisions must not be counted as independent corroboration. MeTTa/Hyperon 0.2.10 selects *eligible, current and unique-source* claims via a native match query. A Python reference independently checks its returned set.
- Actions **skip-r0**, **skip-r1**, or **paid-probe**. A skip chooses an opening and spends up to 4 native tactic calls: apply rule, try rfl, then apply h0 and h1 if the guard passes. A failed rfl ends that policy attempt. Probe checks r0 on a *speculative saved branch* (apply r0; rfl) at a cost of two calls, restores the original Lean state even on success, and then executes the 4-call final proof of the informed branch. All policies have the identical six-call cap. Every successful script is independently compiled and axiom-audited.
- Per-root reward: **1 for a kernel-certified proof less 0.07 per attempted tactic call**. Probe therefore returns 0.58 when successful; a correct unprobed choice returns 0.72. A rejected unprobed branch returns -0.14. Report both proof completions and paid calls, with root-paired confidence descriptors and all seeds.
- Fit genuine source action-value labels from **native Lean executions of all three actions on each training root**. No development outcome or source generator truth is a model feature. Recurrent policy: one small tied self-attention block applied 3 times to the same public witness tokens; one-pass sharing control at depth 1 and full-information flat MLP comparator. Three seeds **17, 29, 43**, identical source root batches and optimizer step budgets (80 updates, batch size 24), no GPU.
- Model/MeTTa ablation: hold model weights fixed and zero the MeTTa admission flags at evaluation. A separate unfiltered all-claims heuristic measures reliance on stale/duplicate evidence. Compare an evidence-majority heuristic, a posterior-informed source-only frequency control, always-probe, fixed-r0 and full-knowledge exact arithmetic oracle.
- Freeze hashes for raw root statements, source provenance, MeTTa program text, native Lean and Hyperon versions, train/dev/OOD outputs and checkpoint states. Replay native transitions from fresh Lean states; compiled source must use each successful policy's exact chosen rule. Native proof failures, costs and predecision observations must be present in the artifact.

## Attribution limits and decision

This is **not** a partially hidden theorem to Lean, only a restricted policy interface. The source testimony is synthetic; Hyperon verifies lineage/admission rules and never certifies factual truth. The exact arithmetic oracle remains the algorithmic ceiling. A learned return improvement at fixed six-call cap would be evidence of inference-time verifier-budget allocation on the restricted interface, not a new full Lean reasoning ability. A recurrent mechanism claim requires gains over the best same-information direct/one-pass control, calibrated equal-compute comparisons, causal message interventions and a separate prospective EvalMath FSI-01 sham/power check.

Keep the initial cohort open as development only and PR #318 in draft.
