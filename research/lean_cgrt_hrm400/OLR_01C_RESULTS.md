# OLR-01C: first native arithmetic-guard learning results

**Follow-up:** The [one-pass/matched-sham and deterministic-checkpoint audit](OLR_01C_COMPUTE_ABLATION.md) supersedes this first exploratory scorecard for scientific interpretation. The first two hosted runs exhibited checkpoint drift across workers; the later controlled runs are bit-identical for all common model tensors. This page retains the initial run exactly as observed.

**Date:** 2026-11-23  
**Status:** Successful CI and native verifier integration; **exploratory neural pilot, no confirmed mechanism gain**.  
**Protocol:** [OLR_01C_GUARD_LEARNING.md](OLR_01C_GUARD_LEARNING.md)  
**GitHub Actions:** [PPT G03 native run 45167635953](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45167635953) (success), [CPU regression run 45167635782](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45167635782) (success).  
**Exact source and model receipts:** [olr-01c-native-guard-pilot.zip, artifact 14815601193](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45167635953/artifacts/14815601193). Uploaded artifact archive SHA-256: `aec24abdb47bee2477874244a30b98f339841e711afc8dfa24bc4b299fc07bbe`.

## Executed protocol

- Registered native source seed 20261124: **32 train pairs (64 worlds), 12 dev pairs (24 worlds), eight OOD pairs (16 worlds)**. Within each pair, the winning guard swaps from r0 to r1, preserving both candidate costs and their public grammar.
- The training sampler uses three model seeds **17, 29, 43**, 80 AdamW updates each, depth 3, width 32 and learning rate 0.003. The same source rows and minibatch indices are used across independently fitted arms.
- Each scorer has **17,985 stored parameters**. Cross and isolated recurrence train 17,985; frozen recurrent body trains 3,425. The direct model bypasses the cell and therefore uses less active computation. Strict matched-FLOP attribution is a later gate.
- The predecision model receives three separately normalized public operands per candidate. It receives neither Python's computed equality result nor the source winner. All training labels come from native Lean's `apply rN; rfl` and independent certificate checks.
- The exact host-arithmetic policy closes all 24 dev and 16 OOD worlds. The frozen r0-only policy closes 12 and eight respectively; both-world pair score is zero by construction. The proof menu is complete and the two-call budget is feasible.
- All **104 source and evaluation witness scripts** were independently compiled and axiom-audited before learning. At evaluation, the scorer selects the first tactic; Lean executes it and its guard step from a fresh proof root, matching the source's native verdict. The runner records 560 policy decisions and 80 native reruns of distinct selected root/action combinations. The **40 distinct evaluable winning scripts** compiled independently, axiom-free.

## Outcome: completed pairs (both worlds certified)

The same theorem pairs are re-used across model seeds. These are repeated seeded evaluations, **not independent samples**. Pair successes are a stricter measure of whether the policy can reverse its root choice when the guard positions change.

| Arm | Dev seed 17 / 12 | Dev seed 29 / 12 | Dev seed 43 / 12 | Dev sum / 36 | OOD seed 17 / 8 | OOD seed 29 / 8 | OOD seed 43 / 8 | OOD sum / 24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cross-message tied loop | 5 | 3 | 9 | **17** | 3 | 3 | 8 | **14** |
| Isolated tied loop | 4 | 2 | 11 | **17** | 0 | 3 | 4 | **7** |
| Direct reader | 4 | 3 | 10 | **17** | 0 | 3 | 4 | **7** |
| Frozen recurrent body | 5 | 5 | 10 | **20** | 2 | 3 | 3 | **8** |

Development gives **no complete-pair advantage** for learned cross-message recurrence over the simpler direct or isolated scorers. Frozen-body calibration is strongest there. The OOD panel shows a +7/24 descriptive complete-pair advantage for cross over both isolated and direct, and +6/24 over the frozen recurrent body. However, the seed-specific OOD cross-versus-isolated differences are +3, zero and +4 pairs from the **same eight roots**. The seed-43 score saturates at 8/8 while seed-17 and seed-29 close only 3/8. This is too unstable and small for a mechanism claim, especially because OOD was already an expected diagnostic split in the original CI protocol.

## Individual native Lean completions

| Arm | Development proofs by seed (out of 24 each) | OOD proofs by seed (out of 16 each) |
| --- | --- | --- |
| Cross-message tied loop | 10, 6, 18 | 7, 6, 16 |
| Isolated tied loop | 9, 4, 22 | 4, 6, 8 |
| Direct reader | 16, 6, 20 | 4, 6, 8 |
| Frozen recurrent body | 10, 10, 20 | 6, 6, 6 |
| Exact arithmetic oracle | 24 on every evaluation | 16 on every evaluation |
| Fixed r0 | 12 on every evaluation | 8 on every evaluation |

Every policy spent exactly two native tactic calls per evaluated world. The **first-action selection** is therefore the operative difference; there is no learned tool-query or stopping benefit in this experiment.

## Training and optimization diagnostic

The cross-message model's source accuracy varies across seeds: 39/64, 42/64 and 54/64 training worlds. Its final minibatch losses were 0.8686, 0.5864 and 0.5226. Seed 17 ended with a loss above its initial 0.6936, suggesting unstable optimization or atypical final-batch composition at this short schedule. Direct and isolated source fits vary comparably, with seed 43 much stronger than the other two. The fitted models share a frozen source bank and have no separate optimized stopping head.

Because initial root guards in this experiment have a single equality premise, this pilot tests **semantic guard scoring in two recurrent slots**, rather than transport of general AND-OR hypergraph obligations through many Lean goal transitions. The full OLR hypothesis remains untested.

## Verification receipts

- Native Lean 4.19.0; `lean` binary SHA-256: `92c3d35b5bfaa5e0fea413a775d504cf46cd95e1345df61c2274f76779e7e023`.
- Source manifest SHA-256: `ca3d4f2be451d6c49656f8e063f5130e75e35673a2432ebdba8a8d88eb6e6019`.
- Source bank native scripts: 104 compiled, 104 axiom-clean.
- Evaluation independent scripts: 40 compiled, 40 axiom-clean.
- Model state `models.pt` SHA-256: `23a702fa7a634cff50b407588aa78d4022bea700587fdc60cc15cce33a56b1e9`.
- Evaluation `decisions.jsonl` SHA-256: `ce16aca83de77822c3a71fded96b7c85e1f2a1f42ea0064cd4af9657b8dd7c42`.
- Runner-reported source+evaluation execution time for the trainable stage: approximately four seconds of Python/Lean work on the GitHub worker, excluding environment installation and native bank generation.

## Follow-up decision

**Keep OLR at research status.** The out-of-range paired improvement is useful as a *diagnostic hunch*. The source sample is small; a fixed exact host arithmetic procedure dominates every neural arm. Scaling this two-slot numeric equality learner would mainly spend parameters rediscovering arithmetic with strong prior knowledge already available.

A more decisive OLR experiment should provide genuine multi-step obligations and evidence-driven branching, with a meaningful gap between a structural graph solver and a direct semantic reader. Candidate families include arithmetic guards whose premise is unfolded after a tactic, dependent witness reuse, and budgets where choosing or restoring a branch early matters. Pre-register the source-shift, root lineage, stronger matched baselines and causal sham before opening new evidence. Only promote an OLR checkpoint if more roots complete at matched Lean checks and matched inference compute, with a direction-specific effect beyond the sham envelope.
