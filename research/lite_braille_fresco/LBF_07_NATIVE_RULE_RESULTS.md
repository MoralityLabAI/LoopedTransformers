# LBF-07: Native Lean typed-rule association causal controls

**Date:** 2026-11-23  
**Disposition:** `NEGATIVE_DEVELOPMENT_SCREEN`; model competence remains `BLOCKED_INCOMPETENT_CHECKPOINT`.

**Preregistration:** [LBF-07 rule association](LBF_07_CAUSAL_RULE_REGISTRATION.md) committed before any LBF-07 model training or native scoring.

**Successful experiment:** [GitHub Actions run 45187820874](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45187820874).

**Retained machine evidence:** [lbf07-native-rule-association-evidence](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/45187820874/artifacts/14824308070) (source/certifier manifests, all trained checkpoint files, native proof outcomes, independently replayed episodes, typed-rule exposure audit, root-level scorecard, SVG fresco). ZIP SHA-256: `09a23b8c43b314496c56a1a727d7e7793c3c75d5d42eca567499f7a6c9d58643`. The artifact is retained through 2026-12-07.

## Exact question

Following the LBF-06 null result for readable Lean context, we tested whether the **correct associations between candidate tactic names and their native typed premise/head rules** help PPT-B select and complete Lean proofs.

All three conditions retained the same native Lean goal, public candidate menu, 384-byte ByteEncoder limit, candidate-first source-rule inclusion policy, inert/blank readable-context suffix, Unicode-character histogram, exact visible input byte count, architecture and optimizer. Rule emission occurred *before* any permutation so the same public rules fit into each input.

- `hybrid_rule_intact`: unmodified name:premises>head bindings.
- `hybrid_rule_rebind`: source-hashed cyclic relocation of rule payloads under fixed names, preferentially among candidate rules; it deliberately makes some model-facing facts incorrect.
- `hybrid_rule_order`: cyclic relocation of complete, truthful name+payload entries, preserving each association. This controls where the entries occur in the input byte sequence.

Both permutations are deterministic functions of public predecision packets and candidate menus, with no verifier labels. The source audit enforces exact bytes, Unicode-character histograms and baseline equivalence to the prior blank-PP hybrid representation.

## Native evidence and CPU budget

The G03 suite was newly frozen under seed `20261128`. Real Lean 4.19.0 and the standalone Lean certifier successfully compiled **48 train-only theorem roots**, yielding **251 native shortest-continuation training frames**. **44/48** reference proofs fit the eight-tactic budget. The comparison evaluated **32 disjoint calibration roots**, using the same eight-tactic and recurrent-depth-two allowances in every arm; the native reference ceiling was **25/32**. External source-family ancestry against prior generated suites remains unverified.

Each of two seeds (107, 109) trained each input variant on the same GitHub-hosted CPU worker, under deterministic PyTorch algorithms with one thread, using a common 57,475-parameter PPT-B, width 32, four attention heads, 384-byte public packet, two recurrent training passes, 720 FP32 optimizer updates, learning rate 0.001 and the same root-sampling stream. A separate duplicate seed-107 intact fit reproduced named parameter tensors and optimizer trace **exactly** on that worker.

The native runner independently compiled successful scripts and replayed all 96 action episodes per checkpoint without the model. All six independent-replay and source/checkpoint custody checks passed. Ordinary-test and OOD outcome sets remained unopened.

## The perturbation was active

The predecision-only source audit recorded:

| Audit | Observed |
|---|---:|
| Independent train roots | 48 |
| Training frames | 251 |
| Frames with changed candidate binding | 240 / 251 |
| Frame-weighted candidate-binding changes | 95.62% |
| **Root-weighted candidate-binding changes** | **96.93%** |
| Frames with rule-position changes | 251 / 251 |
| Same exact visible byte count and character histogram | Pass |
| Identical inert readable-context suffix | Pass |
| Corrected LBF-07 registered 50% intervention-exposure gate | Pass |
| Model-facing verdict labels from the audit | 0 |

The first CI execution built and certified all native training roots but stopped at an unused command-line option on the audit script. The corrected workflow invocation reran the same preregistered seed and controls and completed. The source encoding, model hyperparameters and comparison thresholds were unchanged.

## Kernel-certified native outcomes

| Seed | Intact rules | Rebound associations | Order-only placebo | Public auto8 | Native budget reachable |
| --- | ---: | ---: | ---: | ---: | ---: |
| 107 | **4/32** | 5/32 | 5/32 | 5/32 | 25/32 |
| 109 | **2/32** | 4/32 | 2/32 | 5/32 | 25/32 |

Native tactic checks, trained AB order:

| Seed | Intact | Rebound | Order placebo |
|---|---:|---:|---:|
| 107 | 128 | 158 | 162 |
| 109 | 121 | 136 | 130 |

Every reported completion is a native-kernel success backed by independent standalone compilation and original episode replay. Tactic-check consumption differed by model policy and is recorded rather than implicitly treated as equal.

### Preregistered root-paired contrasts

Each row pairs **32 independent theorem roots**, averaging seed effects inside each root. The 95% intervals are descriptive root-cluster bootstraps, 2,000 draws.

| Difference | Mean | Descriptive 95% interval | Wins/losses/ties |
|---|---:|---:|---:|
| Intact − rebound | **−4.6875 pp** | [−12.50, +3.125] pp | 1 / 3 / 28 |
| Intact − order-only | **−1.5625 pp** | [−14.0625, +12.50] pp | 4 / 6 / 22 |
| Association-specific (order-only − rebound) | **−3.125 pp** | [−15.625, +7.8125] pp | 4 / 4 / 24 |

The fixed development requirement was both a **+5-point intact-minus-rebound gain** and a **+5-point extra association-specific gain beyond the positional control**. The registered screen **missed** on both. Interventions materially changed the inputs, so the negative finding cannot be attributed to a failed or inert rebinding operation in the training source.

The strongest arm closed only **5/32 = 15.625%**; the competence threshold remains 40%. An exact native candidate-uniform baseline and an FSI-01 powered confirmation remain absent, so the result is development-only and `scientific_eligible=false`.

## Luminous recurrence probe

Using the frozen seed-109 intact checkpoint, the read-only LBF adapter rendered **four neural episodes** and ran **84** previous-pass message zero/roll probes. First-pass zero-memory invariance passed. Maximum absolute candidate-logit displacement was **0.2486540675163269**; **zero** probes changed the selected top tactic. Rendering consumed zero new Lean actions or verdict labels.

The large continuous-logit displacement shows that cross-pass messages can affect this fitted model's numerical scores; it does not establish action-policy or kernel-verified proof gain.

## Scientific interpretation

This result further weakens an architectural story in which the Fresco/Braille notation reveals an already-existing, reliable syntactic proof calculus inside the PPT-B controller. The visual alphabet continues to encode instrumented dynamics correctly, while the policy itself has low proof competence and did not exploit correct typed-rule associations in this matched development comparison.

LBF-05's positive raw-versus-hybrid result and LBF-06's negative readable-context ordering result remain separate evidence. LBF-07 narrows the remaining explanation: fixed input length, rule-name vocabulary, restricted training states or recurrent-policy optimization may dominate these small-capacity Lean outcomes. The present perturbations are *deliberately false model inputs*; their failure to hurt proof closure is not evidence that Lean inference is invariant to logical premise changes.

### Next registered direction: LBF-08 / verified on-policy repair

Prioritize training from **model-visited rejected tactics, accepted nonterminal successor states, and certified full-proof recoveries**, with an action-set target that credits all equally good native continuations. Compare the resulting learner with the existing reference-path imitation and no-repair controls at fixed proof-check and compute budgets. Include explicit candidate-set coverage and STOP efficiency. Preserve source-lineage custody, the full native theorem compiler and model-free replay checks. Use the fresco to inspect which recurrent messages change during actual verified recoveries.

Avoid opening held-out ordinary or OOD tests until proof competence, independent calibration and prospective matched-sham gates are satisfied.

## Source links

- [LBF-07 preregistration](LBF_07_CAUSAL_RULE_REGISTRATION.md)
- [LBF-07 serializer](../lean_cgrt_hrm400/ppt_gym/lbf_hybrid.py)
- [Public-source binding audit](lbf07_audit.py)
- [Native root-paired comparator](lbf07_compare.py)
- [Full CI workflow](../../.github/workflows/lbf07-rule-association.yml)
- [LBF-06 context controls](LBF_06_PRETTY_CAUSAL_RESULTS.md)
