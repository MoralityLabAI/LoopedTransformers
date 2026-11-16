# AF-24B: Frozen recurrent proof policies fail to transfer to nonisomorphic Horn grammars

**Registered scientific verdict: NEURAL_TRANSFER_FAIL (0/8 qualified seeds).**  
**Hosted engineering and independently audited proof replay: PASS (11/11 CI jobs).**  
**Result date:** November 16, 2026.

The source-registered AF-24B experiment tested the original eight AF-23 TypedLoop checkpoints, without further training, on 384 new positive-Horn rule graphs spanning shortest proofs of five and six legal firings. A programmatically fixed host verified every rule action. The preregistered two-depth neural-versus-symbolic transfer gate failed in all eight eligible seeds.

**A design limitation affects the strength of that gate:** the new graph-ancestor greedy and bounded BFS controls attain **100% proof completion on both new positive cohorts**. Consequently the requirement that a neural policy exceed the *best* such control by ten percentage points was impossible to satisfy. The paired neural accuracy and premature-STOP failures remain independently observed results, but the primary relative gate cannot distinguish a near-perfect neural policy from a weak one on this roster.

## 1. Immutable source and receipts

- **Registered protocol:** [AF-24B PROTOCOL.json](../../experiments/activation_funnel_af24b/PROTOCOL.json), committed at [`7e845ad`](https://github.com/MoralityLabAI/LoopedTransformers/commit/7e845ad5977eb36a629da496e57908c620507d5f) before writing or scoring the new generator. Protocol SHA-256 `fb7d2d58dc3266745a7eb133f1c51e235331ff5c507301d2bab82921e43c27e6`.
- **Source-only structural preflight:** [CI #43822626726](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43822626726) passed before neural scoring. The frozen manifest SHA-256 is `105e82553efaa1859cb01ac1f59e31d75107a7e5d557353cedffab449868c1fd`.
- **Official source-locked model evaluation:** [CI #43823769006](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43823769006), scientific source commit `e277916b15a4331eb2106232dba7578dfd3d3f7e`, conclusion SUCCESS. Source lock, frozen-checkpoint smoke, all eight independent seed jobs, and independent aggregation passed.
- **Original AF-23 model source:** [AF-23 Track B CI #43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916), commit `1312d138b685d1054d527e56e16e1ea757a90b08`. All eight original SHA-256-pinned checkpoint artifact ZIPs, original training receipts, Torch/NumPy weight arrays and AF-23 source hashes were validated before running the new test.
- **Official aggregate artifact:** `af24b-frozen-transfer-aggregate`, artifact ID **14205901442**, artifact ZIP SHA-256 `637a80185cca9e0c68da5aba712b90963e1b8c89475ca9499d41bba146278ad0`, containing `af24b_results.json` SHA-256 `b3123412d2d13e73acae8433f76b362805190ee5c9a6c6636696423a513a4df1`.
- **Independent mathematics:** [shared-subproof additive overcount obstruction](AF24B_SHARED_SUBPROOF_OBSTRUCTION.md), including exact cost formulas for the adversarial generated family.

This document was committed after the frozen evaluation. No result-bearing experiment code, data, thresholds, model checkpoints, or parent scientific verdicts were changed after the source lock.

## 2. Design and hypothesis

The original AF-23 model is a **6,036-parameter TypedLoop** with four recurrent calls and public graph-ancestor readout. The exact checkpoint from each of the eight fixed source seeds (17,29,43,61,73,89,103,131) was reused unchanged across three inference algorithms:

| Neural arm | Decision |
|---|---|
| Direct | Highest-scoring legal physical rule or STOP from the original raw logits |
| VI-1 | One-step learned successor value plus the root STOP cost |
| VI-2 | Two-step min-plus Bellman successor backup using the same trained weights |

Nonlearned comparator arms were first-legal fixed, goal-ancestor greedy, additive-cost Horn closure, BFS with 16 or 64 public host-state expansions per decision, and an exact shortest-proof oracle as an evaluation upper bound. The graph-aware algorithms were reused from AF-24 with their original scoring rules.

The frozen new grammar has two goal-concluding rules, rather than AF-23's original single conjunctive goal rule. Three separate families were tested, each on new physical layouts. All use at most 15 atoms and 14 rules, with role markings compatible with the original network. The heldout depth-5 and depth-6 cohorts each contain **192 cases**, comprising **80 provable** and **112 unprovable**. Source-only independent bitset BFS confirmed every positive proof length and negative reachability.

The families were:

1. **Shared-overcount:** a productive chain is reused to reach two goal subgoals; additive-cost closure overcounts the prefix and chooses a longer alternative goal route. Registered structural preflight closure: 0/24 positive cases at each depth.
2. **Competing OR-goal:** one subgoal route is shorter, while a distinct alternative also directly concludes the goal. Structural preflight closure: 24/24 at each depth.
3. **Crossed conjunction:** goal-subproof dependencies cross through shared intermediates, with a separate longer direct-goal route. Structural preflight closure: 32/32 at each depth.

The model is exposed to both new rule topology and input-statistics shift. Positive starts have only two or three given facts, outside AF-23's original four-fact positive-root training template. Performance changes cannot isolate abstract graph-generalization ability from this covariate shift.

### Original registered gate

VI-2 must, **in the same seed at each heldout depth**, complete at least eight more of the 80 provable cases than the best of the four registered graph-aware symbolic comparators, solve at least 40/80 absolute cases, achieve at least 90% immediate correct STOP on the 112 unprovable cases, and have zero false accepted Horn proofs. At least six seeds must pass both depths. Missing or invalid original evidence would yield INCONCLUSIVE; a complete qualifying study failing the bar yields NEURAL_TRANSFER_FAIL.

Every original checkpoint qualified, and the complete study yields **0/8** two-depth pass.

## 3. Official complete pooled proof success

Each depth uses **640 provable scored episodes** (the same 80 graphs evaluated against eight independent original neural checkpoints). Deterministic symbolic policies repeat the same physical cases; the repeated count is for paired comparison and is not 640 independent graph samples.

| Controller | Depth 5 solved /640 | Depth 6 solved /640 |
|---|---:|---:|
| Direct, frozen | 112 (17.50%) | 24 (3.75%) |
| VI-1, frozen | 80 (12.50%) | 112 (17.50%) |
| **VI-2, frozen** | **121 (18.91%)** | **148 (23.13%)** |
| First legal, fixed | 160 (25.00%) | 240 (37.50%) |
| Additive-cost closure | 448 (70.00%) | 448 (70.00%) |
| **Goal-ancestor greedy** | **640 (100%)** | **640 (100%)** |
| BFS cap 16 | **640 (100%)** | **640 (100%)** |
| BFS cap 64 | **640 (100%)** | **640 (100%)** |
| Exact shortest oracle | **640 (100%)** | **640 (100%)** |

The originally preregistered neural transfer criterion is **mathematically unreachable on these cases** because the strongest named symbolic comparator solves every provable goal. Even a hypothetical perfect neural model would have a success difference of zero, below the required +10 percentage points. The observed 18.91% and 23.13% VI-2 completions independently indicate poor transfer of the frozen neural policy to these OOD graphs.

Both the deterministic ancestor heuristic and the capped BFS recover **all 24 shared-overcount positive cases per depth**, while additive closure solves none. This narrows the structural weakness of AF-24's previous controller to a specific non-shared-cost heuristic, rather than an inherent difficulty of the new benchmark for cheap symbolic methods.

### Pooled performance by new family

Counts below repeat the same family graph cases across eight neural seeds.

| Family | Provable /depth | Depth 5 VI-2 | Depth 6 VI-2 | Closure both depths | Ancestor both depths |
|---|---:|---:|---:|---:|---:|
| Shared-overcount | 192 | 33/192 | 17/192 | 0/192 | 192/192 |
| OR-goal competition | 192 | 24/192 | 35/192 | 192/192 | 192/192 |
| Crossed conjunction | 256 | 64/256 | 96/256 | 256/256 | 256/256 |

The shared-overcount family produces comparatively little neural success at both depths. There is limited partial recovery from the Bellman backup in some seeds, but no cohort-wide advantage against a simple goal-ancestor heuristic.

## 4. All registered neural seeds

Number of provable goals completed by VI-2, out of 80 at each depth:

| Seed | Held depth 5 | Held depth 6 | Registered both-depth gate |
|---:|---:|---:|---|
| 17 | 32 | 43 | FAIL |
| 29 | 0 | 0 | FAIL |
| 43 | 0 | 0 | FAIL |
| 61 | **80** | **73** | FAIL |
| 73 | 9 | 0 | FAIL |
| 89 | 0 | 0 | FAIL |
| 103 | 0 | 0 | FAIL |
| 131 | 0 | 32 | FAIL |

Seed 61 demonstrates that a particular frozen checkpoint can finish almost every new provable case using VI-2. Six of eight seeds, however, solve fewer than half the new cases at both depths, and each seed remains below the best graph-aware comparator.

The direct policy sometimes performs better than VI-2, including seed 17's 56 depth-5 Direct completions against 32 VI-2. This result is retained without post-hoc policy selection.

## 5. STOP behavior and direct failure modes

The dominant neural failure is excessive STOP on provable cases, which had also been central to AF-22.

| Policy | Depth 5: immediate STOP on provable /640 | Depth 6: immediate STOP on provable /640 | Correct initial STOP on unprovable /896 (both depths) |
|---|---:|---:|---:|
| Direct | 504 | 568 | 896 |
| VI-1 | 408 | 432 | 896 |
| **VI-2** | **429** | **408** | **896** |

Thus the frozen VI-2 model immediately stops on **67.03%** of provable depth-5 cases and **63.75%** of provable depth-6 cases. It also correctly stops at the start of **every one of the 896 unprovable episodes** at both depths.

This is consistent with a threshold/readout-calibration failure under graph and initial-fact distribution shift. It does not establish whether the error comes predominantly from goal relevance, the reachability head, learned distance, or the root STOP comparison. Each is a prospectively testable hypothesis.

## 6. Compute measurements and independent audit

The source-pinned hosted run completed **all 11 jobs**: preflight source lock, a sampled original-checkpoint smoke, eight per-seed full-rollout-and-audit jobs, and independent aggregation.

| Verification item | Completed count |
|---|---:|
| Audited proof episodes | **27,648** |
| Reconstructed logged neural planner nodes | **85,821** |
| Original frozen Torch checkpoint forward replays | **96** |
| False accepted Horn proofs | **0** |
| Original model retraining updates | **0** |

The independent auditor decodes the public Horn incidence from the actual **model-visible tensors**, recomputes shortest proof distances with a separate physical-bitset BFS, checks every proof step and STOP decision, and recomputes min-plus backup costs from stored neural values. Torch forward replay uses the original AF-23 pinned CPU framework and numeric tolerance. Source/checkpoint hashes, Torch/NumPy weight parity, all per-seed proof episode digests, and original artifact IDs were independently validated.

Computational work is recorded in compatible *native* counts, not equated across algorithms:

| Source operation total, both depths combined | Count |
|---|---:|
| VI-2 neural boundary-node evaluations | **30,149** |
| Goal-ancestor greedy public rule examinations | **666,544** |
| Additive closure public rule examinations | **550,664** |
| BFS cap 16 public state expansions | **76,872** |

The source study does **not** establish comparable GPU FLOPs, matched neural/symbolic CPU latency, or amortized compute parity. All graph-aware controllers have access to the same public incidence and current facts, while the symbolic policies can explore more host states than a two-step neural planner.

## 7. What remains unresolved

AF-24B gives credible evidence of **frozen neural policy fragility under unseen Horn structure and root-fact distributions**. The weak direct and VI-2 proof completion rates, alongside large provable early-STOP counts, are independently auditable. The new grammar's additive-cost closure obstruction is also a genuine, source-verified mathematical counterexample.

The relative transfer criterion has low discriminating power because goal-ancestor greedy and BFS cap 16 are perfect. A future study should enforce the **full benchmark control floor before neural scoring**: require meaningful failure rates for *all* registered cheap graph-aware controls, report the oracle/cheap-control gap by family, and freeze those controls' inference budgets before training or evaluation. The original current study's failed transfer criterion remains recorded, and its outcomes must never be retroactively reclassified.

A stronger next test would introduce non-oracular graph motifs with misleading **short reverse-hop goal ancestors**, competitive shared and independent subproofs, and genuinely costly AND prerequisites. It should retain positive and negative goals with matched root-fact counts so that stopping competence is tested independently from input-count shift. New models would need separate prospective train/dev data with fixed checkpoint selection.

**Source-pinned scientific status:** AF-24B NEURAL_TRANSFER_FAIL. Earlier registered verdicts are intact: AF-22 FAIL, AF-23 Track B PASS against fixed/random, and AF-24 SYMBOLIC_CONTROL_MATCH. This report changes documentation only; [PR #126](https://github.com/MoralityLabAI/LoopedTransformers/pull/126) remains **draft and unmerged**.
