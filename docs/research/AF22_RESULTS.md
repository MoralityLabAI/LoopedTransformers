# AF-22: STOP calibration failures in a low-branching Horn-rule gym

**Corrected research report, November 16, 2026. Registered scientific verdict: FAIL. Independent engineering audit: PASS.** This document supersedes the earlier AF-22 narrative at [7034858](https://github.com/MoralityLabAI/LoopedTransformers/commit/7034858ea921960beac48fa6b7788fa39649f37f), which omitted the experiment's fixed and oracle arms and made unsupported positive claims about neural proof search.

**Frozen scientific source:** [`5c16b342323977b3045f34c9be4eb24113dc691e`](https://github.com/MoralityLabAI/LoopedTransformers/commit/5c16b342323977b3045f34c9be4eb24113dc691e). [Official source-pinned CI run #43788715688](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43788715688), ten successful jobs. [Draft, unmerged PR #107](https://github.com/MoralityLabAI/LoopedTransformers/pull/107). The source, registered thresholds, training results, and failed scientific verdict remain unchanged.

- Registered `PROTOCOL.json` SHA-256: `df8f6cec2a7c79931278e4cd0c01d8bba38204432b1e4988a83b425e4bb47d5f`.
- Held-out `fresh_manifest.json` SHA-256: `18b97f3351dd4addad38762aa95b9a3532acf66641edcbbc41111f95b1f28c33`.
- Original official aggregate `results.json` SHA-256: `9c37b107f0bf8f693da731b880e2f3b3830e2a727b95abd0b26cfa64fa77325b`.
- All **eight original seed artifact ZIPs** were retrieved and verified against GitHub's SHA-256 digests. Their `evaluation/evaluation.json` files contain the full five-arm held-out results. The aggregate `results.json` contains fixed/oracle results for `held_layout`, but omits them from its `held_depth4` and `held_depth5` subsections. This reporting omission explains the earlier incomplete comparison.
- The immutable [source-bound evidence release](https://github.com/MoralityLabAI/LoopedTransformers/releases/tag/af22-ci-evidence-5c16b34-20261116) and [artifact checksum manifest](AF22_EVIDENCE_MANIFEST.json) preserve the original scoring receipts. The release contains the pre-correction report snapshot; this branch contains the corrected narrative.

## 1. Question, scope, and experiment

AF-22 tests whether a frozen **6,036-parameter TypedLoop** model with four native recurrent calls can improve its own proof completion when its outputs are used to score legal Horn successors. Eight initializations (17, 29, 43, 61, 73, 89, 103, 131) received 240 fixed AdamW updates on 928 inherited training cases; 352 cases were used for development. The held-out family contains **416** new physical-layout cases on layouts 82–86: 32 layout cases, 192 depth-4 cases, and 192 depth-5 cases.

The original experiment executes **five** policies on each held-out case. The three neural policies use the same frozen weights within each seed:

| Arm | Decision rule |
|---|---|
| **Direct** | Original neural logits over legal Horn rules and STOP |
| **VI-1** | Minimize one-step learned successor cost or learned root STOP cost |
| **VI-2** | Minimize a two-step Bellman backup or learned root STOP cost |
| **Fixed** | Fire the first legal physical rule, continuing until the goal, no rule, or the exact episode budget |
| **Shortest** | Exact shortest-proof host oracle; theoretical upper control |

The host enumerates and enforces legal rule firings. It has access to the public Horn graph, current facts and STOP, while the learned inference heads receive no true BFS distances. The oracle is an evaluation control, not a neural inference feature. All accepted Horn proofs are independently verified.

The primary depth-4 and depth-5 scores each cover **80 provable goals** out of 192 cases per depth per seed. The budget is the independent shortest-proof length plus one firing. The remaining 112 cases per depth are unprovable and are needed to evaluate appropriate STOP decisions.

## 2. Registered outcome and missing baseline

All eight seeds passed the preregistered 65% direct-policy train/development eligibility requirement. The primary gate required **each qualified seed** to exceed Direct by at least 5 percentage points with VI-2, reach **70% absolute success at both held-out depths**, limit excess premature STOP to 2 percentage points, remain within 3 points of VI-1, and accept zero false Horn proofs. **Every seed fails the full two-depth gate. The registered science verdict is FAIL.**

The eligibility and absolute-success thresholds were weak relative to the task's uninformed baseline. Fixed achieves **60/80 (75%)** at depth 4 and **64/80 (80%)** at depth 5 with no model. A separate reviewer recomputation reports **125/150 (83.3%)** provable development completions for fixed, versus the 65% neural qualification threshold; that development baseline is labeled a reviewer reconstruction rather than an official aggregate field. Three eligible seeds (29, 43, 131) solve zero held-out provable goals with Direct at both depths.

### Full five-arm results: solved provable goals out of 80 per seed

**Held-out depth 4**

| Seed | Direct | VI-1 | VI-2 | Fixed | Shortest oracle |
|---:|---:|---:|---:|---:|---:|
| 17 | 58 | 58 | 59 | 60 | 80 |
| 29 | 0 | 2 | 14 | 60 | 80 |
| 43 | 0 | 0 | 0 | 60 | 80 |
| 61 | 54 | 50 | 57 | 60 | 80 |
| 73 | 54 | 40 | 53 | 60 | 80 |
| 89 | 24 | 60 | 62 | 60 | 80 |
| 103 | 12 | 0 | 20 | 60 | 80 |
| 131 | 0 | 36 | 42 | 60 | 80 |

**Held-out depth 5**

| Seed | Direct | VI-1 | VI-2 | Fixed | Shortest oracle |
|---:|---:|---:|---:|---:|---:|
| 17 | 56 | 36 | 55 | 64 | 80 |
| 29 | 0 | 0 | 0 | 64 | 80 |
| 43 | 0 | 0 | 0 | 64 | 80 |
| 61 | 0 | 0 | 0 | 64 | 80 |
| 73 | 40 | 26 | 32 | 64 | 80 |
| 89 | 6 | 20 | 32 | 64 | 80 |
| 103 | 6 | 0 | 0 | 64 | 80 |
| 131 | 0 | 20 | 28 | 64 | 80 |

**Pooled across eight seeds**

| Arm | Depth 4, /640 | Depth 5, /640 |
|---|---:|---:|
| Direct | 202 (31.56%) | 108 (16.88%) |
| VI-1 | 246 (38.44%) | 102 (15.94%) |
| VI-2 | 307 (47.97%) | 147 (22.97%) |
| **Fixed** | **480 (75.00%)** | **512 (80.00%)** |
| Shortest oracle | 640 (100%) | 640 (100%) |

**Fixed meets or exceeds neural performance in 47 of 48 neural-arm × seed × depth cells.** The single exception is seed 89, depth-4 VI-2, at 62/80 against fixed's 60/80. VI-2 is **173 proofs below fixed** at depth 4 and **365 below fixed** at depth 5 in pooled counts. The older report emphasized VI-2's gain over Direct (+105 and +39), which is insufficient to infer useful neural search against the available zero-parameter policy.

The original report selectively highlighted seeds 89 and 131. Their depth-4 improvements are mostly due to the switch from Direct logits to the **VI-1 learned-cost/STOP scoring rule**:

| Seed | Direct → VI-1, depth 4 | VI-1 → VI-2, depth 4 |
|---:|---:|---:|
| 89 | 24 → 60 (**+36**) | 60 → 62 (**+2**) |
| 131 | 0 → 36 (**+36**) | 36 → 42 (**+6**) |

Across all seeds, Direct → VI-1 changes pooled completion by **+44** at depth 4 and **−6** at depth 5, with individual depth-4 changes from −14 to +36. VI-1 → VI-2 is nonnegative in all 16 seed-depth cells, adding **+61** depth-4 and **+45** depth-5 proofs. The second Bellman backup therefore improves this *particular learned-cost policy* consistently in the saved experiment, while the resulting policy remains markedly below the fixed control.

## 3. Little branching and STOP-dominated outcomes

The original signed per-episode traces establish that the **80 provable starting states** at each depth have:

- **42** start states with precisely one enabled rule and **38** with two.
- **4.425** neural evaluation nodes per two-step neighbourhood on average; 354 nodes across one 80-case provable depth family.
- An independent reviewer's graph reconstruction additionally finds **50/80** starts where every enabled rule is optimal. This optimal-action multiplicity count is from that reconstruction, rather than a field in the original aggregate.

On the 1,280 provable initial VI-2 decisions across eight seeds and both depths, the retrospective diagnostic records **465 optimal rule choices** and **796 immediate STOP choices**, a combined **1,261/1,280 (98.5%)**. Only 19 initial choices are suboptimal non-STOP rules. The dominant failure is STOP selection.

**Initial STOP decisions** can be reconstructed directly from the saved `episodes.jsonl` records. Counts below use all eight seeds at each depth; every provable STOP is premature, while STOP is a correct termination decision for an unprovable goal.

| Arm | Depth 4: STOP on provable /640 | Depth 4: STOP on unprovable /896 | Depth 5: STOP on provable /640 | Depth 5: STOP on unprovable /896 |
|---|---:|---:|---:|---:|
| Direct | 392 | 854 | 478 | 858 |
| VI-1 | 382 | 884 | 526 | 896 |
| VI-2 | 308 | 848 | 488 | 886 |
| Fixed | **0** | 384 | **0** | 384 |
| Shortest oracle | **0** | **896** | **0** | **896** |

First-legal only selects STOP when no physical rule remains enabled. For unprovable cases with irrelevant enabled rules, the correct optimal decision is often to stop immediately; that is the plausible learnable component of this small gym. The neural models frequently classify those correctly, but at the price of substantial premature STOP on provable cases. Future evaluations should jointly measure this discrimination, additional rule firings, and proofs completed.

Depth-5 VI-1 has higher premature STOP than VI-2 wherever the two differ. For example, seed 17 moves **45% → 30%** under the second backup. At depth 4, individual exceptions exist (seed 17 rises from 22.5% to 25%). The learned-cost STOP rule appears at the Direct → VI-1 intervention; VI-2 partly repairs its effect, rather than being the sole source of premature stopping.

## 4. Bellman formula, reachability sign, and retrospective mechanism

For a state `s` with estimated reachability `p(s)=sigmoid(r(s))` and clipped distance estimate `d(s)`, the frozen implementation uses

\[
V_0(s) = p(s)d(s) + [1-p(s)]16 = 16 - p(s)[16-d(s)],
\qquad C_{\mathrm{STOP}}(s) = 16p(s).
\]

For a nonterminal successor `s'`, the one-step interior backup is

\[
V_1(s')=\min\left\{16p(s'),\ 1+\min_{a\in A(s')}V_0(T(s',a))\right\}.
\]

When distance is fixed below 16, `V0` **decreases** with estimated reachability while the interior **STOP cost increases** with it. When `16p(s')` is selected in the minimum, the VI-2 rule score is instead governed by **`1+16p(s')`**, reversing the dependence on reachability. For positive Horn closure, legal rule firing preserves the exact eventual-reachability label. In these short provable tasks, an exact reachable-state estimate with shortest costs would make an interior STOP candidate dominated by continuing toward a goal. The observed interior STOP binding is thus evidence about learned-head error, subject to the fixed cap and recorded horizon.

A fresh reconstruction of the **original stored neural node ledgers** found that interior STOP wins the min-plus comparison against a nonempty set of second-step continuations in **336/944 candidate successor nodes (35.6%)** at depth 4 and **614/944 (65.0%)** at depth 5. These are post-scoring diagnostics, computed on each of the 640 provable first-decision traces per depth; the same node may be visited in different seed policies. They count *binding cost branches*, rather than causal effects on final completion. The earlier reviewer could not measure this because the full artifact ZIPs were inaccessible; the immutable original ZIPs were subsequently retrieved and verified.

The stored initial neural STOP costs directly imply average predicted reachability `C_STOP/16` on **known-provable** goals as low as **0.041** and as high as **0.558** across the 16 seed-depth groups. Several models therefore assign low reachability to solvable roots.

### Conditional mathematics and certificate limits

For the same legal transition graph and STOP boundary, the min-plus operator is nonexpansive:

\[
\|\mathcal TV-\mathcal TW\|_\infty\le\|V-W\|_\infty.
\]

When action-value errors are uniformly bounded by `ε` and the exact gap between optimal and suboptimal actions exceeds `2ε`, choosing the estimated minimum is optimal. The mathematical statements are correct. Lean 4.19.0 independently compiled `BellmanIteration.lean`, which proves a conditional exact-cost descent termination statement.

The retrospective action certificate uses the **actual exact costs** to calculate `ε` and `Δ` on the observed inputs. Its **74 certified decisions** (50 at depth 4, 24 at depth 5), each optimal, are consistent with the theorem **by construction**. Zero certified false positives provides no independent evidence of neural calibration or a deployable uncertainty certificate. The certification rate is 74/1,280 (5.8%) on provable initial decisions, and the check is retrospective.

See [Bellman certificate note](AF22_BELLMAN_CERTIFICATE.md) and [original post-hoc diagnostic JSON](AF22_POSTHOC_VALUE_DIAGNOSTIC.json). Neither changes the registered FAIL.

## 5. Independent verification and reproducibility

The source-pinned hosted run passed **ten jobs**: preflight including Lean 4.19.0, eight independent Torch 2.5.1 training/evaluation jobs and aggregate. The audit checked **16,640 proof episodes** (= 8 seeds × 416 cases × **five arms**), **56,814 planner-node records**, and **606 checkpoint-forward replays**, with **zero false accepted Horn proofs**. The three neural modes share the identical checkpoint within each seed. The symbolic fixed and shortest baselines do not use these weights.

An independently runnable [original-receipt control audit](af22_original_receipt_audit.py) recomputes all five-arm solved counts, first-action STOP distributions and interior STOP binding using only the eight original seed ZIPs and the Python standard library. It checks each ZIP SHA-256 and each episode member digest against the original receipts. This audit adds no neural training or held-out model selection.

The original nine artifact ZIPs, including the official aggregate, are preserved in the immutable [AF-22 evidence release](https://github.com/MoralityLabAI/LoopedTransformers/releases/tag/af22-ci-evidence-5c16b34-20261116). The release TAR is **3,973,120 bytes**, SHA-256 `9221513b43ce48a151990e06e8392738fc474fbe0fb065adea817e1de5aec07c`. The [manifest](AF22_EVIDENCE_MANIFEST.json) enumerates every original ZIP and member SHA-256, enabling independent verification of this report's tables and traces. This corrected report changes documentation only.

The test is a finite, synthetic, positive-Horn-rule gym with host-controlled execution. Verification supports the soundness of accepted proofs and exact replay of the recorded arithmetic. It does not show autonomous Lean proving, nontrivial proof search, or useful generalizable neural heuristics.

## 6. Interpretation and requirements for AF-23

AF-22 demonstrates that switching from Direct neural logits to learned-cost scoring changes behavior substantially, and that a second Bellman backup repairs a portion of those mistakes. **It does not establish a capability advantage over uninformed legal-rule execution.** Fixed is stronger in 47 of 48 paired cells, and the proof graphs offer minimal branching at their roots. The result is primarily about **STOP calibration and scoring-rule sensitivity** in a compact model.

A successor experiment should have two separately registered tracks:

1. **STOP discrimination in the existing gym.** Publish all five arms on every split and seed; jointly score erroneous STOP on provable inputs and correct initial STOP on unprovable inputs, with computational cost. Require improvement against **first-legal fixed** on proof completion, with a separate required gain over the fixed rule budget or other appropriate efficiency target. Compare a root-tied reachability arm (same `p(root)` reused across the two-step neighbourhood) and a no-interior-STOP arm. Report how often the interior STOP boundary binds and how often it alters the chosen action.
2. **Genuine branching proof search.** Register new graphs with several enabled actions, misleading branches and meaningful cost differences, plus an explicit minimum branching/optimal-action-ambiguity distribution before training. Score proof length and success under equal inference budgets against first-legal, randomized legal-rule, oracle, and host BFS/A* baselines. Hold out graph structures and avoid conditioning on already-scored seed selections.

Registration must precede any AF-23 results. AF-22's source, original registered FAIL, and complete negative evidence remain frozen. [PR #107](https://github.com/MoralityLabAI/LoopedTransformers/pull/107) stays **draft and unmerged**.
