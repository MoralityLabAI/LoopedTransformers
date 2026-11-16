# AF-26: Frozen-rule ranking dominates the Bellman-policy proof deficit

**Preregistered factorial outcome: RULE_ONLY.**  
**Hosted engineering and independent proof audit: PASS, all eleven source-pinned jobs after one failed source-parity attempt and one bounded retry.**  
**Result date:** November 16, 2026.

## Abstract

AF-26 factorizes the rule-ranking and STOP decisions of the original fixed **6,036-parameter TypedLoop** neural checkpoint from AF-25. In that parent study, original Direct neural action selection completed **240/384** repeated hard depth-five Horn proofs, while the same weights scored through two-step Bellman VI-2 completed **131/384**.

Using the original eight frozen checkpoints, AF-26 independently reruns six closed-loop policies crossing the native Direct/VI-2 STOP gates with their legal-rule rankings. Source-only pre-registration fixed a **109-proof original gap**, a **55-proof half-gap recovery criterion**, the exact source policy controls and the independent host audit before scoring.

The official all-seed result is:

- Direct/Direct (DD): **240/384** hard positive proofs
- Direct STOP + VI-2 rules (DV): **114/384**
- VI-2 STOP + Direct rules (VD): **288/384**
- VI-2/VI-2 (VV): **131/384**

Therefore, swapping **only the legal-rule ranking** from VI-2 to Direct while preserving VI-2's STOP gate recovers **157 proofs**, exceeding the registered half-gap by a substantial margin. Swapping only the STOP gate loses **17 proofs**. The observed factorial interaction is **−31 proofs**. The preregistered classification is **RULE_ONLY**.

The result identifies a specific closed-loop *policy-level* source of the fixed-weight performance gap in the AF-25 synthetic grammar. It does not establish an equal-compute neural-search advantage, autonomous theorem proving, or a unique causal mediation through neural heads. Reproducibility is qualified by rare exact-tie-sensitive VI-2 decisions across hosted CPU runs, described below.

## 1. Frozen source and registration

- **Scientific registration:** [AF-26 PROTOCOL.json](../../experiments/activation_funnel_af26/PROTOCOL.json), first committed at [`aadb956`](https://github.com/MoralityLabAI/LoopedTransformers/commit/aadb9569092b582f1bf1e12ebb7fd6b213fa82ee) **before AF-26 scored policy interventions**. Protocol SHA-256: `aeec1069863a06f8f11bce6a841378f122d8ebd6fad253fb3f1896bd5acfc0b9`.
- **Original AF-25 frozen evidence:** [scored CI #43829157457](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43829157457), source commit `13c5301a90590d98e0e0efa066247f0f66efdeed`. Original aggregate `af25_results.json` SHA-256 `5624d2bfe913c870dada6be00a722eceb8157be26824a875d36d0723eec99163`, from original Actions ZIP artifact ID `14208435623` (ZIP SHA-256 `5ca614b06cee3e2e3fdccb64bb842cda9533938d3dfb001fb3c4861effbc1f5c`).
- **Original weights:** all eight immutable original AF-23 fixed-final-update checkpoints, source run [#43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916), pinned by per-artifact/member SHA-256, original training receipts and exact Torch/NumPy state-dict parity. **No additional optimizer updates or checkpoint selection**.
- **Preregistered pure-policy tests:** [source-only CI #43864568599](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43864568599) passed all nine tests after correcting a synthetic illegal-rule-mask test fixture.
- **Official hosted factorial evidence:** [run #43866180821](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43866180821), successful **attempt 2** at immutable source-lock commit `a130f5e592b6efb08d0d91316d69d5b8c0d473e2`. All eight independent seed audits and the aggregate passed.
- **Official aggregate:** artifact `af26-factorial-aggregate`, ID **14226413322**, artifact ZIP SHA-256 `c41aba7a9bb246b734b2ca42f7804c01c44bbace62056720a2daad04f3a52f9f`. Its internal `af26_results.json` SHA-256 is `3093f8687c36a48d4145fee2b4cfbcc0d9dcf3ed9c70d2d5faf918cf3747e410`. The same ZIP also preserves the verified original AF-25 aggregate used for exact parity comparisons.

This report was committed **after scoring**. Original AF-25 code, model weights, registered scientific FAIL, and all earlier research outcomes remain unchanged. [AF-26 draft PR #133](https://github.com/MoralityLabAI/LoopedTransformers/pull/133) remains unmerged.

## 2. Experiment and exact inference intervention

Every source checkpoint uses four native recurrent calls per visited state. The host alone enumerates currently enabled physical Horn rules and verifies each legal rule firing. The original Direct and VI-2 scorers both receive exactly the same model weights and public graph at a state, while differing in the action scores they derive.

AF-26 builds a **binary STOP gate** from each source scorer's native full legal-action/STOP argmax. If the gate chooses to continue, a separate registered **rule-ranking source** supplies the highest-scoring *enabled* physical rule. No STOP logit from one scorer is added to a rule score on another scorer's numerical scale.

| Registered arm | STOP decision source | Legal-rule ranking source |
|---|---|---|
| **DD** | Direct | Direct, original AF-25 policy |
| **DV** | Direct | VI-2 |
| **VD** | VI-2 | Direct |
| **VV** | VI-2 | VI-2, original AF-25 policy |
| **ND** | Continue if a legal rule exists | Direct |
| **NV** | Continue if a legal rule exists | VI-2 |

Each arm produces a genuine closed-loop proof episode: after a selected legal rule fires, new model states and complete Bellman neighborhoods are recomputed at the resulting host state. DD and VV use AF-25's unchanged **three-mode Direct/VI-1/VI-2 source batching** to reproduce those original policies, with their **newly executed** source-node ledgers factored into the six-arm result. The four crossed/forced-continuation arms independently reroll their newly reached states. This engineering specification was introduced in a dated source-lock amendment following the initial batch-shape parity failure.

AF-25's original held-out cohorts are reused exactly. Each depth contains **80 provable** and **112 unprovable** cases per seed. The primary depth-five hard family has **48 positive unique Horn graphs** per seed; these same 48 graphs are repeated for eight fixed initializations, giving 384 *seeded outcomes*, not 384 independent graph instances. The official action budget is shortest independent physical Horn proof length plus one rule firing; unprovable inputs have two decision steps.

## 3. Primary registered half-gap result

The preregistered primary endpoint is completed hard depth-five proofs in the original `decoy_and_shared_overcount` family.

| Arm | Completed /384 | Success rate |
|---|---:|---:|
| DD | 240 | 62.50% |
| DV | 114 | 29.69% |
| **VD** | **288** | **75.00%** |
| VV | 131 | 34.11% |
| ND | 288 | 75.00% |
| NV | 131 | 34.11% |

The registered comparisons are

\[
G=N_{DD}-N_{VV}=240-131=109,
\]

\[
\Delta_{\rm STOP}=N_{DV}-N_{VV}=114-131=-17,
\]

\[
\Delta_{\rm RULE}=N_{VD}-N_{VV}=288-131=157.
\]

The factorial interaction is

\[
I=N_{DD}-N_{DV}-N_{VD}+N_{VV}
=240-114-288+131=-31.
\]

The exact identity \(G=\Delta_{\rm STOP}+\Delta_{\rm RULE}+I\) reproduces \(109=-17+157-31\). The source-locked registered half-gap threshold is **55 completed proofs**.

**Registered outcome: RULE_ONLY.** Holding the VI-2 STOP gate fixed and replacing only VI-2's legal-rule ranking with Direct ranking recovers substantially more than half the original performance deficit. Replacing only the STOP gate fails that criterion and decreases completed proofs.

### Paired proof outcomes (same case and checkpoint)

| Comparison against original VV | Newly completed | Previously completed but now failed | Both completed | Neither completed |
|---|---:|---:|---:|---:|
| **VD versus VV** | **157** | **0** | 131 | 96 |
| DV versus VV | 0 | 17 | 114 | 253 |
| DD versus VV | 126 | 17 | 114 | 127 |
| NV versus VV | 0 | 0 | 131 | 253 |

The legal-rule swap **rescued 157 complete hard proofs without losing any previously completed VV hard proof** among the 384 seeded pairs. The gate-only swap **lost 17** without rescuing any. The forced-continuation NV arm exactly reproduces VV's hard successes, further supporting rule-ranking as the dominant tested limitation on this family.

### Every original seed, hard depth-five cases /48

| Seed | DD | DV | VD | VV | ND | NV |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 48 | 0 | 48 | 0 | 48 | 0 |
| 29 | 48 | 37 | 48 | 37 | 48 | 37 |
| 43 | 0 | 0 | 0 | 0 | 0 | 0 |
| 61 | 48 | 48 | 48 | 48 | 48 | 48 |
| 73 | 0 | 0 | 0 | 0 | 0 | 0 |
| 89 | 48 | 0 | 48 | 0 | 48 | 0 |
| 103 | 0 | 0 | **48** | 17 | **48** | 17 |
| 131 | 48 | 29 | 48 | 29 | 48 | 29 |

Seed 103 accounts for the nonadditive interaction: the Direct-only source policy fails all 48 hard cases, yet coupling Direct rule ranking with VI-2's STOP gate completes every case. This illustrates why native discrete policy swaps cannot be interpreted as a unique additive neural-head mediation.

The mathematical identity, exact-source action-recovery theorem, and finite counterexample to single-factor mediation are developed separately in [AF26_FACTORIAL_IDENTITIES.md](AF26_FACTORIAL_IDENTITIES.md).

## 4. Secondary full-depth results

Each row is pooled over **640 provable outcomes** per depth, from the fixed 80 graph cases and eight neural checkpoints.

| Policy | Held depth 5 solved /640 | Held depth 6 solved /640 |
|---|---:|---:|
| DD, original Direct | **432 (67.50%)** | 128 (20.00%) |
| DV, Direct gate + VI-2 rules | 178 (27.81%) | 89 (13.91%) |
| **VD, VI-2 gate + Direct rules** | **512 (80.00%)** | **240 (37.50%)** |
| VV, original VI-2 | 195 (30.47%) | 146 (22.81%) |
| ND, never STOP + Direct rules | 512 (80.00%) | **320 (50.00%)** |
| NV, never STOP + VI-2 rules | 195 (30.47%) | 146 (22.81%) |

Holding the VI-2 STOP gate fixed, substituting Direct rule ranking improves completion by **317/640 proofs at depth five** and **94/640 at depth six**. By itself, disabling STOP produces no gain with VI-2 rule ordering, whereas disabling STOP with Direct rule ordering improves depth-six completion relative to VD by 80 proofs.

The depth-six difference therefore leaves a genuine role for STOP decisions, particularly later in proof construction. The registered **RULE_ONLY** label applies to the primary **depth-five hard family** and does not assert that STOP calibration is unimportant in other settings.

### Family breakdown

| Family / arm | D5 hard /384 | D5 balanced /256 | D6 hard /384 | D6 balanced /256 |
|---|---:|---:|---:|---:|
| DD | 240 | 192 | 96 | 32 |
| DV | 114 | 64 | 38 | 51 |
| VD | **288** | **224** | **144** | 96 |
| VV | 131 | 64 | 95 | 51 |
| ND | 288 | 224 | **192** | **128** |
| NV | 131 | 64 | 95 | 51 |

## 5. STOP calibration remains weak

Correct immediate STOP on an **unprovable** goal is a separate task. Here the denominator is **896 negative episodes** per depth (112 cases × eight seeds).

| Arm | D5 correct initial STOP /896 | D6 correct initial STOP /896 |
|---|---:|---:|
| DD | 588 (65.63%) | 672 (75.00%) |
| DV | 588 (65.63%) | 672 (75.00%) |
| VD | 560 (62.50%) | 658 (73.44%) |
| VV | 560 (62.50%) | 658 (73.44%) |
| ND | **0** | **0** |
| NV | **0** | **0** |

As expected by construction, the STOP gate determines the initial STOP decision for the two rank variants that share it. Force-continuation controls lose all correct initial STOP decisions on the negative inputs in this gym. Their provable completion figures are therefore **diagnostic upper comparisons**, not a calibrated solution.

The number of **later premature STOP decisions on provable inputs** changes with the rule-ranking source because the policy subsequently visits different states:

| Arm | D5 later premature STOP /640 | D6 later premature STOP /640 |
|---|---:|---:|
| DD | 128 | 320 |
| DV | 145 | 361 |
| VD | **0** | **128** |
| VV | 160 | 267 |
| ND | 0 | 0 |
| NV | 0 | 0 |

Switching to Direct rule ordering under the same VI-2 STOP gate eliminates observed later premature stopping on depth-five positives, even though the gate itself is unchanged. A plausible mechanism is that better-ranked paths reach states where the value-based gate no longer favors stopping. The saved policy-level intervention establishes that the **sequence of reached states matters**, while the specific neural feature responsible remains a separate mechanistic question.

## 6. Independent verification

The successful source-pinned run passed its source lock, genuine original-checkpoint smoke, all eight seed training-free evaluations and audits, and independent aggregation.

| Verification item | Official total |
|---|---:|
| Eight original checkpoint identities | 8 |
| Host-checked proof episodes | **18,432** |
| Independently reconstructed neural planner-node records | **616,064** |
| Original checkpoint Torch forward replays | **192** |
| False accepted Horn proofs | **0** |
| Original models retrained | **0** |

Each saved decision carries the complete original two-step public host-state ledger, native Direct and VI-2 score vectors, Boolean gate choice, independent legal-rule rank and selected rule. The separate verifier reconstructs physical incidence from model-visible tensors, independently recomputes shortest proof cost, checks every proof step and numerical Bellman backup, and replays sampled true model forwards. It also independently recomputes all per-seed source hashes and requires **exact original AF-25 per-seed DD/VV proof-success parity at both depths**, followed by pooled and primary hard-family parity.

Actual work differs among policies because they reach different states and STOP at different times. The recorded total neural boundary-node evaluations are not a matched FLOP or latency comparison, and the source-original baseline requires an additional VI-1 grouped inference arm for exact CPU-batch parity. The algorithmic inference cost must remain explicit when comparing with AF-25's perfectly successful higher-expansion BFS64 symbolic diagnostic.

## 7. Numerical reproducibility and engineering amendments

Two earlier failed hosted attempts remain public and must be read alongside this scored result.

1. [**CI #43864977372**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43864977372) used a six-policy joint CPU batching schedule. Its VI-2 original-control parity check failed for seeds **103 and 131**. This was a numerical/source-reproduction failure, not an AF-26 scientific verdict. DD and VV were changed to use the original AF-25 three-mode batching path, while all four crossed and forced-continuation policies remained fresh closed-loop evaluations. The original protocol and outcome thresholds remained unchanged.
2. [**CI #43866180821**, attempt 1](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43866180821) used that corrected source path and passed seven independent seed jobs, while seed **29** still failed VI-2 original policy parity. No aggregate result was certified from that first attempt.
3. [**Read-only seed-29 source diagnostic #43867433873**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43867433873) verified the exact original model ZIP and AF-25 per-seed proof archive. A separate fresh use of the original AF-25 source replayed **every Direct, VI-1 and VI-2 action** and returned the exact original **37/80 VI-2 depth-five** and **20/80 VI-2 depth-six** scores. Its original source VI-2 receipt has **116 exactly tied selected-action comparisons at depth five** and **147 at depth six**, while Direct has none.
4. A single bounded re-run of failed CI jobs, using the **same source commit and preregistered thresholds**, completed the missing seed-29 independent proof audit. [CI #43866180821 attempt 2](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43866180821) then aggregated all eight verified seed receipts and produced the original-score-parity-checked **RULE_ONLY** result.

These failures show sensitivity of near-tied FP32 VI-2 action selection to the execution/batching environment. A successful source-locked GitHub Actions run therefore does **not** establish hardware-independent exact action reproducibility. The accepted attempt is disclosed alongside the unsuccessful ones.

### Seven-seed robustness bound established before the retry

The seven originally passing independent seed receipts produced hard depth-five counts:

\[
(N_{DD},N_{DV},N_{VD},N_{VV})=(192,77,240,94).
\]

Original AF-25's immutable eighth seed (29) had \(N_{DD}=48\), \(N_{VV}=37\) on these same 48 hard cases. Regardless of its new crossed outcomes \(0\le N_{DV,29},N_{VD,29}\le48\), the final two one-factor recoveries were mathematically confined to

\[
-54\le\Delta_{\rm STOP}\le-6,\qquad
109\le\Delta_{\rm RULE}\le157.
\]

Thus the **RULE_ONLY half-gap classification was already logically forced by the seven independently audited seeds and the frozen original eighth-seed controls**, before the rerun. The successful eighth-seed receipt narrows these bounds to **−17** and **157** without changing the classification. This limits the effect that selective runner success could have on the qualitative preregistered conclusion, while preserving the numerical reproducibility concern.

## 8. Scientific interpretation and next test

The AF-26 factorial intervention identifies **legal-rule ranking** as the dominant source of the original VI-2 performance deficit on depth-five hard Horn proofs. It provides source-locked closed-loop evidence that retaining VI-2's native STOP gate while switching to Direct rule selection performs markedly better than the original VI-2 policy in this finite, synthetic gym.

The result is conditional on the frozen eight original models, the specific AF-25 Horn grammar, the original shortest+one proof-action budgets and the source-matched CPU score implementation. The repeated graph roster restricts statistical independence, and observed per-seed variation remains substantial. It does not show a universal priority ordering for neural heads or that no value estimate could improve planning.

A prospectively registered follow-up should preserve Direct's rule-ranking head, then calibrate a separate STOP decision on **fresh train/development grammar families** with both provable and unprovable cases. A second, independent question is whether learned value estimates can help rank rules when Direct logits are ambiguous. Both studies need fresh held-out graphs and a computation-matched symbolic baseline, including the previously perfect BFS64 controller.

**Research status: AF-26 registered RULE_ONLY factorial classification, independent engineering PASS with disclosed source-replay amendments.** Earlier AF-22 through AF-25 registered results remain untouched. Keep [PR #133](https://github.com/MoralityLabAI/LoopedTransformers/pull/133) **draft and unmerged** pending independent review.
