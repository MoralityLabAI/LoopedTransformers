# AF-26: Frozen-rule ranking dominates the Bellman-policy proof deficit

**Preregistered factorial outcome: RULE_ONLY.**  
**Hosted engineering and independent proof audit: PASS, all 11 jobs in the latest source-pinned archived-control run. Two earlier source-parity attempts and an earlier successful retry remain preserved.**  
**Result date:** November 16, 2026.

## Abstract

AF-26 factorizes the rule-ranking and STOP decisions of the original fixed **6,036-parameter TypedLoop** neural checkpoint from AF-25. In that parent study, original Direct neural action selection completed **240/384** repeated hard depth-five Horn proofs, while the same weights scored through two-step Bellman VI-2 completed **131/384**.

Using the eight frozen checkpoints, AF-26 evaluates four fresh closed-loop interventions crossing the native Direct/VI-2 STOP gates with legal-rule rankings, alongside the two original AF-25 source-authenticated DD/VV controls. Source-only pre-registration fixed a **109-proof original gap**, a **55-proof half-gap recovery criterion**, the exact source policy controls and the independent host audit before scoring.

The official all-seed result is:

- Direct/Direct (DD): **240/384** hard positive proofs
- Direct STOP + VI-2 rules (DV): **112/384**
- VI-2 STOP + Direct rules (VD): **288/384**
- VI-2/VI-2 (VV): **131/384**

Therefore, swapping **only the legal-rule ranking** from VI-2 to Direct while preserving VI-2's STOP gate recovers **157 proofs**, exceeding the registered half-gap by a substantial margin. Swapping only the STOP gate loses **19 proofs**. The observed factorial interaction is **−29 proofs**. The preregistered classification is **RULE_ONLY**.

The result identifies a specific closed-loop *policy-level* source of the fixed-weight performance gap in the AF-25 synthetic grammar. It does not establish an equal-compute neural-search advantage, autonomous theorem proving, or a unique causal mediation through neural heads. Reproducibility is qualified by rare exact-tie-sensitive VI-2 decisions across hosted CPU runs, described below.

## 1. Frozen source and registration

- **Scientific registration:** [AF-26 PROTOCOL.json](../../experiments/activation_funnel_af26/PROTOCOL.json), first committed at [`aadb956`](https://github.com/MoralityLabAI/LoopedTransformers/commit/aadb9569092b582f1bf1e12ebb7fd6b213fa82ee) **before AF-26 scored policy interventions**. Protocol SHA-256: `aeec1069863a06f8f11bce6a841378f122d8ebd6fad253fb3f1896bd5acfc0b9`.
- **Original AF-25 frozen evidence:** [scored CI #43829157457](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43829157457), source commit `13c5301a90590d98e0e0efa066247f0f66efdeed`. Original aggregate `af25_results.json` SHA-256 `5624d2bfe913c870dada6be00a722eceb8157be26824a875d36d0723eec99163`, from original Actions ZIP artifact ID `14208435623` (ZIP SHA-256 `5ca614b06cee3e2e3fdccb64bb842cda9533938d3dfb001fb3c4861effbc1f5c`).
- **Original weights:** all eight immutable original AF-23 fixed-final-update checkpoints, source run [#43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916), pinned by per-artifact/member SHA-256, original training receipts and exact Torch/NumPy state-dict parity. **No additional optimizer updates or checkpoint selection**.
- **Preregistered pure-policy tests:** [source-only CI #43864568599](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43864568599) passed all nine tests after correcting a synthetic illegal-rule-mask test fixture.
- **Latest official hosted factorial evidence:** [run #43868738894](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43868738894), immutable scored source commit `9e85b31ab598d91b56c7d9545ee543d6e992f426`. All 11 jobs passed, including eight independent seed audits. An earlier [successful bounded retry of #43866180821](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43866180821) reported the same registered RULE_ONLY class with two different borderline counterfactual successes; that historical run remains in the commit history.
- **Latest official aggregate:** artifact `af26-factorial-aggregate`, ID **14227096110**, original artifact ZIP SHA-256 `06182ddf35a29016d4aef7d6336a1b0c4e1659bcfad8e80ae6b2ccc25aac461d`. Its `af26_results.json` member SHA-256 is `72b11bf1c3c55e7e9e461c02929d737b107ad44bdfa1661a5ffc22c2d223f2ff`; the same archive retains the exact original AF-25 aggregate. The earlier successful retry aggregate (artifact 14226413322) remains historical evidence.

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

**DV, VD, ND and NV** run genuinely fresh closed-loop trajectories. Every newly reached state triggers a frozen-model forward over its host-verified two-step neighborhood. **DD and VV** are the original, independently audited AF-25 source proof-episode records, each recovered from a separately SHA-256-pinned original seed ZIP and rescored from its saved neural node ledger. The original native score vector, action and host proof are preserved exactly. Each source step is independently verified, and sampled true checkpoint forwards are compared numerically. This arrangement preserves rare source CPU tie breaks; DD/VV are explicitly historical controls rather than claimed fresh rerolls.

AF-25's original held-out cohorts are reused exactly. Each depth contains **80 provable** and **112 unprovable** cases per seed. The primary depth-five hard family has **48 positive unique Horn graphs** per seed; these same 48 graphs are repeated for eight fixed initializations, giving 384 *seeded outcomes*, not 384 independent graph instances. The official action budget is shortest independent physical Horn proof length plus one rule firing; unprovable inputs have two decision steps.

## 3. Primary registered half-gap result

The preregistered primary endpoint is completion of the **48 hard depth-five proofs**, evaluated once per original model seed, for 384 seeded observations.

| Arm | Completed /384 | Success rate |
|---|---:|---:|
| DD | 240 | 62.50% |
| DV | 112 | 29.17% |
| **VD** | **288** | **75.00%** |
| VV | 131 | 34.11% |
| ND | 288 | 75.00% |
| NV | 129 | 33.59% |

The originally registered Direct–VI-2 gap is

\[
G=N_{DD}-N_{VV}=240-131=109.
\]

The two one-factor recovery scores are

\[
\Delta_{\rm STOP}=N_{DV}-N_{VV}=112-131=\boxed{-19},
\]

\[
\Delta_{\rm RULE}=N_{VD}-N_{VV}=288-131=\boxed{157}.
\]

The factorial interaction is

\[
I=N_{DD}-N_{DV}-N_{VD}+N_{VV}
=240-112-288+131=\boxed{-29}.
\]

The exact accounting identity reproduces \(109=-19+157-29\). The registered half-gap criterion was **55** completed proofs, frozen before any AF-26 counterfactual scoring. **Registered outcome: RULE_ONLY.** Switching VI-2's legal-rule ranking to Direct while preserving VI-2's STOP gate recovers 157 hard proofs. Switching the STOP gate alone loses 19.

### Paired source proof outcomes

Each comparison uses the identical case and original checkpoint. The denominator is **384 paired seeded outcomes**, comprising 48 unique Horn graphs under eight model initializations.

| Comparison | First succeeds, second fails | Second succeeds, first fails | Both succeed | Neither succeeds |
|---|---:|---:|---:|---:|
| **VD vs VV** | **157** | **0** | 131 | 96 |
| DV vs VV | 0 | 19 | 112 | 253 |
| DD vs VV | 126 | 17 | 114 | 127 |
| ND vs DD | 48 | 0 | 240 | 96 |
| NV vs VV | 0 | 2 | 129 | 253 |

The VI-2-gated **VD** policy rescues 157 complete hard proofs without losing an original VV success in the paired set. The DV gate-only swap loses 19 and adds none. This is a **closed-loop policy intervention**: different selected rules can lead to different later neural STOP estimates.

### Every frozen source seed, hard depth-five cases /48

| Seed | DD | DV | VD | VV | ND | NV |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 48 | 0 | 48 | 0 | 48 | 0 |
| 29 | 48 | 36 | 48 | 37 | 48 | 36 |
| 43 | 0 | 0 | 0 | 0 | 0 | 0 |
| 61 | 48 | 48 | 48 | 48 | 48 | 48 |
| 73 | 0 | 0 | 0 | 0 | 0 | 0 |
| 89 | 48 | 0 | 48 | 0 | 48 | 0 |
| 103 | 0 | 0 | 48 | 17 | 48 | 17 |
| 131 | 48 | 28 | 48 | 29 | 48 | 28 |
| **Pooled** | **240** | **112** | **288** | **131** | **288** | **129** |

Seed 103 illustrates a substantial STOP × rank interaction. Its source Direct policy finishes none of the 48 hard cases, while the same Direct rule ranking paired with VI-2's STOP gate finishes **48/48**. Seed 61 succeeds under every arm, while seeds 43 and 73 fail under every arm. These differences illustrate the dependence on the original model initialization.

The exact policy identity, closed-loop interaction equation and counterexample to additive causal mediation appear in [AF26_FACTORIAL_IDENTITIES.md](AF26_FACTORIAL_IDENTITIES.md).

## 4. Secondary full-depth results

Each depth contains **80 positive cases × eight frozen model seeds = 640 completed-proof outcomes**. Source DD/VV controls are original authenticated AF-25 trajectories, and the four intervention/forced arms are fresh full-episode rerolls.

| Arm | Held depth 5 solved /640 | Held depth 6 solved /640 |
|---|---:|---:|
| DD, original Direct | **432 (67.50%)** | 128 (20.00%) |
| DV, Direct gate + VI-2 rules | 176 (27.50%) | 90 (14.06%) |
| **VD, VI-2 gate + Direct rules** | **512 (80.00%)** | **240 (37.50%)** |
| VV, original VI-2 | 195 (30.47%) | 146 (22.81%) |
| ND, never STOP + Direct rules | 512 (80.00%) | **320 (50.00%)** |
| NV, never STOP + VI-2 rules | 193 (30.16%) | 147 (22.97%) |

Holding the original VI-2 STOP gate fixed while replacing its legal-rule ranker with Direct improves **317/640** full-depth-five and **94/640** full-depth-six completions. Forcing continuation with Direct rule ranking adds a further **80 depth-six** successes relative to VD, while making every immediate unprovable STOP decision incorrect.

The preregistered RULE_ONLY class applies to **hard depth-five proofs**. STOP calibration remains consequential for deeper trajectories and correct termination.

### Positive-family breakdown

| Family / arm | D5 hard /384 | D5 balanced /256 | D6 hard /384 | D6 balanced /256 |
|---|---:|---:|---:|---:|
| DD | 240 | 192 | 96 | 32 |
| DV | 112 | 64 | 39 | 51 |
| **VD** | **288** | **224** | **144** | 96 |
| VV | 131 | 64 | 95 | 51 |
| ND | 288 | 224 | 192 | 128 |
| NV | 129 | 64 | 96 | 51 |

## 5. STOP calibration and future-state sensitivity

Correct immediate STOP on **unprovable goals** has denominator **896 negative episodes** per depth (112 unique negative cases × eight frozen seeds).

| Arm | D5 correct initial STOP /896 | D6 correct initial STOP /896 |
|---|---:|---:|
| DD | 588 (65.63%) | 672 (75.00%) |
| DV | 588 (65.63%) | 672 (75.00%) |
| VD | 560 (62.50%) | 658 (73.44%) |
| VV | 560 (62.50%) | 658 (73.44%) |
| ND | **0** | **0** |
| NV | **0** | **0** |

As expected, the native gate fixes the *initial* STOP decision across rankers that share it. Forced continuation eliminates correct initial STOP on every unprovable case where a legal rule remains, exposing the reachability cost of an always-fire policy.

Premature STOP on **provable goals** can occur at the root or later after the policy has taken different Horn-rule actions:

| Arm | D5 initial /640 | D5 later /640 | D6 initial /640 | D6 later /640 |
|---|---:|---:|---:|---:|
| DD | 32 | 128 | 112 | 320 |
| DV | 32 | 145 | 112 | 361 |
| VD | 0 | **0** | 112 | 128 |
| VV | 0 | 160 | 112 | 267 |
| ND | 0 | 0 | 0 | 0 |
| NV | 0 | 0 | 0 | 0 |

The VD intervention removes all observed premature STOP on depth-five positive cases while using the **same original VI-2 gate** as VV. The rule-ranking intervention changes the states reached by later STOP decisions. At depth six, VD still stops too early on 240/640 positive cases, including 112 initial and 128 later cases. This points to both future-state and STOP calibration as targets for a subsequent controlled test.

## 6. Independent verification and compute scope

The latest official source-pinned run [#43868738894](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43868738894) passed source lock, genuine frozen-checkpoint smoke, all eight full per-seed proof/audit jobs and final independent eight-seed aggregation. All original model checkpoints and source-scored policies have unchanged source hashes and original weights.

| Audit item | Completed count |
|---|---:|
| Eight original checkpoint identities | **8** |
| Exact original AF-25 source control actions verified | **16,162** |
| Independently audited six-arm proof episodes | **18,432** |
| Reconstructed source and new planner-node evaluations | **615,663** |
| Sampled genuine frozen checkpoint Torch forwards | **192** |
| Numerically verified forward samples with a near-tie action ambiguity | **1** |
| False accepted physical Horn proofs | **0** |
| New optimizer/training updates | **0** |

Every archived original DD/VV step is matched to its exact independently audited AF-25 source record, including the complete model-node ledger and selected rule. The physical incidence is reconstructed separately from the model-visible tensors; independent BFS checks shortest proof budgets. Numeric min-plus scores and native STOP/rule choices are recomputed, with sampled actual checkpoint forwards and limited logged numerical near-tie exceptions.

The table below records native boundary-node evaluations for each source policy, rather than claiming equal CPU runtime or matched FLOPs.

| Arm | Depth 5 model-node evaluations | Depth 6 model-node evaluations |
|---|---:|---:|
| DD, original source workload | 60,644 | 35,904 |
| DV, fresh | 52,260 | 33,625 |
| VD, fresh | 65,208 | 44,058 |
| VV, original source workload | 53,097 | 38,444 |
| ND, fresh | 67,168 | 55,296 |
| NV, fresh | 60,372 | 49,587 |

The original AF-25 source family was explicitly selected to defeat cheap graph-aware policies. A larger public BFS64 controller still completed every provable case; no general neural-vs-symbolic search advantage or equal-latency claim follows from AF-26.

## 7. Numerical source-parity issue and disclosed amendments

AF-26's initial engineering attempts exposed source-policy sensitivity to near-tied floating-point scores. The original complete scientific registrations and thresholds remained unchanged throughout.

1. [**CI #43864977372**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43864977372) grouped six policies together for neural CPU inference. Original VV proof-completion parity failed in seeds **103 and 131**. This was a reproducibility blocker, not a registered scientific outcome.
2. [**CI #43866180821, initial attempt**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43866180821) used the original three-mode grouped inference for live DD/VV and passed seven of eight independent seed jobs, while seed **29** still failed exact original VI-2 parity. The blocked source-locked attempt was retained.
3. A [**read-only seed-29 diagnostic #43867433873**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43867433873) independently validated the original model and original AF-25 episode archives, showing that exact tie-breaking and batch context can matter. Inspection of the immutable original seed-29 record finds **263 near-tied Direct/VI-2 action steps out of 2,003** with a top-two score margin below \(10^{-5}\), including exact ties. **A bounded retry of the earlier #43866180821** later completed all eight source audits and reported the same RULE_ONLY classification with **DV=114** and **NV=131**, producing a historical aggregate retained in GitHub Actions.
4. The latest independently [**scored run #43868738894**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43868738894) uses **SHA-pinned original AF-25 DD/VV per-seed proof episodes** rather than newly batched DD/VV controls. Every source action, score vector, numerical Bellman ledger, host proof, and sampled checkpoint forward is independently revalidated. The four crossed interventions are still freshly rerolled. This source-lock amendment produced **DV=112**, **NV=129**, an interaction of **−29**, and the same registered **RULE_ONLY** outcome with all 11 jobs passing.

The two complete scored runs differ slightly in near-tie-sensitive *crossed-arm* outcomes, while reproducing the registered DD/VV totals and the 157-proof rule-swap recovery. The difference is disclosed, and a successful CI result should not be read as bit-for-bit hardware-independent action reproducibility for barely separated FP32 scores.

### Robustness to the previously blocked seed

The seven originally passing seed receipts from the second attempt had

\[
(N_{DD},N_{DV},N_{VD},N_{VV})=(192,77,240,94)
\]

before the source reproduction of the eighth seed (29). With that seed's source DD=48 and VV=37, allowing either crossed arm 0–48 possible proof completions bounds the eventual one-factor recoveries:

\[
-54\le\Delta_{\rm STOP}\le-6,\qquad
109\le\Delta_{\rm RULE}\le157.
\]

These bounds are tied to the **historical second attempt**. They already force the qualitative registered RULE_ONLY outcome for that attempt, independent of how the eighth seed is resolved. The latest complete source-locked experiment confirms that classification directly, using exact original DD/VV reference archives and all new crossed-arm scores.

## 8. Scientific interpretation and next test

The AF-26 factorial intervention identifies **the original VI-2 learned-cost ordering of enabled rules** as the dominant tested source of its performance deficit on depth-five hard Horn proofs. It provides source-locked closed-loop evidence that retaining VI-2's native STOP gate while switching to Direct rule selection performs markedly better than the original VI-2 policy in this finite, synthetic gym.

The result is conditional on the frozen eight original models, the specific AF-25 Horn grammar, the original shortest+one proof-action budgets and the source-matched CPU score implementation. The repeated graph roster restricts statistical independence, and observed per-seed variation remains substantial. It does not show a universal priority ordering for neural heads or that no value estimate could improve planning.

A prospectively registered follow-up should preserve Direct's rule-ranking head, then calibrate a separate STOP decision on **fresh train/development grammar families** with both provable and unprovable cases. A second, independent question is whether learned value estimates can help rank rules when Direct logits are ambiguous. Both studies need fresh held-out graphs and a computation-matched symbolic baseline, including the previously perfect BFS64 controller.

**Research status: AF-26 registered RULE_ONLY factorial classification, independent engineering PASS in the latest source-locked run, with archived control provenance and numerical tie sensitivity disclosed.** Earlier AF-22 through AF-25 registered results remain untouched. Keep [PR #133](https://github.com/MoralityLabAI/LoopedTransformers/pull/133) **draft and unmerged** pending independent review.
