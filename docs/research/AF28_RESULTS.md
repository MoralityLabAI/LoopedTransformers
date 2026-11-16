# AF-28: Prospective STOP calibration improves proof continuation but fails negative-goal termination transfer

**Registered scientific verdict: CALIBRATED_STOP_TRANSFER_FAIL (0/8 seeds).**  
**Hosted engineering and independent proof audit: PASS (19/19 CI jobs).**  
**Result date: November 17, 2026.**

## Abstract

AF-28 tests whether a small, prospectively fitted STOP classifier can improve a frozen recurrent Horn-rule policy on grammars absent from its calibration data. Eight original AF-23 checkpoints, each containing 6,036 parameters, supplied unchanged Direct rule rankings and frozen model features. A seven-parameter logistic gate was fitted on separate training grammars. A development-only threshold was then fixed before testing two unseen proof-depth cohorts. A one-feature, equally trained logistic controller served as a capacity control, alongside the original Direct and VI-2 STOP gates and public graph-aware symbolic algorithms.

The preregistered two-depth criterion **failed in all eight source-verified seeds**. The calibrated policy, CAL, completed **280/640** provable depth-five proofs compared with **160/640** for Direct, and **120/640** at depth six compared with **40/640**. It also reduced correct immediate STOP on unprovable goals from **616/896** to **364/896** at depth five, and from **672/896** to **406/896** at depth six. Correctly identifying an unreachable root is the central metric the added classifier needed to improve, and no seed passed both held-out depth gates.

CAL improves initial-root probabilistic Brier and log-loss relative to the equal-budget one-feature THR controller, yet the gains do not restore the original Direct gate's negative STOP decisions. All public-closure and BFS16 controls solve every provable case and correctly STOP on every unprovable case, limiting neural proof-search capability claims on these finite synthetic Horn programs.

The official hosted verification covered **27,648 complete proof episodes**, **231,264** model-node evaluations, **160** sampled original checkpoint forward replays and **5,696** independently recomputed train/dev feature rows. Every accepted Horn proof was sound.

## 1. Source registration and immutable evidence

- **Original protocol:** [AF-28 PROTOCOL.json](../../experiments/activation_funnel_af28/PROTOCOL.json) registered at [commit `45c1544`](https://github.com/MoralityLabAI/LoopedTransformers/commit/45c15443f5b7847b6414b0bc6caf63c1ef6ca3ec), before source generator construction, STOP fitting, or new model scoring. Four dated **source-only pre-fitting** amendments correct negative-root proof reachability, root action availability and the exact DEV threshold/one-feature control specification. Frozen protocol SHA-256: `96e050f40a8f60b1e9b97d06df637dbebf4d8838a55cb4dc3437f5aee3cda143`.
- **Grammar source-only proof preflight:** [CI #43914760192](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43914760192), successful before any original checkpoint fit or held-out scoring. Manifest SHA-256 `34c30216b8407fcf9339e7c909512e5011e9375102ef0451941868574b39c36d`. It checks public case hash disjointness, canonical source-rule incidence separation, independent physical bitset BFS and closure truth, every positive shortest proof length, and an actionable legal rule at every root.
- **Original neural checkpoint source:** [AF-23 Track B run #43812708916](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43812708916), eight immutable final 6,036-parameter TypedLoop checkpoints, one per original seed (17,29,43,61,73,89,103,131). Source ZIP/member SHA-256, original train receipt and Torch/NumPy weight parity checked. **Original network training updates: zero**.
- **Official final hosted experiment:** [GitHub Actions #43981312819](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43981312819), successful, **19/19 jobs**, scored source-lock commit `092ee4b49ffaf8e3427e3c029dbfeef63cf04554`. The source lock, frozen-model smoke, eight distinct train/dev-only calibrator jobs, eight independent held-out score/audit jobs and the final aggregation passed.
- **Original official aggregate:** `af28-calibrated-stop-aggregate`, artifact ID **14283278350**, artifact ZIP SHA-256 `273b565742561abf4ac692cfc89cde76b30434f0289de05b9d8ffac90f0a2525`. Contained `af28_results.json` SHA-256 `b82f7828884a356b28b25ee1b9560a2d78fa7e64a01c0cbd5625c216dac3c5ad`.
- **Artifact provenance:** [AF28_EVIDENCE_MANIFEST.json](AF28_EVIDENCE_MANIFEST.json) pins the eight original per-seed scored ZIPs, eight train/dev-only calibrator ZIPs, the final aggregate and original source identities. GitHub Actions artifacts are retention-limited, so the stable report and digest manifest remain committed separately.

The original AF-22 through AF-27 protocols, science verdicts and frozen model weights are preserved.

## 2. Prospective data separation

| Split | Logical grammar families | Physical layouts | Provable / total | Shortest positive proof |
|---|---|---|---:|---|
| Train | `train_shared_gate`, `train_dual_or` | 610–613 | 80/192 | 4 or 5 firings |
| Development | `dev_crossed_and`, `dev_short_or` | 620–623 | 80/192 | 4 or 5 firings |
| Held depth 5 | `held_forked_and`, `held_or_gate` | 630–633 | 80/192 | Exactly 5 firings |
| Held depth 6 | `held_forked_and`, `held_or_gate`, depth-specific topology | 640–643 | 80/192 | Exactly 6 firings |

Each split has **112 unprovable goals**. Its two source families contribute 40 positive and 56 negative cases apiece. Every input starts with **four known facts**, and all goal-concluding routes require a critical root fact 2. Negative examples replace that fact with a separately source-registered, already derived decoy atom. Independent bitset BFS certifies each negative case as unreachable.

Train/dev and held-out cases have different **logical rule-incidence signatures**, beyond physical slot permutations. The generated negative-root fact substitution, however, retains a related missing-critical-input motif across every split. That recurring motif may permit correlations unrelated to general-purpose logical reachability, and should be treated as a limitation of transfer claims.

### Fitting and controls

At every training state, the original frozen recurrent model supplies six features:

1. Direct STOP-minus-best-legal-rule score margin.
2. Frozen root reachability logit, scaled by four.
3. Frozen root distance logit, scaled by four.
4. VI-2 STOP-minus-best-legal-rule margin, scaled by eight.
5. Number of currently legal rules, centered at four and scaled.
6. Current known-fact count, centered at four and scaled.

Features are clipped to `[-8,8]` and standardized using **train only**. The per-seed CAL gate has six trainable slopes and a bias, optimized by exactly **600 full-batch updates** of class-balanced logistic cross entropy plus **0.01 L2** slope penalty. Source training examples include each train root and up to two deterministic legal-rule successor states, selected without looking at model predictions. Original model weights, its Direct rule ranking and its VI-2 head remain frozen.

The **THR control** fits a two-parameter logistic on the Direct STOP margin alone, with exactly the same source data, optimizer budget and regularization. Both choose a threshold independently from the **19-point grid 0.05–0.95**, using development roots only. The chosen threshold maximizes correct DEV negative STOP while continuing on at least 72/80 provable DEV roots, with registered tie breaking. The final calibration coefficients and thresholds are committed as source-bound CI artifacts before held-out scoring begins.

## 3. Registered primary gate and official outcome

The primary target is CAL compared with original DD, using the **same frozen Direct legal-rule ranker**. For each of the eight checkpoint seeds, CAL must meet every requirement at *both* held-out depths:

- Correct immediate STOP on at least **7 additional unprovable cases out of 112** relative to DD.
- Complete at least **DD's provable proof count minus four** out of 80.
- Accept zero false physical Horn proofs.

At least **six of eight** original source-verified checkpoint seeds must pass both depths. All eight qualified. **Zero passed.**

### Full pooled proof completion and correct termination

The numerators repeat **80 unique positive graphs** and **112 unique negative graphs** per depth across the **eight original model initializations**. These are paired seeded outcomes rather than 640 or 896 independent synthetic tasks.

| Policy | Depth 5 proofs /640 | Depth 6 proofs /640 | D5 correct immediate STOP on negative /896 | D6 correct immediate STOP on negative /896 |
|---|---:|---:|---:|---:|
| **DD (original Direct)** | 160 (25.00%) | 40 (6.25%) | **616 (68.75%)** | **672 (75.00%)** |
| VD (VI-2 gate, Direct rules) | 160 (25.00%) | 80 (12.50%) | 546 (60.94%) | 672 (75.00%) |
| **CAL (six-feature logistic)** | **280 (43.75%)** | **120 (18.75%)** | 364 (40.63%) | 406 (45.31%) |
| THR (one-feature logistic) | 240 (37.50%) | 120 (18.75%) | 336 (37.50%) | 392 (43.75%) |
| ND (forced continuation) | 320 (50.00%) | 160 (25.00%) | 0 | 0 |
| Public Horn closure | **640 (100%)** | **640 (100%)** | **896 (100%)** | **896 (100%)** |
| BFS16 and BFS64 | **640 (100%)** | **640 (100%)** | **896 (100%)** | **896 (100%)** |
| Exact shortest oracle | 640 | 640 | 896 | 896 |

CAL gains **120** positive completions over DD at depth five and **80** at depth six, without losing any previously completed DD proof in the paired comparison. Correct negative STOP moves in the opposite direction: **−252** at depth five and **−266** at depth six. The research question concerns correctly stopping on unreachable roots while preserving proof completion, and this two-depth registered criterion is conclusively failed.

### Exact paired differences

| Pair | Depth 5: first-only wins | Depth 5: second-only wins | Depth 6: first-only wins | Depth 6: second-only wins |
|---|---:|---:|---:|---:|
| CAL versus DD, provable proofs | **120** | 0 | **80** | 0 |
| CAL versus DD, correct negative STOP | 70 | **322** | 42 | **308** |
| CAL versus THR, correct negative STOP | **98** | 70 | **112** | 98 |

Both CAL and THR continue on far more provable roots than the original native gates. That improves proof completion for some source checkpoints while sacrificing calibrated termination on negative roots.

## 4. Every original seed

The table preserves the preregistered per-seed unit. Each proof completion value is out of 80; negative correct immediate STOP counts are out of 112 per depth.

| Seed | D5 DD proofs | D5 CAL proofs | D5 DD negative STOP | D5 CAL negative STOP | D6 DD proofs | D6 CAL proofs | D6 DD negative STOP | D6 CAL negative STOP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 40 | 40 | 14 | 28 | 0 | 0 | 56 | 28 |
| 29 | 40 | 40 | 112 | 42 | 0 | 40 | 112 | 56 |
| 43 | 0 | 40 | 70 | 56 | 0 | 0 | 56 | 28 |
| 61 | 0 | 0 | 56 | 14 | 0 | 40 | 56 | 28 |
| 73 | 40 | 40 | 112 | 98 | 0 | 0 | 112 | 98 |
| 89 | 40 | 40 | 42 | 28 | 40 | 40 | 56 | 42 |
| 103 | 0 | 40 | 112 | 14 | 0 | 0 | 112 | 28 |
| 131 | 0 | 40 | 98 | 84 | 0 | 0 | 112 | 98 |
| **Pooled** | **160** | **280** | **616** | **364** | **40** | **120** | **672** | **406** |

Seed 17 passes the **single depth-five** condition, gaining 14 correct negative STOP decisions while preserving 40/80 provable proof completions. It fails depth six. The remaining seeds fail at least one primary requirement at depth five, and all eight fail the complete registered two-depth gate.

### Held-out grammar family breakdown

Every grammar family has **40 unique positive and 56 unique negative cases** per depth, repeated across eight checkpoints.

| Family | D5 DD solved /320 | D5 CAL solved /320 | D6 DD solved /320 | D6 CAL solved /320 |
|---|---:|---:|---:|---:|
| `held_forked_and` | 160 | **280** | 40 | **120** |
| `held_or_gate` | 0 | **0** | 0 | **0** |

The observed positive completion gains come entirely from the forked-AND family. All neural variants, including never-STOP, fail every positive OR-gate test in both held-out depths. Public closure and BFS16 complete both families perfectly. The fixed Direct rule head consequently remains a severe source of difficulty under the OR-gate composition.

## 5. Equal-budget one-feature control and probabilistic calibration

THR uses the same training samples, 600-update logistic fit, class weighting, regularization and dev-only threshold grid. It controls for simply re-scaling a single Direct STOP margin before choosing a new threshold.

### Held-out probability metrics, initial roots

| Metric | D5 CAL | D5 THR | D6 CAL | D6 THR |
|---|---:|---:|---:|---:|
| Brier score, lower is better | **0.22573** | 0.28550 | **0.23285** | 0.27807 |
| Binary log loss, lower is better | **0.81139** | 0.87174 | **0.85723** | 0.86414 |
| Ten-bin ECE, lower is better | **0.18935** | 0.24080 | **0.21543** | 0.24787 |

CAL attains lower Brier, log loss and ten-bin ECE than the one-feature THR control on both depth cohorts. At the **dev-selected operating threshold**, CAL completes **40 more provable depth-five proofs** and earns 28 additional negative STOP decisions at depth five and 14 additional at depth six, pooled over eight checkpoints. Those are specific benefits from the prospectively fitted multivariate features on this fixed test roster.

The substantial negative STOP deficit relative to DD remains. Better continuous probability scores do not establish correct policy behavior at a fixed threshold, especially when source templates change between train/dev and held-out evaluation. A threshold selected to protect DEV provable-root continuation can shift far toward continuation on new negative cases.

The original DD/VD source margins are logits and negative value scores, respectively. They are **not probabilities**, so Brier or probability ECE for those native margins would be meaningless without an additional preregistered calibration map. Only CAL and THR receive these probabilistic metrics.

## 6. STOP error location and proof-search limitation

| Metric | D5 DD /640 | D5 CAL /640 | D6 DD /640 | D6 CAL /640 |
|---|---:|---:|---:|---:|
| Immediate incorrect STOP on provable goals | 400 | **0** | 400 | **80** |
| Any premature STOP on provable goals | 440 | **80** | 600 | **200** |
| Completed provable proofs | 160 | **280** | 40 | **120** |

CAL reduces erroneous positive stopping, including some later premature STOP decisions, but legal-rule ordering remains fixed to Direct. Even with **never STOP**, that ranking completes only **320/640** depth-five and **160/640** depth-six proofs. Its inability to solve held-out OR-gate proofs demonstrates that better STOP calibration alone cannot solve the underlying action-ranking generalization problem.

**Every public Horn closure and BFS16 control solves all 80 unique provable cases and correctly stops on all 112 unreachable cases at each depth.** AF-28 therefore supplies no neural proof-search advantage over inexpensive full-public-graph algorithms. Its negative scientific result concerns the policy-level tradeoff between proof continuation and STOP calibration.

## 7. Independent hosted verification and numerical reproducibility

The successful [source-locked CI #43981312819](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43981312819) completed **19/19 jobs**: source lock, frozen-model/calibration smoke, eight independent source-verified train/dev logistic fits, eight held-out proof/evidence auditors, and final eight-seed aggregation.

| Verification item | Official count |
|---|---:|
| Eligible original frozen checkpoint seeds | **8/8** |
| Complete independently audited held-out proof episodes | **27,648** |
| Reconstructed original model node evaluations | **231,264** |
| Sampled true original checkpoint forward replays | **160** |
| Independently reproduced TRAIN/DEV frozen-model feature rows | **5,696** |
| Feature rows with numerical difference above \(10^{-5}\) | **42** |
| Maximum source-feature numerical difference | **\(1.3113\times10^{-5}\)** |
| Maximum independently refitted logistic coefficient difference | **\(3.7641\times10^{-6}\)** |
| DEV-selected threshold changes | **0** |
| DEV root STOP-decision changes | **0** |
| False accepted host-certified Horn proofs | **0** |
| Updates to original recurrent network weights | **0** |

Every generated model-visible physical rule was independently reconstructed from its encoded tensor incidence. Two separate BFS implementations agree with all original source truth labels and shortest proof budgets; every chosen action, rule-firing proof, STOP decision, logistic feature/probability and per-arm count was rechecked. A source-bound checkpoint forward replay samples actual model logits. The trainer and auditor separately recompute fitted logistic coefficients and dev-only thresholds.

### Source-only integrity amendments and blocked runs

The source generator underwent four **dated pre-fitting** amendments. They repaired missing critical-fact guards in negative goal routes, introduced actionable legal root moves for negative cases, corrected a depth-six pre-derived productive atom that made a negative case provable, and froze the exact scalar logistic/development-threshold control. Each failure was identified by a source-only independent BFS or root-legal test before held-out model evaluation.

The first fully trained [CI #43915083864](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43915083864) passed eight calibrator fits and seven of eight held-out independent audits, while seed 43 failed a strict \(10^{-5}\) source-feature replay. A [second source-identical retry of CI #43916143426](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43916143426) hit a similarly borderline source-feature replay in seed 43. The separately [hosted train/dev-only diagnostic #43980157757](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43980157757) independently reproduced all 520 seed-43 training and 192 development features **exactly** in a different CPU runner, including unchanged probabilities and dev classifications.

The successful final CI commits an **engineering-only numerical verification amendment**. It retains byte-exact saved train/dev feature archives and original model SHA identities while independently generating all features again in their original 32-example Torch batches. It requires an absolute numerical source-feature difference no greater than \(10^{-4}\), independently refits **both** logistic models, requires their coefficient differences no greater than \(10^{-4}\), and rejects any change to the previously selected dev threshold or any DEV root STOP decision. The final measured maxima were an order of magnitude smaller than that acceptance bound; all dev STOP choices remained identical.

These amendments **did not change the registered scientific gate, neural rule-ranking implementation, held-out case outcomes, original checkpoint weights, train/dev labels or 600-update logistic objective**. The earlier failed engineering receipts remain available in GitHub Actions. The source-verification allowance is disclosed as a numerical replay bound, not a tuned inference threshold.

## 8. Interpretation and next discriminating test

Prospective multifeature logistic fitting improves root probability quality relative to a matched one-feature alternative, and it helps some frozen Direct-ranked neural controllers continue valid proofs. The central challenge was obtaining more **correct STOP decisions on unreachable goals** while retaining proof completion. CAL loses hundreds of correct negative STOP decisions at each tested depth, and the registered transfer criterion fails 0/8 original checkpoint seeds.

The recurring missing-critical-fact structure of negative cases and perfect public Horn closure performance further limit the interpretation. A future experiment should examine STOP decisions on structurally paired reachable/unreachable roots with indistinguishable local public counts, conditional on genuinely new goal-derivation motifs, and should select thresholds against an explicit dev-set frontier between provable continuation and correct negative STOP.

A separate recurrent action-ranking study is also required. The fact that all original neural controllers fail the OR-goal family while public closure solves it completely suggests that additional stopping calibration alone will reach a ceiling under fixed Direct legal-rule scores. A source-registered test of representation or planning changes must report competitive public algorithms and their actual compute cost.

**Official research status: AF-28 CALIBRATED_STOP_TRANSFER_FAIL; independent evidence/engineering PASS.** Earlier AF-22 through AF-27 registered scientific outcomes remain unchanged. The [AF-28 draft PR #136](https://github.com/MoralityLabAI/LoopedTransformers/pull/136) remains unmerged for independent review.
