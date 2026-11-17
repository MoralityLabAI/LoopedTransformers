# AF-33: Position-balanced STOP calibration collapses terminal proof execution

**Original registered scientific outcomes, November 17, 2026:**

- **\`POSITION_BALANCED_STOP_TRANSFER_FAIL\`**, 0 of 8 fixed source seeds.
- **\`BAL_STOP_ABSOLUTE_FAIL\`**, 0 of 8 fixed source seeds.
- **\`STRICT_RANKER_SAFEGUARD_PASS\`**, 6 of 8 fixed source seeds.
- **Original source engineering and independent proof audit:** \`PASS_ALL_EIGHT_LEARNED_AND_FROZEN_DIRECT\`, all 18 jobs completed in [scored GitHub Actions #44056762277](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44056762277). Exactly **86,016 complete policy episodes** independently verified with **zero false accepted Horn proofs**.
- **Scored source commit:** \`d4b0e70ee8db5a526901896a6db8bbabb3aee2fd\`. No source graph, optimizer, model, threshold, scoring or original frozen checkpoint modification occurred after opening heldout results.

## Abstract

AF-32 prospectively passed a strict shortest-budget legal-rule-selection test with its eight-feature learned action ranker, but the learned-only STOP controller still generalized poorly across Horn premise-swap locations. AF-33 independently balances all three source swap positions in TRAIN and DEV for the existing ten-feature learned STOP head, while challenging it on **novel heldout Horn goal-conclusion arities and root-diversion premises**. The registered comparison is a fixed 600-update logistic \`BAL_STOP\` trained on all three positions against an otherwise matching \`J12_STOP\` trained only on positions 1 and 2. Both use the identical TRAIN-only eight-feature legal-action ranker for continuation.

Source-locked results show that **all eight BAL_STOP fits complete zero of 96 positive proofs at both held depths**, despite making zero incorrect *initial* positive STOP decisions. The position-ablated J12_STOP completes **96/96** positive proofs in each held cohort. Both correctly immediately STOP on only **32/96 unreachable roots per depth**. Thus BAL's registered 10-case negative STOP improvement is **zero** in every seed, and both its comparative and absolute stopping-science gates fail.

The frozen J12 model selects DEV threshold **0.90**, while BAL selects **0.70**. An independently run retrospective analysis of the original immutable trajectories confirms that BAL follows the productive proof action sequence up to its last required rule and **STOPs at the terminal pre-goal state**: at action index 4 in every depth-five reachable case, or index 5 at depth six. Across all eight seed repeats, it makes **768/768** terminal-step erroneous STOP decisions per held depth. J12 makes none and completes every proof.

The separate strict proof-selection safeguard passes 6/8 seeds. The trained RANK_CERT controller itself solves 96/96 positives at both depths for **all eight** seeds, and its exact public symbolic certificate correctly stops every unreachable root. The registered **+10/96 advantage against the strongest equal-certificate control** fails in two seeds where the untrained eight-feature scorer also solves 96/96 positive proofs. Public Horn closure and BFS64 remain exact and complete.

## 1. Protocol, data and provenance

The first prospective [AF33 PROTOCOL.json](../../experiments/activation_funnel_af33/PROTOCOL.json) was committed at [\`14893fb6\`](https://github.com/MoralityLabAI/LoopedTransformers/commit/14893fb6cb8ecffea44c345cbf0547b48e798ba3) before the AF-33 generator, model fits or heldout tests. Its original SHA-256 is \`9c2827a350560933363cc3178f0e67c5b35c5b0386add9f42ece8cb3d401f856\`.

The expanded [source-only CI #44056526933](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44056526933) passed all 12 independent structural/model-smoke tests before scoring. The source graph manifest SHA-256 is \`b55d9bba33c5a037ea9d7fcf6837829b2491d85e6aeec770649c6968c81a41c4\`. Original [frozen AF23 Torch checkpoint smoke #44056259633](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44056259633) separately passed. The official source-locked scored run [#44056762277](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44056762277) completed with green source lock, genuine original-model smoke, eight independently audited learned-policy seed jobs, eight independently audited frozen original-Direct seed jobs and the aggregate.

AF-33 has four source splits, TRAIN, DEV, HELD depth five and HELD depth six, each containing **96 matched positive/negative graph pairs** (192 encoded cases). TRAIN/DEV/each HELD split have precisely **32 pairs each at premise-swap positions 1, 2 and 3**. The two held-depth sets each balance 48 producer-supported inaccessible two-cycles and 48 previously untrained three-cycles.

Each pair differs by exactly two swapped unary Horn premises, exchanging a reachable intermediate atom 4/5/9 with the unsupported-cycle atom 10 to change goal entailment. Paired positive and negative members share root facts \`{0,1,2,3}\`, marked roles 6/13/14, the same physical atom/rule permutations, complete per-node directed degrees, initial legal rule slots and exact seven-anchor physical-ID-preserving induced directed radius-two incidence. Independent physical-bitset BFS and original logical host search certify all positive exact shortest lengths (five or six rule firings) and all negative nonderivability.

The heldout structural shift is explicit: positive GOAL-concluding clauses have **two premises** on held graphs, including a root-known side condition, versus **one premise** on TRAIN and DEV. Held root-to-12 diversions use different root-known premise pairs. These source logical templates are disjoint across all four splits and historical AF-29 through AF-32. The complete public graph remains visible to learned and nonlearned policies, while neither entailment labels, source logical IDs, matched pair IDs nor the true shortest scorer budget enter inference.

**Independence limitation:** A 96-pair cohort is a small set of logical grammar motifs, swap locations and physical relabelings. The eight optimizer seeds repeat the same 96 encoded held pairs per depth. The 768 counted seed-case evaluations are **not 768 independent novel logical structures**.

## 2. Original prospective policies and gates

The eight-feature legal-action ranker is the same public physical incidence architecture inherited from AF-32, trained on **528 canonical optimal TRAIN proof states** with 500 deterministic full-batch listwise updates at learning rate 0.1 and L2 0.01. Independent initialization seeds are **17, 29, 43, 61, 73, 89, 103, 131**. The eight original immutable AF23 6,036-weight checkpoints are separately evaluated as Native Direct, forced-continue Direct and exact-public-certificate plus Direct, without any original optimizer updates.

The two certificate-free learned STOP heads receive the same ten public graph/current-state summary features, including the ranker's action-score summary, and use the same 600-update balanced logistic BCE fit and L2 0.01. Their source support differs:

| Original fitted STOP head | TRAIN labeled feature rows | Distinct DEV root cases | TRAIN/DEV swap locations | DEV positive continuation floor |
|---|---:|---:|---|---:|
| **BAL_STOP** | 576 | 192 (96 positive, 96 negative) | 1, 2, 3 | 80/96 |
| **J12_STOP** | 384 | 128 (64 positive, 64 negative) | 1, 2 | 54/64 |

Both fit/threshold records are independently rederived by the source auditor, with all coefficients, standardization and DEV selections byte-exact. Their inference receives **no public closure or BFS certificate**; the training sample size difference is a registered caveat when interpreting the balanced-vs-ablated causal comparison.

The primary registered transfer PASS requires BAL_STOP under **strict shortest \`d\` firings**, at both held depths in the same seed, to complete >=60/96 positive proofs, correctly initially STOP >=60/96 unreachable roots, make <=10/96 positive false immediate STOPs, have zero false accepted proofs, **and correctly STOP >=10/96 more negative roots than J12_STOP**. At least six of eight original seeds must pass the complete gate on both depths. The separate absolute BAL gate drops only the ten-case relative gain. The separate strict RANK_CERT safeguard requires >=60/96 positive completed proofs, >=80/96 correct certified negative STOP and >=10/96 more positive complete proofs than the best equal-certified control. Historical positive \`d+1\` firing-budget results are retained only as separately scored sensitivity controls.

Public symbolic closure, BFS16/64 and equal-certificate PATH/ANCESTRY/FIXED/RANDOM/UNTRAINED rankers remain strong nonlearned comparisons. Negative STOP under RANK_CERT is **certified by exact public Horn fixed-point closure**, not the learned network. Its charged rule examinations are part of inference cost.

## 3. Complete strict-heldout scientific results

The table denominators count **96 graph cases × 8 independent seed fits = 768 source-case/seed episodes** per truth label and held depth.

| Strict-budget policy | Depth 5 positive complete proofs | Depth 6 positive complete proofs | D5 correct immediate negative STOP | D6 correct immediate negative STOP |
|---|---:|---:|---:|---:|
| **BAL_STOP** certificate-free learned STOP + learned ranker | **0** | **0** | **256** | **256** |
| **J12_STOP** certificate-free control STOP + same ranker | **768** | **768** | 256 | 256 |
| **RANK_CERT** learned legal rule ranker + public certificate | **768** | **768** | **768** | **768** |
| PATH_CERT | 0 | 0 | 768 | 768 |
| ANCESTRY_CERT | 0 | 0 | 768 | 768 |
| UNTRAINED_CERT | 192 | 192 | 768 | 768 |
| FIXED_CERT | 16 | 40 | 768 | 768 |
| Public CLOSURE | **768** | **768** | **768** | **768** |
| BFS16 | 0 | 0 | 768 | 768 |
| BFS64 | **768** | **768** | **768** | **768** |
| Original AF23 Direct native | 0 | 0 | 480 | 480 |
| Original AF23 Direct forced continuation | 0 | 0 | 0 | 0 |
| Original AF23 Direct + public negative certificate | 0 | 0 | **768** | **768** |

All exact public host proof executions across every arm and seed are sound; the independent audit found **zero false accepted proofs**. The original Direct model completions here are a genuine same-cohort result on the novel AF-33 held motif, rather than a cross-experiment comparison.

All eight seed-specific STOP heads produce the same registered complete proof and negative initial STOP counts:

| Source seed | BAL_STOP D5/D6 proof completions /96 | J12_STOP D5/D6 proofs /96 | BAL vs J12 negative STOP advantage /96 | BAL transfer gate | BAL absolute gate |
|---:|---|---|---:|---|---|
| 17 | 0 / 0 | 96 / 96 | 0 / 0 | FAIL | FAIL |
| 29 | 0 / 0 | 96 / 96 | 0 / 0 | FAIL | FAIL |
| 43 | 0 / 0 | 96 / 96 | 0 / 0 | FAIL | FAIL |
| 61 | 0 / 0 | 96 / 96 | 0 / 0 | FAIL | FAIL |
| 73 | 0 / 0 | 96 / 96 | 0 / 0 | FAIL | FAIL |
| 89 | 0 / 0 | 96 / 96 | 0 / 0 | FAIL | FAIL |
| 103 | 0 / 0 | 96 / 96 | 0 / 0 | FAIL | FAIL |
| 131 | 0 / 0 | 96 / 96 | 0 / 0 | FAIL | FAIL |

The J12 control is trained on fewer unique position-supported examples yet preserves 100% positive proof execution. Both policies' correct held negative initial STOP decisions remain **32/96 per depth**, all at position 1. Positions 2 and 3 have **0/32** correct negative immediate STOP in every source seed. The held negative stopping shortfall therefore persists after BAL sees all three positions in both TRAIN and DEV.

## 4. Temporal failure: late STOP before the goal

The original primary protocol limits false immediate STOP at *positive roots*, while requiring substantial **absolute complete proof execution**. Those two conditions separate dramatically in AF-33: BAL_STOP makes **zero incorrect initial positive STOP decisions**, and its shared learned legal rule ranker follows the productive legal sequence. It then stops at the last decision before the goal.

The source-independent [retrospective replay #44057873549](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44057873549) reexamined all eight SHA-pinned original independently audited learned-policy episode files without rerunning or fitting any model. It confirmed **96/96** BAL premature STOPs at action index **4** in every depth-five positive cohort, and **96/96** at action index **5** in every depth-six positive cohort, for all eight independent seeds. J12 produces no premature STOP in its positive trajectories, completing all source shortest proofs. These temporal facts did not enter the original trained threshold selection or science criteria.

### Fixed DEV thresholds and source-trajectory probabilities

All eight original BAL fits choose frozen DEV threshold **0.70**, compared with **0.90** for J12. The original DEV-only selected threshold records show BAL preserves 96/96 DEV positive *root* continuation with **32/96 negative initial STOP**; J12 preserves 64/64 DEV positive root continuation with **32/64 negative initial STOP**.

On seed 17, whose rounded per-step probabilities and source-only functions reproduce across physical permutations, the productive held positive trajectory shows:

| Pre-goal action decision | BAL_STOP depth 5 | J12_STOP depth 5 | BAL_STOP depth 6 | J12_STOP depth 6 |
|---|---:|---:|---:|---:|
| Root (index 0) | 0.550 | 0.500 | 0.370 | 0.089 |
| Last decision before GOAL | **0.863** | 0.860 | **0.861** | 0.754 |
| DEV-selected STOP threshold | **0.700** | 0.900 | **0.700** | 0.900 |
| Completed held reachable proofs /96 | **0** | **96** | **0** | **96** |

The source STOP supervised feature rows are generated from root states and at most two deterministic first-legal successors (\`t=0,1,2\`), while DEV threshold selection checks **roots only**. Consequently, the final pre-goal states at action index 4/5 are outside the supervised/threshold constraints. This makes the observed late STOP collapse an explicit *closed-loop extrapolation problem*, even though no held positive root is falsely stopped.

The newly balanced TRAIN/DEV premise positions alter the fitted numeric threshold. This numerical mechanism is directly visible in saved decisions; BAL and J12 also have different fitted coefficients and unequal TRAIN sample counts, so the study does **not** isolate threshold value alone as the independent causal cause.

## 5. Matched-pair ranking and probability metrics

The preregistered scorer computed root paired AUC, ten-bin expected calibration error, Brier score and binary log loss from the **original sealed held root probabilities**. Results repeat across the eight fixed ranker initialization seeds for these small logical motif families:

| Held root metric | BAL D5 | J12 D5 | BAL D6 | J12 D6 |
|---|---:|---:|---:|---:|
| Paired root STOP AUC | 0.500 | 0.500 | 0.833 | 0.833 |
| Brier score | 0.2644 | 0.3064 | 0.1817 | 0.1853 |
| Binary log loss | 0.7198 | 0.8812 | 0.5517 | 0.5810 |
| Ten-bin ECE | 0.3520 | 0.3334 | 0.2021 | 0.2056 |

BAL improves root Brier/log loss relative to J12 at both depths, yet destroys complete positive proof execution. In these source families, better static root probability quality is therefore insufficient to produce better on-policy full-episode behavior. The paired AUCs remain identical between BAL and J12: a position-balanced training mix failed to improve the registered within-pair ordering.

All original source feature and scoring artefacts were frozen before heldout analysis. The source-only temporal replay is recorded separately and cannot substitute new thresholds or repair the registered scientific failure.

## 6. Independent proof and inference cost audit

All **18 jobs** in the original [successful scored run #44056762277](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44056762277) completed: source lock, immutable original-checkpoint smoke, eight learned-ranker/BAL/J12 original fit and heldout host-proof audits, eight original AF23 native/forced/cert physical proof and genuine Torch-forward audits, and final aggregate.

Every learned seed generated **8,448 complete episodes** across four strict/slack source cohorts and eleven registered learned/symbolic arms. Every original frozen Direct seed generated **2,304** episodes across the same four cohorts and three Direct arms. Combined total: **86,016 complete independently audited source episodes**. Original frozen neural checkpoint optimizer updates: zero.

Counted public Horn rule examinations under the strict budget, pooled over all eight seeds:

| Controller | Held depth 5 | Held depth 6 |
|---|---:|---:|
| BAL_STOP (early terminal STOP) | 1,275,344 | 1,557,024 |
| J12_STOP (completes proofs) | 1,381,072 | 1,669,136 |
| RANK_CERT with public certificate | 625,408 | 780,192 |
| PATH_CERT | 474,880 | 586,656 |
| Exact public closure | **241,024** | **297,584** |

BAL makes fewer examinations than J12 partly because it **stops prematurely**. Therefore its lower work count is evidence of incomplete execution, not an inference efficiency advantage. Public closure remains a complete deterministic solver with approximately **2.6× fewer counted rule examinations** than RANK_CERT on these very small Horn graphs. Original frozen neural model-node forward counts and physical public rule examinations are separate units, not equal FLOPs or calibrated runtime.

### Strict ranker safeguard nuance

Although the source-trained RANK_CERT action ranker completes 96/96 positive strict proofs at both depths in every seed, the registered requirement of **at least ten additional proofs over the strongest same-certificate control** passes only **six of eight seeds**. In original seeds **17** and **89**, the corresponding seeded *untrained* linear action ranker also completes 96/96 positives at each depth, leaving the trained-vs-best-control margin at zero. All other six seeds meet the safeguard. The official aggregate therefore correctly records **STRICT_RANKER_SAFEGUARD_PASS (6/8)**, with 100% absolute trained completion interpreted separately.

## 7. Evidence, chronology, and next discriminating experiment

The original official aggregate, \`af33-position-balanced-stop-aggregate\`, GitHub Actions artifact ID **14316861374**, has ZIP SHA-256 \`9028bbedba29cf67041e4acafdc5d9af211518eb37cf380876ed144fc2e018ac\`; the member \`af33_results.json\` SHA-256 is \`23637a235ebf62b08343229ecee2ef13fbc9f421a34a27d3f661194e7ab3e6c7\`.

The separate immutable [temporal STOP evidence replay #44057873549](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44057873549) passed. Its \`af33_temporal_stop_posthoc.json\` SHA-256 is \`59294d94857915e25d7a83057c4dc9b34946f1d1cd3f06f77fc5834d8608d63e\`. All **16 original scored seed archive ZIP SHA-256** digests, the original registered protocol and graph source manifests, and the retrospective-only evidence are committed in [AF33_EVIDENCE_MANIFEST.json](AF33_EVIDENCE_MANIFEST.json). The science source and all original score decisions remain unchanged; GitHub artifact bytes are subject to normal retention.

A follow-on **AF-34 prospective on-policy prefix STOP calibration** should explicitly supervise *all positive canonical proof-prefix states*, including the last pre-goal decision, while maintaining a separate root-negative STOP loss. The DEV threshold should be selected under a **full positive proof-completion floor**, rather than a root-only continuation floor. It should compare against an equal-budget last-step-aware public diagnostic, J12 and root-only baseline, and retain exact closure/BFS controls. New held motifs must be frozen before optimization. The strict proof-selection safeguard remains independent of STOP.

The source mathematical [position-mixture threshold note](AF33_POSITION_CALIBRATION_THEOREM.md) gives an exact conditional domain-shift bound for frozen thresholds and reiterates the local radius-two observer's matched-AUC 0.5 ceiling. Position exposure alone supplies no distribution-free guarantee of semantic STOP transfer.

**AF-33 PR #146 remains draft and unmerged. All original AF-22 through AF-32 scientific verdicts and provenance are preserved.**
