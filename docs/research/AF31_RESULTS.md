# AF-31: Producer-supported cycles expose a STOP failure; legal proof ranking matches a public heuristic

**Official preregistered verdicts (November 17, 2026):**

- **Rule-selection transfer:** \`CYCLIC_RULE_SELECTION_FAIL\`, 0/8 independently trained source seeds.
- **Certificate-free learned STOP:** \`LEARNED_STOP_TRANSFER_FAIL\`, 0/8 seeds.
- **Science inputs / independent audits:** **PASS** in all eight TRAIN-only ranker/STOP jobs and all eight original AF-23 frozen Direct jobs. **39,936** complete source-proof episodes independently audited; zero false accepted Horn proofs.
- **Scored source:** \`18fd19b06a67edf6c335b2ad991f005d9301d022\`, [original source-locked CI #44033106880](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44033106880).
- **Successful aggregate from those same immutable original receipts:** [CI replay #44033813293](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44033813293), artifact \`af31-cyclic-rule-stop-official-original-replay\` ID **14305504295**. Its independent evidence checks all PASS; aggregation-only engineering failures and fixes are documented below.

## Abstract

AF-30 showed that a trained eight-feature legal-action ranker could execute every held-out Horn proof, although an equally certified blocked-premise PATH heuristic did the same. In AF-31 the distracting goal corridors contain **inaccessible yet producer-supported two- or three-node Horn cycles**. Exactly matched positive/negative graphs are constructed with the original seven-anchor physical-ID-preserving directed radius-two neighborhoods equal within each pair.

The original 6,036-weight AF-23 frozen Direct checkpoint is finally compared **on these exact same source graphs**, in addition to the source-trained linear ranker and a separate train-only learned STOP logistic. All eight learned rule rankers complete **96/96** reachable proofs at both held depths, while the exact-certifier + original frozen Direct ranker completes only **144/768** at depth five and **80/768** at depth six across eight original checkpoints.

However, the hand-coded public PATH control also completes **768/768 at each depth**. The frozen registration demands at least **ten additional completed proofs per 96** over the strongest same-certificate control. With PATH perfect, the gain is zero for every learned seed. The primary verdict is therefore **CYCLIC_RULE_SELECTION_FAIL**.

The separately trained public-graph logistic \`LRANK_STOP\` retains perfect positive proof execution by always continuing on the held roots, but yields **zero correct negative initial STOP decisions (0/768 at both depths)**. This fails its independent stopping gate. All eight source-fitted DEV thresholds equal \`0.95\`, with DEV 96/96 positive continuation and 0/96 negative STOP. A retrospectively computed paired root STOP score AUC is 0.50 at held depth five and 0.833 at held depth six in every seed; this score ordering is descriptive and **not** a preregistered passing endpoint.

## 1. Source registration and exact construction

The protocol was first committed at [\`b62ee7b\`](https://github.com/MoralityLabAI/LoopedTransformers/commit/b62ee7b1760b4898ed0ae5458fc47b897e436248), before AF-31's new generator and model fitting. Its original SHA-256 is \`50033725ae6c6d1a2e7467c7a12d5f0bb16596f67b44858e77906234106770eb\`. The final generator manifest SHA-256 is \`10e287c5d0d2363ba46d8570788de5fba14b49b6b0d4fbbad1fe8a210bd1440a\`. Source CI [#44032880401](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44032880401) passed before the official scoring. Separate [frozen-model source smoke #44032926377](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44032926377) passed real Torch-forward replay of the original checkpoint.

Each of TRAIN, DEV, HELD depth5 and HELD depth6 contains 96 positive/negative pairs, from two source motifs with 48 pairs each. Every pair has exactly the same initial facts \`{0,1,2,3}\`, marked atoms 6/13/14, physical atom/rule permutations, initial legal-rule mask, complete physical node in/out-degree profile, and directed identity-preserving induced radius-two incidence around all seven root/role anchors. Two middle Horn premises swap between a source-derivable atom 4/5/9 and blocked atom 10. Independent physical bitset BFS and host BFS verify exact opposite entailment labels.

TRAIN/DEV contain producer-supported 2-cycles \`8→10→8\` and swaps at positions 1/2. Each held set has 48 two-cycles and 48 unseen three-cycles \`8→10→13→8\`, with positions 1/2/3 balanced at 32 pairs each. Every noninitial logical atom 4..14 has a producer, and the unreachable SCC has no justified entry from root-known atoms. This is certified independently by [the source-only structural tests](../../experiments/activation_funnel_af31/tests/test_structure.py). The companion [cyclic trapping theorem](AF31_SUPPORTED_CYCLE_THEOREM.md) proves the closure invariant.

The 96 pairs per held depth represent a small collection of motifs and position shifts under unique physical permutations, **not 96 structurally independent graph families**. The eight trained seeds see the same encoded held cases; denominators of 768 refer to the 96 physical graph cases × eight initializations.

## 2. Policies and preregistered science gates

The eight-feature ranker trains for 500 full-batch listwise softmax updates with L2 0.01 and fixed learning rate 0.1. Training examples are 528 source-verified TRAIN-only intermediate proof states. No original recurrent checkpoint is optimized. The learned \`RANK_CERT\` action ranker uses an **exact once-per-root public Horn negative STOP certificate**, charged with its rule examinations. Equivalent negative certificates wrap the untrained linear, PATH, ancestry, fixed and random rankers; public closure and BFS16/BFS64 are independent controls.

The primary scientific PASS requires, at each depth, at least 60/96 completed reachable proofs, 80/96 correct negative immediate STOP, ten more proofs than the strongest certificate-matched control, and zero false accepted proofs, for at least six of eight seeds simultaneously at both held depths.

The separate ten-feature \`LRANK_STOP\` is a train-only, 600-step balanced BCE logistic classifier. Its inference input includes the current public graph, action-score summaries and physical known-state bits. Threshold selection uses DEV only, with at least 80/96 positive DEV roots continuing; the fitted 0.95 threshold is frozen before heldout. LRANK_STOP has **no symbolic closure/BFS access at inference**. Its separate PASS requires 60/96 completed positive proofs, 60/96 correct negative STOP and at most ten positive initial false STOPs at both held depths, in six of eight seeds.

The historical controls use all eight exact original AF-23 6,036-weight checkpoints, unchanged: Native Direct (its own STOP and legal-rule selection), forced Direct continuation, and original Direct legal ranking with the **same public negative certificate** as RANK_CERT/PATH. Every original frozen prediction is on the identical AF-31 held source graph and action budget. Independent auditors replay every physical rule application, stored original logits and a stratified sample of genuine Torch forwards.

## 3. Complete source-verified heldout results

Every cell has 768 source-case × seed occurrences, with 96 unique reachable and 96 unique unreachable cases at each held depth.

| Policy | Depth 5 positive proofs | Depth 6 positive proofs | D5 correct immediate negative STOP | D6 correct immediate negative STOP |
|---|---:|---:|---:|---:|
| **RANK_CERT** trained legal ranking + public STOP | **768** | **768** | **768** | **768** |
| **PATH_CERT** reverse path heuristic + same STOP | **768** | **768** | **768** | **768** |
| **ANCESTRY_CERT** unconditional reverse relevance | **768** | **768** | **768** | **768** |
| UNTRAINED_CERT eight-feature randomized linear | 480 | 480 | 768 | 768 |
| FIXED_CERT first legal action | 184 | 136 | 768 | 768 |
| RANDOM_CERT deterministic seeded random legal | 86 | 34 | 768 | 768 |
| **LRANK_STOP** learned legal ranking + learned STOP | **768** | **768** | **0** | **0** |
| AF23_DIRECT_NATIVE | 64 | 0 | 512 | 528 |
| AF23_DIRECT_FORCED | 144 | 80 | 0 | 0 |
| AF23_DIRECT_CERT | 144 | 80 | **768** | **768** |
| Public CLOSURE | **768** | **768** | **768** | **768** |
| BFS16 | **768** | **768** | **768** | **768** |
| BFS64 | **768** | **768** | **768** | **768** |

All rows have **zero false accepted proofs**. The complete exact source rule trace and registered scorer budget are independently replayed.

### Seed-level original Direct comparison

| Original AF23 seed | Native D5/D6 completed | Forced D5/D6 completed | Native D5/D6 correct negative initial STOP |
|---:|---|---|---|
| 17 | 48 / 0 | 48 / 48 | 0 / 0 |
| 29 | 0 / 0 | 0 / 0 | 96 / 96 |
| 43 | 0 / 0 | 32 / 32 | 96 / 96 |
| 61 | 16 / 0 | 16 / 0 | 0 / 0 |
| 73 | 0 / 0 | 0 / 0 | 48 / 48 |
| 89 | 0 / 0 | 16 / 0 | 96 / 96 |
| 103 | 0 / 0 | 16 / 0 | 96 / 96 |
| 131 | 0 / 0 | 16 / 0 | 80 / 96 |

All eight source-trained RANK_CERT seeds solve 96/96 at both depths; all eight PATH controls do likewise on exactly those cases. **Ranker gain over PATH is zero in every seed**. The same-cohort frozen Direct comparison confirms substantial proof recovery by the new trained action ranker. It does not establish that the new ranker is better than a well-chosen symbolic heuristic.

## 4. Why the stronger cycle did not separate PATH

Every AF-31 unsupported cycle has at least one producer per atom, defeating the **specific producerless-premise test** of AF-30. The reverse goal path scorer still sees an attractive spurious route that typically causes only one irrelevant root action, which the registered **shortest-plus-one** budget permits without failing proof completion. Once that action is used, the productive chain remains favored and can still finish.

Consequently **even unconditional reverse ancestry** completes every AF-31 held proof, as does the more nuanced PATH heuristic. The cycle construction establishes a genuine logical obstruction to reverse reachability as a sound certificate, while the *action budget and number of competing root decisions* make the current proof benchmark easy for these heuristics.

The learned-only STOP result exhibits a distinct failure. All eight DEV threshold selections were \`0.95\`; each preserved 96/96 positive continuation at the expense of **0/96 correct negative immediate STOP**. Every learned-only held negative root also continues. Its score receives retrospective paired AUC **0.50** at depth five (32 wins, 32 losses, 32 ties per 96 pairs) and **0.833** at depth six (64 wins, 32 ties), in every seed. These per-pair AUCs were calculated from original sealed held episodes after the official run; they were not preregistered success gates. The original source receipts preserve the probabilities.

## 5. Explicit computation accounting

| Counted source rule examinations across all eight seeds | Depth 5 | Depth 6 |
|---|---:|---:|
| RANK_CERT | 651,056 | 811,104 |
| PATH_CERT | 640,416 | 784,112 |
| ANCESTRY_CERT | 640,416 | 784,112 |
| LRANK_STOP | 1,502,816 | 1,802,752 |
| Public closure | **245,056** | **305,984** |

RANK_CERT examines **2.657×** as many public rules as closure at depth five and **2.651×** at depth six, plus its prior supervised training cost. Counts include the one-time public certificate for certified controllers and repeated inference-feature work. The frozen original neural policies require genuine checkpoint forward nodes, which are counted separately: 21,760 D5 / 24,160 D6 for Direct+cert, and 30,768 D5 / 33,168 D6 for forced Direct. These are different algorithmic units; no equal-FLOP or equal-wall-time claim follows.

## 6. Original scored evidence and post-score aggregation repairs

Original [source-locked CI #44033106880](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44033106880) passed the source lock, original-checkpoint smoke, all **eight** individually trained-ranker/learned-STOP evaluations and independent audits, and all **eight** frozen original-Direct evaluations and independent genuine-Torch audits. The final aggregation job failed for an **aggregation-only import**: it imported a Torch-dependent evaluator into a Python-stdlib runner. No scored seed job failed.

[Replay #44033672941](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44033672941) downloaded all 16 original artifact archives and passed their source membership reads, then encountered a second erroneous aggregation assertion: it required identical **STOP feature-row** digests across seeds despite those features containing each seed's separately trained ranker logit. The corrected aggregate checks the common source **ranker training graph/action SHA** and preserves every seed's distinct STOP TRAIN/DEV feature digest. No fit, threshold, held decision, score or science criterion was changed.

The dated post-score engineering record and all **16 original ZIP SHA-256** fingerprints are in [REAGGREGATE_AF31.json](../../experiments/activation_funnel_af31/REAGGREGATE_AF31.json). Successful [original-evidence replay #44033813293](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44033813293) fetched the same 16 archived source artifacts from scored run #44033106880, verified all per-member source/fit/proof checksums, reaggregated exact seed scores, and produced the two original registered **FAIL** verdicts under a green engineering audit.

Successful aggregate ZIP SHA-256: \`35cac26cbe0432cc97b1aeb341715a06cab2bfa75b9d9f17312a9465a398f186\`. Original aggregate member \`af31_results.json\` SHA-256: \`33ff2ffb42c14c9c86929e3e660862e3beb7ee74f47590d23f737f0f88d5855f\`. [AF31_EVIDENCE_MANIFEST.json](AF31_EVIDENCE_MANIFEST.json) pins this provenance. GitHub Actions artifacts remain subject to retention.

## 7. Scientific implication and next prospective discriminator

AF-31 independently establishes two distinct limits: learned rule selection can close proofs on these held graph families with the same effectiveness as strong public backward heuristics, and a source-fitted global-summary logistic STOP still fails absolute termination when calibrated on grammar-separated DEV roots. The original Direct checkpoint's low proof completion provides a genuine within-cohort neural-policy reference.

A new source-registered experiment should create **at least two independent, goal-visible distracting root actions** with supported unreachable cycles. It should measure **strict shortest-budget proof completion** and the existing shortest-plus-one protocol as separately predeclared arms, keeping PATH/ANCESTRY/public closure/BFS strong. This tests whether a learned ranker can preserve proof-optimal actions when an extraneous firing is costly. A STOP calibration study should preregister paired root AUC, absolute threshold transfer and domain-family-conditioned calibration separately, with an equal-budget direct-margin and a public symbolic certificate baseline.

**AF-31 remains draft and unmerged, with source-verified scientific failures preserved. AF-22 through AF-30 verdicts remain unchanged.**
