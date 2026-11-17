# AF-30: Complete held-out Horn proofs, no learned advantage over a public path-cost heuristic

**Official scientific verdict: \`RULE_SELECTION_TRANSFER_FAIL\`, 0/8 source-qualified seeds.**  
**Source proof and independent action audit: PASS, all eight seeds.**  
**Scored source:** \`579d7ff64addc9e6dded1fbdd93fdbceb1b1247b\`, November 17, 2026.  
**Scored GitHub Actions:** [run #44028290737](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44028290737) (source lock + eight seed jobs + independent aggregate all SUCCESS).

## Abstract

AF-29 achieved paired AUC 1.00 for globally distinct Horn STOP decisions but finished zero positive proofs with its frozen Direct rule ordering, even when forced to continue. AF-30 trains a lightweight eight-feature legal-action ranker on fresh matched Horn pair grammars, including a previously unseen premise-swap position. At each held depth, the trained ranker attached to a publicly computed negative-goal certificate completes **96/96 positive proofs in all eight independently initialized rankers**, while stopping correctly on **96/96 negative goals in all eight**.

A pre-scoring stronger hand-coded, equal-certificate **PATH** controller also finishes **96/96 positives in every seed and both depths**, using the same public feature extraction. Therefore the prospectively registered improvement requirement (+10/96 positive proofs over the strongest certificate-matched nonlearned controller) fails **in every seed**. The source-verified result is \`RULE_SELECTION_TRANSFER_FAIL\`, although its absolute completion and stopping floors pass.

The public full closure baseline likewise finishes every positive proof and correctly stops every negative. It uses **fewer counted rule examinations** than the learned ranker including certificate and repeated feature construction. AF-30 shows that legal-rule selection is sufficient to repair proof execution on these families with a new trained head, while demonstrating that a simple nonlearned path-cost rule obtains the same coverage more economically.

The negative STOP certificate in RANK_CERT is exact symbolic Horn closure run once on the public root graph. **No neural STOP/general reachability claim follows from the negative count.** This is a different heldout graph cohort from AF-29; the earlier zero-proof Direct result is historical context, not a new same-source Direct baseline.

## 1. Registration and original source lock

The first scientific protocol was [committed at \`78a5b1e\`](https://github.com/MoralityLabAI/LoopedTransformers/commit/78a5b1e19924206b59551cd8683e5535cbd12bf3) before any AF-30 generator, feature code or test scoring. The dated, **pre-model-fitting** amendment records the original structural preflight's unique-template failure [#44027188956](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44027188956) and the repaired preflight [#44027333019](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44027333019). A second pre-scoring amendment adds PATH_CERT to the certificate-matched control pool, which strictly strengthens the positive proof-improvement requirement. The original **6/8, 60/96, 80/96 and +10/96** science thresholds remain fixed.

The final expanded source preflight [#44028198537](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44028198537) passed the 384-pair structural certificate and deterministic training/closed-loop tests. The official scored source was frozen at \`579d7ff\`, launching only after the successful preflight, with:

- Source manifest SHA-256 \`28fada655b813d444a5345f0130c0dfded7477f5a3c49263f14dcb70ca464b94\`.
- Protocol SHA-256 \`7ea12cdc02c78cc852eb457f31e6ccb40792959eca079c441b71afd35ced82b8\`.
- Scored aggregate archive SHA-256 \`7b0481db0a35c648ffa3691e7106be36617dce7a3ac47f622bbb29fd40cb6da3\`.
- Aggregate member \`af30_results.json\` SHA-256 \`c150fa5a09655b5fd8e1c09beedc9cc2611fafe5cdff4bce3560995e0a7700fb\`.

All source artifact identities and independent per-seed audits are pinned in [AF30_EVIDENCE_MANIFEST.json](AF30_EVIDENCE_MANIFEST.json).

## 2. Matched grammar construction

There are four distinct source splits, **96 reachable/unreachable matched pairs (192 graphs) each**, with two grammar families per split and 48 pairs per family.

| Split | Pairs | Productive shortest positive path | Swapped productive premise positions |
|---|---:|---|---|
| TRAIN | 96 | 5 or 6 firings | 1:48, 2:48 |
| DEV | 96 | 5 or 6 firings | 1:48, 2:48 |
| HELD depth 5 | 96 | Exactly 5 firings | 1:32, 2:32, 3:32 |
| HELD depth 6 | 96 | Exactly 6 firings | 1:32, 2:32, 3:32 |

Each positive source chain progresses from root facts \`{0,1,2,3}\` through productive atom heads \`4→5→9→11\`, then to GOAL atom 14 at depth 5, or through marked atom 6 at depth 6. The paired negative switches the premise on one of chain steps 1/2/3 to unreachable atom 10, compensating with the reverse premise swap on a decoy rule. All source cases have two blocked apparent goal corridors from atoms 7 and 12 to marked atom 13. Those corridors require unproducible atom 10.

Every pair has precisely the same physical node/rule permutations, complete directed degree profile, root legal-rule mask, initial known facts and marked role locations. The **seven-anchor, physical-ID-preserving directed radius-two incidence signatures are exactly equal**. Only two single-premise rule slots differ. Host state search and independent public-physical bitset BFS certify opposite proof outcomes and exact positive shortest lengths. The positive templates differ across source-registered swap positions; the original preflight blocked on a duplicate logical template before the repair.

The 96 pairs per split are **two motif families with swap-position variants and fresh physical permutations**. Their number measures distinct encoded cases, not 96 independent logical grammar structures. The source manifest checks complete graph identity against historical AF-29 graphs and across all AF-30 splits.

## 3. Train-only policy and controls

The learned model is a deterministic **eight-feature linear action ranker**, fitted by 500 full-batch listwise softmax updates at fixed learning rate 0.1 and L2 0.01. All features come from the complete public physical graph and current proven-state bits, including reverse goal ancestry, a blocked-premise-aware reverse path cost, downstream productivity and rule arity. Exact shortest-proof action labels are used **only on TRAIN**; no heldout truth, pair identity, split or budget is a model input. Each of eight independent initialization seeds trains over the same 528 labeled TRAIN states. The original AF-23 recurrent model is not loaded or updated.

The \`RANK_CERT\` controller uses the learned ranker plus an exact public Horn fixed-point **negative STOP certificate** computed once at the root and explicitly charged to the work ledger. \`UNTRAINED_CERT\` receives the same features and standardization but uses unoptimized initialization weights. \`PATH_CERT\` is a strong hand-coded selector using that same extractor's blocked reverse goal distance. \`ANCESTRY_CERT\`, \`FIXED_CERT\` and \`RANDOM_CERT\` share the root certificate, while public closure and BFS16/BFS64 run as separate full-symbolic controllers.

The frozen proof budget is shortest positive proof length plus one firing, or two decisions on negative graphs. The budget is held by the evaluator and never passed to inference. The DEV split was reserved but no DEV search or hyperparameter selection was performed: optimization and architecture were fixed before scoring. A fully trained STOP-only neural policy was not evaluated in this run; neural-only stopping is an explicit further diagnostic, separate from the scored certificate-assisted result.

### Registered science PASS requirement

In a seed at **both** held depths, \`RANK_CERT\` must finish at least **60/96** positive proofs, correctly initially STOP at least **80/96** negative graphs, finish at least **10/96 more** positive proofs than the best equal-certificate nonlearned controller including PATH, and accept no false Horn proof. At least **6 of 8** source-initialization seeds must pass every condition simultaneously. Any completely source-audited failure is \`RULE_SELECTION_TRANSFER_FAIL\`.

## 4. Official complete-proof outcome

Each table denominator is 96 distinct held source cases **reused across eight seed fits**, giving 768 evaluated source-case/seed occurrences at each depth and truth label.

| Policy | Depth 5 completed /768 | Depth 6 completed /768 | D5 negative immediate STOP /768 | D6 negative immediate STOP /768 |
|---|---:|---:|---:|---:|
| **RANK_CERT learned ranker** | **768** | **768** | **768** | **768** |
| **PATH_CERT hand-coded** | **768** | **768** | **768** | **768** |
| UNTRAINED_CERT | 160 | 128 | 768 | 768 |
| ANCESTRY_CERT | 0 | 0 | 768 | 768 |
| FIXED_CERT | 80 | 56 | 768 | 768 |
| RANDOM_CERT | 12 | 5 | 768 | 768 |
| CLOSURE | **768** | **768** | **768** | **768** |
| BFS16 | 96 | 0 | 256 | 256 |
| BFS64 | **768** | **768** | **768** | **768** |

The 8/8 trained seeds reach **96/96 positive completions at depth 5 and 96/96 at depth 6**. Both the trained controller and PATH solve every swap location, including the entirely held-out position 3. Every seed has **0/96** improvement over PATH at either depth, which gives **0/8** successful seeds under the stronger +10/96 comparison. Absolute trained proof completion and certified negative STOP both pass. All source arms accept **zero false Horn proofs**.

The excellent learned coverage can be attributed to train-supervised rule ranking **on these source templates**; its equality to PATH means the learned action rule has no demonstrated advantage over a compact nonlearned symbolic heuristic on this family. The frozen AF-29 original Direct rule ranker was not rerun on the AF-30 graph split, so a same-graph neural-vs-Direct comparison is not part of this evidence.

## 5. Inference work

These totals count original source rule examinations performed by each algorithm across all eight seeds and the full 192-case per-depth cohort. They charge RANK/PATH for the single root certificate **and** for repeated reverse-feature calculation.

| Algorithm | Depth 5 rule examinations | Depth 6 rule examinations |
|---|---:|---:|
| RANK_CERT | 711,424 | 873,824 |
| PATH_CERT | 711,424 | 873,824 |
| ANCESTRY_CERT | 701,568 | 847,952 |
| Public CLOSURE | **243,152** | **305,984** |
| BFS16 | 1,026,032 | 1,169,280 |
| BFS64 | 1,107,120 | 1,562,624 |

RANK_CERT requires **2.93×** the counted rule examinations of closure at depth 5 and **2.86×** at depth 6. Algorithmic work units are auditable but differ from common FLOP counts. CI per-policy nanosecond timers were collected on independent hosted runners and should not be interpreted as a calibrated speed benchmark. Training costs are additional to learned inference.

## 6. Source-independent verification

The source lock, eight per-seed jobs and final aggregate in [GitHub Actions #44028290737](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44028290737) passed. Each seed generated **3,456 complete trajectories** (192 cases × 2 depths × 9 arms) and underwent independent public-bitset replays of every legal action, exact STOP/certificate label, final proof and work ledger, together with an identical train-only linear model refit. The aggregate confirms:

| Audit gate | Verified |
|---|---:|
| Original scored seed fits | 8 |
| Complete independently audited episodes | **27,648** |
| Source/pair/graph roster and physical transitions | PASS |
| Train-only deterministic ranker refit | 8/8 |
| All heldout action decisions reconstructed | PASS |
| Per-seed per-depth policy counts reconstructed | PASS |
| Zero false accepted Horn proofs | PASS |
| Registered science seeds passed | **0/8** |

### Scientific interpretation and next discriminating test

AF-30's specific false-short goal branches contain a locally detectable bottleneck: the other premise on the apparent shortcut is atom 10, which has no rule producer. A public blocked-premise-aware reverse distance therefore ranks the true chain first without learning; this explains the perfect PATH control and why the new benchmark remains tractable to a small symbolic method. The [companion constructive family theorem](AF30_BLOCKED_PATH_THEOREM.md) derives this result for a whole penalty interval, independent of physical relabelings.

A follow-on experiment should vary **recursive unsupported dependencies** where every apparently needed atom has a producer yet the goal-support dependency graph is a closed unreachable cycle, and should evaluate same-source native Direct and its learned alternatives. Freeze that generator and same-feature PATH/closure baselines before collection. Keep any neural STOP-head result separate from public-certificate STOP.

**AF-30 PR #138 remains draft and unmerged; AF-29 PR #137 remains draft.** Earlier AF-22 through AF-29 science verdicts remain unchanged.
