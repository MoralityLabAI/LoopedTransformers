# AF-32: strict-budget rule selection passes 8/8; learned-only STOP fails 0/8

**Registered scientific verdicts (November 17, 2026):**

- **\`STRICT_RULE_SELECTION_PASS\`**, eight of eight source-qualified fixed-seed fits, against the prospectively registered six-of-eight requirement.
- **\`LEARNED_STOP_STRICT_FAIL\`**, zero of eight separately audited certificate-free STOP fits.
- **Full engineering/proof audit PASS:** source lock, original frozen AF-23 model smoke, eight TRAIN-only linear+STOP training/scoring/physical-proof audits, eight original immutable AF-23 Direct full-trajectory/Torch audits and final aggregate all completed in [official scored CI #44043230390](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44043230390). **79,872** complete policy episodes independently replayed, with **zero false accepted Horn proofs**.
- **Original scored commit:** \`21de716b0a370274cd482bee41e102069d4dd064\`. **Original science protocol first committed at \`7493b18\`** before AF-32 generator and source training.

## Abstract

AF-31's producer-supported cycle benchmark allowed a shortest-proof-length+1 firing budget, in which a goal-visible but nonprogressing first action could be hidden by eventual proof completion. AF-32 freezes two independent root actions whose heads 7 and 12 each appear to lead to the GOAL through \`(7,10)->13\` and \`(12,10)->13\`, while positive Horn forward closure cannot derive prerequisite atom 10. Both misleading actions are legal from the root. An independent shortest-proof oracle certifies that neither advances the required proof length, and firing both leaves the productive proof length unchanged. AF-32 evaluates **strict shortest \`d\`** and **historical slack \`d+1\`** budgets on fresh but identical source graph distributions.

On the 96 source-held positive graphs of each depth, all eight independently trained eight-feature linear rankers complete **96/96 proofs under the strict budget** and correctly choose a shortest-progressing root action. The strongest equal-symbolic-STOP-certificate control, first-legal FIXED_CERT, completes only **4/96 depth-five and 5/96 depth-six strict proofs**, while PATH and unconditional ancestry complete **zero**. The registered learned advantage is therefore +92/+91 proofs per 96, well above the frozen +10 floor, simultaneously at both depths in all eight seeds.

The negative STOP decisions of \`RANK_CERT\` are supplied by a **public exact Horn fixed-point certificate**; they are not neural negative-goal recognition. The certificate-free \`LRANK_STOP\` classifier completes the same 96/96 positive proofs but correctly immediately stops only **32/96 negative held roots at each depth**, in every seed. Its registered floor is 60/96, so the independent STOP verdict is FAIL. Exact public closure and BFS64 still finish every positive strict proof and stop on every negative; closure uses less counted rule work than the learned hybrid. AF-32 establishes **useful source-family learned action selection against the frozen same-certificate controls**, with no claim of end-to-end neural proof search or computational advantage over exact public Horn inference.

## 1. Prospective registration, local matching and source integrity

The first [AF-32 PROTOCOL.json](../../experiments/activation_funnel_af32/PROTOCOL.json) was committed at [\`7493b18\`](https://github.com/MoralityLabAI/LoopedTransformers/commit/7493b184e350e572f8fee8b60a81f91114befd8d) before new AF-32 generator code, fitting, or heldout scoring. The protocol SHA-256 is \`cf40cdab5d844fc34a02792c67538345c41363775cc48e9752e14f05708c6550\`. Independent [source-only preflight #44042821031](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44042821031) passed before scoring; the final physical source manifest SHA-256 is \`bdef2605dfda3e9b6383622f91cea7f54e48e50869e4ceac6f2f45d6f6e90ecb\`. Original [AF-23 frozen model pre-score smoke #44042556562](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44042556562) separately passed real Torch forwards and independent source physical-step replay.

Each split contains **96 source-matched positive/negative pairs** (192 graph cases), from two logical grammar families with source-unique physical atom and rule permutations. The four splits are TRAIN, DEV, HELD depth 5 and HELD depth 6. Within a matched pair, exact root facts \`{0,1,2,3}\`, initial physical rule mask, marked role locations 6/13/14, complete directed physical per-node degrees, both node/rule permutations and seven-anchor physical-ID-preserving induced directed radius-two incidence signatures are identical. Exactly two unary premise incidences swap between productive source atom 4/5/9 and source-inaccessible atom 10; independent original host and physical-bitset BFS certify opposite global goal entailment labels.

Both enabled root diversions feed public reverse paths toward GOAL 14 through unreachable marked atom 13, one via source head 7 and one via head 12. Atom 10 has a producer in both held cycle families but lacks a derivable starting point inside its SCC. TRAIN/DEV use supported 2-cycles; HELD depths balance **48 supported 2-cycle** and **48 previously untrained 3-cycle** graph pairs, with premise-swap positions 1/2/3 contributing 32 pairs each. All noninitial atoms have a producer. The [constructive strict-budget theorem](AF32_STRICT_DISTRACTION_THEOREM.md) proves that a nonprogressing first firing costs one extra rule and two independent nonprogressing firings cost two.

The 96 pairs per split are physical relabelings and swap variants of a **small number of logical motifs**. Repeating the same 96 held positive and 96 held negative encoded graphs in eight independent fits produces denominators of 768 **source-case/seed occurrences**, not 768 independent grammar families.

### Pre-scoring engineering chronology

The source-only initial test-fixture cohort-key failure was repaired and the full 12-test matched-pair/policy preflight passed [#44042236860](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44042236860). The source-only proof was then strengthened to check that taking *both* independent root distractions leaves the exact shortest remaining goal distance unchanged; source CI [#44042711532](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44042711532) passed.

The first scored-workflow attempt [#44042993523](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44042993523) was blocked in its source-lock job, before any seed/model scoring: a copied engineering assertion searched for \`+10/96\` in the protocol text, which uses \`>=10/96\`. The science gate was already correct. The dated [AMENDMENTS.json](../../experiments/activation_funnel_af32/AMENDMENTS.json) retains this blocked launch, its exact correction commit and confirmation that every seed job was skipped. The fresh source-locked official run [#44043230390](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44043230390) passed all jobs with the original frozen scientific thresholds.

## 2. Registered policies and scientific gates

\`RANK_CERT\` uses the inherited AF-31 **eight-feature public physical graph legal-action ranker**, trained only on 528 canonical positive TRAIN proof states by fixed 500-update full-batch listwise softmax (LR 0.1, L2 0.01). The original AF-23 recurrent network is never optimized. Eight initial seeds are **17, 29, 43, 61, 73, 89, 103, 131**.

A public forward Horn least-fixed-point certificate determines immediate negative STOP for all \`*_CERT\` policies, including \`RANK_CERT\`, the same-feature untrained ranker, \`PATH_CERT\`, \`ANCESTRY_CERT\`, \`FIXED_CERT\` and \`RANDOM_CERT\`. Its rule examinations are charged once per episode, along with every repeated public reverse-feature evaluation. Public closure and BFS16/BFS64 are independent full symbolic controllers. Equal certificate access prevents a learned-ranker proof advantage from being conflated with negative-stop supervision.

\`LRANK_STOP\` uses the same trained legal-action ranker with a separate **ten-feature TRAIN-only 600-update logistic STOP**. It sees current physical public graph and state summaries, including the ranker's own action scores. A frozen DEV-only threshold is selected from 0.05..0.95 under the registered positive-continuation constraint. It receives no public closure/BFS truth at inference.

The eight historical original 6,036-weight AF-23 frozen checkpoints are scored on exactly the same AF-32 held graphs under native Direct STOP + ranking (\`AF23_DIRECT_NATIVE\`), forced continuation Direct legal ranking, and equal public negative certificate + Direct legal ranking. The original checkpoint ZIP/member identities remain SHA-pinned; original optimizer updates are **zero**. Independent auditors replay every physical rule action, all original emitted logit decisions and a sampled genuine Torch original-checkpoint reforward, separately from the synthetic ranker source.

### Frozen strict scientific requirements

**Primary \`STRICT_RULE_SELECTION_PASS\`** requires at both held depths in one model seed:

- \`RANK_CERT\` completes at least **60/96** reachable proofs in at most the exact shortest number \(d\) of rule firings.
- \`RANK_CERT\` correctly initially STOPs on at least **80/96** unreachable roots, with no false accepted Horn proofs.
- It completes at least **10/96 more reachable strict proofs** than the strongest *equal-certificate* untrained/PATH/ANCESTRY/FIXED/RANDOM arm.

At least **6/8** independent ranker initializations must satisfy all requirements on both depths. The **certificate-free learned STOP** has a separate strict gate: >=60/96 positive completed proofs, >=60/96 correct negative initial STOP, <=10/96 false positive initial STOP, zero accepted false proofs in >=6/8 seeds. The historical slack \(d+1\) budget is scored as a registered sensitivity arm and cannot rescue a failed strict criterion.

## 3. Complete strict-budget results

Counts are out of **768 source-case/seed episodes per truth class and held depth**.

| Strict-budget policy | D5 completed positives | D6 completed positives | D5 correct negative initial STOP | D6 correct negative initial STOP |
|---|---:|---:|---:|---:|
| **RANK_CERT learned legal-action ranker** + public STOP | **768** | **768** | **768** | **768** |
| **LRANK_STOP** learned legal ranking + learned-only STOP | **768** | **768** | **256** | **256** |
| **PATH_CERT** reverse blocked-path controller | 0 | 0 | 768 | 768 |
| **ANCESTRY_CERT** reverse-ancestry controller | 0 | 0 | 768 | 768 |
| UNTRAINED_CERT same eight public features | 0 | 0 | 768 | 768 |
| **FIXED_CERT** first-legal controller | 32 | 40 | 768 | 768 |
| RANDOM_CERT deterministic seeded random legal | 2 | 0 | 768 | 768 |
| BFS16 public state search | 0 | 0 | 768 | 768 |
| **BFS64 public state search** | **768** | **768** | **768** | **768** |
| **Public CLOSURE** | **768** | **768** | **768** | **768** |
| AF23_DIRECT_NATIVE | 0 | 0 | 480 | 528 |
| AF23_DIRECT_FORCED | 0 | 0 | 0 | 0 |
| AF23_DIRECT_CERT | 0 | 0 | 768 | 768 |

Every policy has **zero false accepted proofs**, independently replayed. Every \`RANK_CERT\` strictly completed proof has a source-certified minimum number of legal firings.

All eight source rankers independently achieve identical registered strict counts:

| Ranker seed | D5 completed /96 | D6 completed /96 | Strongest same-cert control D5/D6 | Registered proof gain D5/D6 | Primary two-depth gate |
|---:|---:|---:|---|---|---|
| 17 | 96 | 96 | 4 / 5 | +92 / +91 | PASS |
| 29 | 96 | 96 | 4 / 5 | +92 / +91 | PASS |
| 43 | 96 | 96 | 4 / 5 | +92 / +91 | PASS |
| 61 | 96 | 96 | 4 / 5 | +92 / +91 | PASS |
| 73 | 96 | 96 | 4 / 5 | +92 / +91 | PASS |
| 89 | 96 | 96 | 4 / 5 | +92 / +91 | PASS |
| 103 | 96 | 96 | 4 / 5 | +92 / +91 | PASS |
| 131 | 96 | 96 | 4 / 5 | +92 / +91 | PASS |

\`FIXED_CERT\` is the strongest equal-certificate control at depth five (4/96 per seed) and depth six (5/96). This registers **8/8 PASS** under the frozen 6/8 standard.

### Source structural group coverage and first-action errors

For each trained RANK_CERT seed at each held depth, **48/48 positive two-cycle** and **48/48 positive three-cycle** cases complete strictly. The premise swap positions each contribute **32/32** successful positives. The ranker makes **0/96 nonprogressing first legal rule choices**; PATH and ANCESTRY each make **96/96** such choices, independently reproduced by host bitset BFS.

Thus the new strict budget exposes a meaningful source-family rule-selection improvement that was obscured by earlier slack-only benchmarks. The original frozen Direct networks complete zero strict positives even when forced to continue on the *same* graphs, establishing a same-cohort comparison. The all-seed complete proof performance of the newly trained linear action ranker is far stronger on this small registered grammar family.

## 4. Strict vs historical shortest-plus-one

The same heldout source graphs, frozen learned parameters and original checkpoints are freshly rolled out under two separately registered action budgets.

| Policy | D5 strict | D5 slack (+1) | D6 strict | D6 slack (+1) |
|---|---:|---:|---:|---:|
| RANK_CERT | **768** | **768** | **768** | **768** |
| PATH_CERT | 0 | 0 | 0 | 0 |
| ANCESTRY_CERT | 0 | 0 | 0 | 0 |
| FIXED_CERT | 32 | 232 | 40 | 208 |
| BFS16 | 0 | **768** | 0 | **768** |
| BFS64 | **768** | **768** | **768** | **768** |
| CLOSURE | **768** | **768** | **768** | **768** |
| AF23_DIRECT_FORCED | 0 | 0 | 0 | 0 |

In every held positive case PATH and ANCESTRY first choose a nonprogressing root action. Their recorded policies also fail to complete within the one-firing slack budget; the source generator makes two independently goal-visible diversions available, while their specific later decisions account for the observed slack failures. BFS16 chooses a nonprogressing first action on every positive source case but recovers under the extra one-firing budget; this is exactly the registered budget-dependence the experiment seeks to reveal. Symbolic closure and BFS64 select shortest-proof routes, solving every strict positive case.

## 5. Certificate-free learned STOP

\`LRANK_STOP\` completes **96/96 strict positive proofs and zero positive false immediate STOPs per seed at each depth**, using the successful learned legal-action ranker. Its negative initial termination remains incomplete:

| HELD truth label/condition | D5 per seed | D6 per seed |
|---|---:|---:|
| Positive completed proofs | **96/96** | **96/96** |
| Correct negative initial STOP | **32/96** | **32/96** |
| Positive false immediate STOP | 0/96 | 0/96 |
| Required negative initial STOP | **60/96** | **60/96** |

All eight independent STOP fits fail the negative termination floor at both depths, yielding **LEARNED_STOP_STRICT_FAIL, 0/8**. The frozen symbolically certified \`RANK_CERT\` stops on every negative, but the learned-only diagnostic has no closure certificate.

**Retrospective, descriptive root-score analysis:** Original SHA-pinned \`episodes.jsonl\` and STOP fit receipts were read *after* the complete source-locked run. All eight DEV-fitted thresholds are **0.90**, at which each DEV cohort has 96/96 correct positive continuation and **48/96 correct negative initial STOP**. In the held cohorts, the negative STOP count is **32/96**, concentrated entirely at **swap position 1** (32/32), with **zero** at positions 2 and 3. The model's held root STOP probabilities yield paired AUC **0.50 at depth five** (32 wins, 32 ties, 32 losses) and **0.833 at depth six** (64 wins, 32 ties). These post-hoc score-ordering measurements cannot change either preregistered verdict and do not establish a transferable learned reachability theorem.

The separately recorded [retrospective original-receipt CI #44044792809](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44044792809) fetched all eight original SHA-pinned learned-seed artifacts from the scored run, verified each original `episodes.jsonl` SHA against its independent source audit, reconstructed matched root probabilities for both held depths, and independently confirmed the 0.50/0.833 paired AUCs and position-1-only STOP behavior. It performed **zero new model fitting, original checkpoint inference, DEV threshold selection, or heldout rescoring**. Its archived JSON member SHA-256 is `c70fb8496e0dba29da37c7734d32c101b56bdad6f28e6367e64377fb651710ae`.

## 6. Audited inference work and comparator limitations

Counted source public-Horn rule examinations at strict budget across all eight seeds and both positive/negative held roots:

| Controller | D5 examinations | D6 examinations |
|---|---:|---:|
| RANK_CERT including exact root negative certificate | 621,488 | 769,440 |
| PATH_CERT | 470,960 | 575,904 |
| LRANK_STOP public feature and learned STOP work | 1,369,312 | 1,649,872 |
| **Exact public closure** | **246,400** | **300,944** |
| BFS16 | 635,600 | 769,440 |
| BFS64 | 678,496 | 926,576 |

RANK_CERT performs **2.52×** as many counted rule examinations as exact closure at depth five and **2.56×** at depth six, in addition to the source-supervised model training cost. It performs fewer rule examinations than the source BFS64 controller, although the algorithms have distinct computational operations and the count alone cannot establish runtime/FLOP superiority. The original AF-23 neural checkpoints report their actual model-node forwards separately; these are not fungible with symbolic rule examinations.

Exact public closure remains a complete and more economical nonlearned solver on these tiny positive Horn graphs. Thus the registered PASS is a **qualified action-ranking transfer within a frozen public-feature/comparator budget**, not evidence that the model has discovered a general efficient algorithm for unrestricted Horn entailment. The source split has a small number of structural motifs and repeated physical permutations; reporting a confidence interval treating all 768 seed-case episodes as independent graph families would overstate generalization.

## 7. Original source audit and reproducibility

Successful original [GitHub Actions #44043230390](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/44043230390) completed all source-lock, frozen model smoke, eight full source-fit/score/audit jobs, eight original frozen AF-23 full-forward/physical-proof audit jobs, and the final scientific aggregate. Every seed has **7,680 learned-policy source episodes and 2,304 original Direct-policy source episodes**, over four strict/slack cohort keys, for **79,872** complete audited episodes total. Every stored policy action is checked against original host physical rule application and independent physical-bitset proof truth, while original neural decisions are independently reconstructed and sampled against real original Torch checkpoint forwards. Independent model refits reproduce the train-only ranker and DEV-frozen learned STOP fits.

The source-locked official aggregate archive, \`af32-cyclic-rule-stop-aggregate\`, GitHub artifact ID **14309778991**, has ZIP SHA-256 \`6b94ce19cf583bda27b01fd2e0c57669dfc6b5979a61b3fc5de213c919b6d366\`. Its exact member \`af32_results.json\` has SHA-256 \`b2b39b649bbf4b7a4a552f0d4bb8d795d5321fb7f3d99b11a8c739643e732974\`. The original sixteen seed ZIP artifact IDs and SHA digests, official scored source/protocol/manifest identity, and descriptive post-hoc STOP analysis provenance are committed in [AF32_EVIDENCE_MANIFEST.json](AF32_EVIDENCE_MANIFEST.json). All earlier engineering blocked runs remain visible. GitHub artifact bytes are subject to platform retention; science source and digests are committed in the repository.

The first prospective main source registration commit is \`7493b18\`, and the successful scored implementation is \`21de716\`. Source-only engineering corrections were documented before any model-scored job, including the blocked first source-lock attempt \`#44042993523\`. This chronology leaves the original thresholds, grammar family, optimizer, heldout partitions and scientific verdict invariant.

## 8. Follow-on discrimination

The positive source result deserves a stronger **semantic transfer** stress test before broader claims: freeze novel recursive/cyclic Horn motifs with distinct proof-support topologies rather than just fresh physical permutations or premise-swap positions. Preserve the same eight-feature learned ranker without retraining, and test it against a stronger, publicly implementable proof-planning heuristic matched on inference cost. Exact public closure must remain a comparator for both proof accuracy and source work.

A separate learned STOP study should preregister a **position-balanced calibration** against development-negative graphs that exercise all three productive premise-swap locations, plus absolute negative STOP, root paired AUC and per-family false STOP/continue. The AF-32 learned STOP head concentrates all successful held negative STOP decisions at swap position 1. Changes to this original held protocol or post-hoc threshold optimization must be treated as a new prospective experiment.

**AF-32 PR #143 remains draft and unmerged. AF-22 through AF-31 scientific verdicts remain unchanged.**
