# AF-21: direct greedy STOP-margin training, scored report

**Official registered scientific verdict: FAIL. Hosted engineering and independent audit: PASS.**

The [original GitHub Actions run #43729435530](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43729435530) passed all ten jobs. The source-pinned preflight compiled the conditional Bellman theorem with Lean 4.19.0 and ran both Python test modes with Python 3.12 / Torch 2.5.1+cpu. All eight seed jobs and the cross-seed independent aggregation passed.

**Frozen scientific source:** [0eaeb80a68dc1232cae36a59642d49efdbf85797](https://github.com/MoralityLabAI/LoopedTransformers/commit/0eaeb80a68dc1232cae36a59642d49efdbf85797), with exact Git blob matches for the ten locally tested files.
**Registered protocol:** [PROTOCOL.json](../../experiments/activation_funnel_af21/PROTOCOL.json), SHA-256 e4ca05075ddd390933da3ad55183a3700a98762170efd560658a7b7cb90cf14e.
**Prospective case manifest:** [fresh_manifest.json](../../experiments/activation_funnel_af21/fresh_manifest.json), SHA-256 3d6011b2b02ff68893eb63829b33acb2c2d5d58e5c02dc85da7ea1c5befc1921.
**Official aggregate results.json SHA-256:** 62547234c624ba3cf0ae1bc3b593f618178b78faac54f8999863c75586fb0b0b.
**Draft PR:** [#106](https://github.com/MoralityLabAI/LoopedTransformers/pull/106), stacked on AF-20 PR #102; preserve the source and keep unmerged.

## Registered design

The original 6,036-parameter source-hash-pinned TypedLoop(16) with graph-ancestor readout and four native recurrence calls ranks Horn rule firings. The host verifies each accepted inference against public rule premises. Models have identical seeds, initialization, 240 scheduled update sample streams and source native forward calls for the main arms; extra backward FLOPs differ.

Original AF-20 train/development cases remain exactly 928/352. The 304 changed-reachability and 484 equal-cost connected training pairs are unchanged. New prospective heldouts comprise 416 cases in layouts 72–76 with three previously unseen recombinations of cross-branch Horn dependencies and shortcuts. These are novel Horn compositions within the existing logical operators.

| Arm | Training objective | Updates |
|---|---|---:|
| BCE | AF-20 logsumexp STOP mass, reach, action and remaining-cost supervision | 240 |
| Greedy | BCE plus exact greedy STOP-versus-best-legal-rule margin | 240 |
| Greedy + rule | Greedy plus best optimal rule versus best suboptimal legal rule margin | 240 |
| Extended ranking | Same combined loss with 50% more updates | 360 |

The true greedy STOP margin is m = score(STOP) minus max score(enabled rule). On states with a valid proof continuation, the new loss uses softplus(0.5 + m); on states where STOP is optimal but rules are enabled, it uses softplus(0.5 - m). The rule objective uses softplus(0.5 + best_bad_rule - best_optimal_rule). This directly targets the host's greedy action selection.

## Original hosted eight-seed heldout scores

All eight seeds pass the preregistered 65% training/development proof-success qualification requirement for the three main arms. Every eligible seed fails one or more registered held-depth4/held-depth5 improvement, absolute success, early STOP or connected-distractor requirements.

Proof success percentages below count solved provable goals within the exact minimum Horn rule-fire budget plus one.

| Seed | BCE depth4 / depth5 | Greedy depth4 / depth5 | Greedy + rule depth4 / depth5 | Extended ranking depth4 / depth5 |
|---:|---|---|---|---|
| 17 | 55% / 55% | 12.5% / 7.5% | 25% / 22.5% | 27.5% / 5% |
| 29 | 72.5% / 0% | 72.5% / 27.5% | 0% / 0% | 0% / 0% |
| 43 | 35% / 35% | 7.5% / 5% | 12.5% / 0% | 0% / 0% |
| 61 | 0% / 0% | 0% / 0% | 0% / 0% | 65% / 37.5% |
| 73 | 0% / 0% | 0% / 0% | 0% / 0% | 0% / 0% |
| 89 | 0% / 0% | 0% / 0% | 27.5% / 27.5% | 2.5% / 0% |
| 103 | 17.5% / 17.5% | 10% / 7.5% | 2.5% / 0% | 12.5% / 10% |
| 131 | 75% / 30% | 70% / 10% | 25% / 25% | 57.5% / 0% |

Mean held-depth4 success over eight seeds: BCE 31.88%, greedy 21.56%, combined 11.56%, extended ranking 20.62%. For depth5: BCE 17.19%, greedy 7.19%, combined 9.38%, extended ranking 6.56%. The registered combined-versus-BCE mean difference is **minus 20.31 percentage points at depth4** and **minus 7.81 points at depth5**.

Seed 89 has positive combined-loss gains of 27.5 points at both depths, with only 27.5% absolute success. Seed 29's combined model prematurely selects STOP on every initially provable deep example despite its high training/development scores. The connected-distractor requirement fails in multiple seeds. The host accepted zero false Horn proofs, a structural consequence of the legality gate.

## Independent verification receipts

Eight original CI seed ZIPs contain 32 frozen Torch checkpoints and their matching named NumPy exports, full action logits, proof transitions, pair inputs and audit records. The original hosted cross-seed audit confirmed:

| Verified observations | Total |
|---|---:|
| Host proof episodes | 23,296 |
| Connected and reachability-flip counterfactual records | 10,528 |
| Native policy checkpoint forward replays | 798 |
| Paired-state neural forward replays | 384 |
| Independent training-pair source oracle checks | 6,304 |

All eight independent audit receipts returned verified=true, with zero falsely accepted Horn proofs. [Original artifact IDs, ZIP SHA-256 digests, and provenance](AF21_EVIDENCE_MANIFEST.json) are preserved for rerun and long-term archiving.

The separately labeled local Python 3.13 / Torch 2.10 run also achieved a registered FAIL, with 833 policy replays and some different individual seed outcomes. Local aggregate SHA-256 ef4b864cfa6f7a6012d0fe27d7979db06a543f5674c6da4986d9c32263c8ab1d. The **hosted Torch 2.5.1 scores are the official result of record**.

## Formal implication and mechanism

[AF21_GREEDY_POLICY_THEOREM.md](AF21_GREEDY_POLICY_THEOREM.md) distinguishes the conditional mathematical proof-completion theorem from trained model performance. If the selected action is optimal at every visited reachable nonterminal state, it decreases the exact remaining Horn proof cost by one; induction on the initial cost establishes completion in the shortest number of rule firings. Lean checks that conditional descent theorem. Its premises are empirical questions for a neural policy.

AF-20 identified a logsumexp versus greedy action mismatch. AF-21 removes that mismatch from the new objective, yet the combined model frequently assigns STOP excessive absolute priority on unfamiliar Horn programs. A relative or supervised margin on training states has not yielded robust out-of-composition calibration. The next architectural experiment should test recurrent value iteration over host-certified proof successor states and exact remaining cost, including a compute-matched control.

**Ownership:** Preserve AF-01 through AF-20 experimental evidence. AF-21 remains a draft PR, and its failed outcome must stay fixed.
