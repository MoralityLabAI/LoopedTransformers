# ROC-04B: prospective reserved-permutation confirmation, 384 new rule identities

**Registration frozen 17 November 2026 at commit `f032c52db2ebb544b745b3972f5f86f4c47c8b38`, before constructing these 384 input examples or labels.**

## Confirmatory question

ROC-04A [merged #155](https://github.com/MoralityLabAI/LoopedTransformers/pull/155) found 6/6 positive development accuracy differences after training the same query-conditioned recurrent architecture on four rules versus 512 new permutations, with a +0.477865 average and exact sign-flip p=1/64. That outcome used 256 exposed dev tables, and the separate complete-table competence gate failed. ROC-04B evaluates the same exact twelve frozen model checkpoints at their fixed 1,920th update on the 384 table identities reserved ahead of ROC-04A model training.

This test has **fresh source-table identities and fresh example schedules**, though it shares the same permutation sampling method and the same model population as ROC-04A. Its evidence concerns independent rules for these frozen trained networks, rather than new independent architecture training.

## Frozen provenance

The registered ROC-04A reproducibility archive SHA-256 is `e16cfca1feae0a495cf86a93a2ff669ab1c699c1b61a6d550a68c2f7de5e12e6`. The registration config pins the SHA-256 for each of the twelve `.pt` checkpoints, four architecture/training constants, the original high-training and development pool digests, and the full 384-table reserved pool digest `f37c937bdbd5825a5c6726c7a8b16f31a98277e16492028de9d71d181fae99ae`.

The registered source generator uses the ROC-04A shuffled 8!-8 bijective rule universe, with indices 768:1152 as reserved tables, and a new schedule seed `20261117101`. Each of those 384 identities receives exactly one legal delayed-rule case, with a variable 2–16 event delay, source-B/C distractors and a final query. The raw source-table and interpreter lookup determine identical labels. Source IDs and labels are produced **only after the preceding registration commit**.

The original ROC-02 separately sealed source families 6/7 are excluded from the rule universe and their input files are never loaded. The reserved rule table identities themselves were pinned before ROC-04A training. ROC-04B has no fitting loop, optimizer updates, gamma selection or model ranking.

## Fixed primary and secondary gates

For each of six frozen seeds `2026111705, 07, 11, 13, 17, 19` and the same 384 new source records, compute task accuracy of the **high-permutation512** and **low-affine4** models, with the existing visible query-to-table-row route. A primary replication passes only when high improves for at least five of six seeds, the six-seed average absolute accuracy improvement is at least 0.20, and the one-sided exact 64-sign-flip probability is no greater than 0.05.

The additional competence gate again requires at least five of six high-diversity models to exceed 70% test answer accuracy **and** at least five of six to reconstruct the complete eight-row table on 50% of inputs. Keep the gates separate. Report every seed, CE, rowwise accuracy, exact table accuracy and raw numerator of correct test cases. Multiple comparison families and case-cluster dependence remain visible; six model seeds are the inferential unit.

The predeclared secondary task controls use a valid initial-A permutation row-swap changing the exact answer while leaving the final query unchanged, plus a B/C distractor-table replacement keeping the exact target fixed. Report legal-edit new-label accuracy, base-plus-edit joint correctness and distractor prediction stability among eligible cases. Those natural-dose discrete input edits are task checks, rather than matched-L2 hidden-state mechanism interventions.

## Publication and audit

Open the frozen 384 cases once, score all twelve frozen original checkpoints, publish the complete 384-example source and its SHA-256, and preserve 4,608 per-case score/control records (384 × six seeds × two arms), a six-seed CSV, a signed-permutation audit, and source/weight/receipt SHA manifests. Keep the original ROC-04A development result intact. The exact symbolic interpreter is a 100% reference on valid records.

All ROC-04B results are a model-fixed, previously reserved rule-identity holdout from the 8! permutation source. This does not establish natural-language reasoning, hidden-state mechanism type, a universal low-rank observer rank floor, or ROC-01's physically matched five-mechanism classification. A new independent training-seed and architecture campaign is an optional later replication after this result is published.
