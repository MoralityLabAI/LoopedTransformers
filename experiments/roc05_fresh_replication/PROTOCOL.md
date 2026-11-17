# ROC-05A: Independent source-pool and model-seed permutation transfer replication

**Source and inference registration frozen at `a56c177bbc956274fbf65018b5724d291e25ef3c` on 17 November 2026**, before ROC-05 samples or ROC-05 model fitting. This protocol repeats the ROC-04A fixed-budget experiment using a genuinely new rule-table pool and **six new model initializations**.

ROC-04A on exposed development rule identities achieved mean 512-rule minus four-rule task accuracy +0.4778646 (six positive seeds, one-sided exact signflip p=0.015625); ROC-04B independently confirmed the same frozen model cohort on its 384 previously reserved rule tables with gain +0.51953125. Neither cohort demonstrated high complete-eight-row table accuracy. Those old ROC-04 models, its 256 dev rules and its 384 consumed test rules are **excluded from ROC-05 selection and scoring**. ROC-02's original six/seven family examples remain sealed.

## Independent source and model lock

Generate all \(8!=40,320\) bijections over the labels 0–7. Remove the eight original affine source families and **all 1,152 deterministic ROC-04 training/development/test table identities** from its source pool; this leaves 39,160 candidate permutations. The new Python Random seed `20261117211` fixes a fresh deterministic shuffle. The first 512 tables are the high-diversity train rules, the next 256 are new exposed development rules, and the next 384 are **allocated only as future reserved identities**. ROC-05A constructs exactly three source files: `train_low4.jsonl`, `train_high512.jsonl` and `development.jsonl`. The fourth reserved partition's input examples and labels are ungenerated and never scored.

Use identical deterministic 1,024 train event skeletons for the two arms (schedule seed `20261117217`) including source B/C distractors, 2–16 event delays and final query. Only the initial source A table and the exact query answer change. Four-rule training uses the already exposed ROC-02 training rules, 256 examples per rule. High-diversity training assigns two cases to each newly sampled permutation table. The development schedule seed `20261117219` creates one example per disjoint new development table. The assignment seed `20261117229` sets fixed example assignment. The exact interpreter and literal first-source-table lookup must agree on every answer.

Six new seeds `20261117231, 20261117233, 20261117239, 20261117241, 20261117251, 20261117257` train the exact same ROC-03B query-conditioned recurrent model with 51,520 parameters, a 128D hidden state and eight tied residual steps. Clone identical initialization per seed; both arms use the same sampled case-index stream and identical 1,920 AdamW updates (lr 0.001, batch32, weight decay 0, clip5). The loss is eight-way task cross-entropy plus 0.5 times eight-row source-table reconstruction cross-entropy. Choose the fixed **step 1,920** checkpoint regardless of dev outcomes. Milestones at 240/480/960 are descriptive only.

## Registered primary gate and inference

For each seed calculate accuracy on the **same 256 new development permutations** for high512 and low4, and report the six paired differences. The registered primary succeeds only if (1) at least 5/6 differences are positive, (2) mean gain >=0.20 absolute, and (3) one-sided exact sign-flip p <=0.05 across all 64 sign allocations. The six independently initialized paired models, rather than 256 dev examples within a model, are the inference units.

A **separate competence gate** succeeds only if at least 5/6 high-diversity models achieve dev answer accuracy >=70% and at least 5/6 achieve **complete all-eight-row table** accuracy >=50%. This second gate stays independent of the primary. Print task CE, train accuracy, per-row reconstruction, whole-table exactness and the raw correct-case counts.

After the final checkpoint, run two exact symbolic controls on each development example. First swap the source A binding table outputs for the queried row and row q+4 (mod 8), retaining query and timing, and score new-label answer accuracy and jointly correct original/new target pairs. Then replace an intervening source B/C distractor table when present, preserving the true A answer; score prediction stability and task correctness. Their costs are natural and unequal, so these are task-level specificity controls, not ROC-01 physically normalized hidden-state interventions.

## Reproducibility, sealed controls and follow-up

Publish the three source JSONL hashes, exact train/development/reserved table-identity digests, the two original 1920-step model checkpoints and optimizer states per seed (12 files), all six independently verified model receipts, twelve per-case prediction/control CSVs, exact sign-flip counts, independent source checks, and one SHA-256 indexed archive. Hosted PyTorch CPU requires no paid GPU.

The source-only and training code **must not read or generate** the original ROC-02 sealed source-family-6/7 cases, ROC-05's 384 future reserved examples, or any ROC-04 reserved examples as new test cases. ROC-04's previously consumed table identities are used solely as an exclusion list.

The next study, **ROC-05B**, can use the same exposed ROC-05 dev cases for physically matched rule-state perturbations after locking the new checkpoints. It will test the true changed-target margin under source A row swaps, compare input-softened and hidden-space directions at equal midpoint norm, and report rank-7 null-calibrated A/B/C/H diagnostic cards. It will not establish independent rule-family generalization until the separately reserved ROC-05 test is opened under its own frozen protocol.

This ROC-05A replication studies the effect of training **table diversity** under a model/optimizer budget matched within its own cohort. ROC-04 results are informative prior context; differences between ROC-04 and ROC-05 dev means do not constitute independent causal arm comparisons.
