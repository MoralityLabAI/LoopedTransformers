# RAM-07: naturally mined original Lean exact actions, first complete new-family transfer

**November 15, 2026. Original 120-case preregistered result: negative against the three strong non-neural comparators.** The source-only miner independently verified a positive `exact_typed` application from an actual old MathCan TRAIN theorem state, with no synthetic auxiliary examples. The subsequent small recurrent policies certified **4/12** original new-family theorems, matching global frequency, shape frequency and flat64; the fixed native order certified **3/12**.

## Source and time-ordered original proof context

The [original preregistration](PREREGISTRATION.md) began at
`e2a717aca927221f21fd3df9b47520b6f74f1f7c`, before collecting any RAM-07 genuine source-state actions or opening new theorem-family goal states. Two original MathCan source revisions were pinned, each with Lean4.35:

- TRAIN/DEV source: `patrickdugan/MathCan@c84e2025334e9a967c1297425a490f98d5b3d189`, the original FC-12 27-root train-only selection plus twenty additional source-order theorem roots from already evaluated older modules.
- New original heldout source: `patrickdugan/MathCan@fca46bfb3430ac85ffca16f6a6a83bb0b0231e41`, exactly the first three theorem declarations each from CoarseStateDependence, OrderedJet, TacticFibers and TreeReconstruction. All original source-reference proofs compiled in the new immutable original toolchain; 12/12 source-oracle certificates are reported separately from policy results.

The new source revision was accessible exclusively to the separate heldout evaluation jobs after old-source train-only independent Lean audits and nine model-checkpoint SHA freezes. Original teacher proof prefixes, file names and new heldout labels did not enter the train action generator or any ranker features.

## Training: genuine natural `exact` action finally verified

The original full source-only collector visited 47 old MathCan theorem roots, with a fixed 12-state-per-root, 5-transition, 8-goal bound. Its first breadth-first pass independently checked **10,564** compiler action outcomes but saw zero natural exact positives. [Pretest amendment A](PRETEST_MINING_AMENDMENT_A.md) discloses the preserved original negative run and subsequent TRAIN-only queue scheduling repair. The repair prioritized a genuinely accepted named introduction so the collector could reach the next actual original theorem goal within the same state budget, before any new heldout theorem access.

The amended source-only [CI training run #43682912918](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43682912918) then independently replayed **10,644** candidate effects with **zero disagreements** across 47 fresh Lean sessions:

| Original MathCan source cohort | Root theorems | Genuine goal states | Multigoal input states | Compiler-observed action outcomes |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 42 | 320 | 143 | 9,163 |
| DEVELOPMENT | 5 | 50 | 27 | 1,481 |
| Total | 47 | **370** | **170** | **10,644** |

Unlike RAM-06's synthetic exact curriculum, RAM-07's TRAIN observations have **exactly one** genuine positive `exact_typed` from the original previously proved `MathCan/Intervention.lean:obsEqOn_trans` theorem. The actual source-derived action sequence was:

`intro ram07_arg0`

`intro ram07_arg1`

`exact (hst ram07_arg0 ram07_arg1).trans (htr ram07_arg0 ram07_arg1)`

A distinct previous DEVELOPMENT theorem, `Intervention.obsEqOn_symm`, had one unique positive visible-state application from the analogous two generated named introductions and `exact (h ram07_arg0 ram07_arg1).symm`; the collector recorded the development action twice at the same goal-state hash. The separate [natural source-only smoke #43682572642](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43682572642) compiled both complete original theorem sources with empty axiom inventories and independent byte-identical compiler replay, without copying any reference proof prefix.

The **single natural TRAIN exact-positive** was oversampled for 10% of the effective batch. This verifies the mechanics of training from authentic source states; one unique theorem context supplies insufficient diversity for statistical evidence of broad typed-exact generalization.

The audited original train dataset SHA256 is
`7ac405fabdb559fb1ffa1baae54c29197177d24ace95a8fad7bb60df888cadd4`. All original FC-12 source-state/effect hashes remain unchanged and separately retained.

### Nine frozen low-memory CPU checkpoints

Training used CPU-only PyTorch2.6.0 and two CPU threads. The 30-update calibration achieved **80.15 updates/second** on the hosted runner, with 463,654,912 bytes peak train process-tree resident RAM (~442 MiB). All nine original model fits completed with the predeclared development-only selection rule.

| Model | Parameter count | Selected seed | Best development loss | Exact checkpoint SHA256 |
| --- | ---: | ---: | ---: | --- |
| Looped32, four tied passes | 85,125 | 29 | 0.620855 | `15aefb27ef9dd6cb9219fcd1e2e826fa542159a68b2eb2dc2ac95a03af4f271c` |
| Looped64, four tied passes | 207,109 | 17 | 0.641407 | `24be13905f50f111b2e3c285af48bfef003de97b1d1d857adbcd859781a04eda` |
| Flat64, no recurrent passes | 182,213 | 29 | 0.661141 | `ba04242b82b3a08a6c96d74352a875beb40e468719bfa570b243b53c9d8b6777` |

All nine seed-level optimizer/RNG checkpoints, source compiler traces and original train dataset are retained in artifact `ram07-frozen-audited-train-43682912918`, ID `14148788243`; ZIP SHA256 `c622367da9a7ced515df1db2869937b4edb61f3cc0736b4b51035c5442038db0`. See [frozen training receipt](TRAINING_RECEIPT_20261115.md).

## Original, contemporary six-controller source-family holdout

Primary: **30 seconds per theorem, 48 logical tactic evaluations, at most eight actual advancing transitions**, with complete full-source proof and independent in-budget source replay required for success. Secondary: **60 seconds, 128 logical evaluations, sixteen transitions**. Every controller used the same original source-derived 19-family grammar and printed goal, with at most eight pending goals and two bounded restart opportunities.

| Comparison | Original Lean theorems certified | Total search wall seconds | Logical Lean tactic evaluations | Lean processes |
| --- | ---: | ---: | ---: | ---: |
| Primary fixed native | **3/12** | 14.733 | 237 | 18 |
| Primary global TRAIN frequency | **4/12** | 18.106 | 221 | 20 |
| Primary goal-shape TRAIN frequency | **4/12** | 17.771 | 206 | 20 |
| Primary Looped32 | **4/12** | 19.511 | 237 | 20 |
| Primary Looped64 | **4/12** | 17.951 | 215 | 20 |
| Primary Flat64 | **4/12** | 16.858 | 198 | 20 |
| Secondary fixed native | **3/12** | 16.843 | 397 | 18 |
| Secondary global TRAIN frequency | **4/12** | 20.546 | 301 | 20 |
| Secondary goal-shape TRAIN frequency | **4/12** | 20.129 | 286 | 20 |
| Secondary Looped64 | **4/12** | 20.096 | 295 | 20 |

There are **four distinct original theorems** covered across all controllers, with 38 successful policy/theorem instances in the 120 scored attempts. Scores by original theorem follow:

| Newer original MathCan theorem | Native | Frequency | Shape frequency | Looped32 | Looped64 | Flat64 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| CoarseStateDependence.two_parent_covariance_numerator | · | · | · | · | · | · |
| CoarseStateDependence.joint_exceeds_product_marginal | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CoarseStateDependence.shared_critical_linear_growth | · | · | · | · | · | · |
| OrderedJet.scalar_order_invisible | · | · | · | · | · | · |
| OrderedJet.adjacent_weighted_numerator | · | · | · | · | · | · |
| OrderedJet.adjacent_coefficient_numerator | · | · | · | · | · | · |
| TacticFibers.exactly_two_matching_orbit | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| TacticFibers.diagonal_is_legal | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TacticFibers.shifted_is_legal | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TreeReconstruction.weights_sum_to_fifteen | · | · | · | · | · | · |
| TreeReconstruction.contrast_transfer_eigenvalue | · | · | · | · | · | · |
| TreeReconstruction.moment_gap_identity | · | · | · | · | · | · |

All four originally proved MathCan statements were certified using bounded simplification, arithmetic and literal unfoldings, without a natural `exact_typed` action in the accepted proof script.

### Newly found script beyond native ordering

The first previously known source theorem which fixed native ordering missed was `MathCan/TacticFibers.lean:exactly_two_matching_orbit`. Both recurrent models, the feed-forward control and both training-only symbolic-frequency policies certified it.

**Global frequency proof script**, five registered actions:

`simp_all only [MathCan.TacticFibers.legalTriple]`

`simp_all`

`simp_all only [MathCan.TacticFibers.allowed]`

`simp_all`

`omega`

Full source SHA256 `32d97bbdefc838d23a9ad6161b6976650b3e06aa155fca4e4e5dbaee07941d96`. Allowed axioms: `propext`, `Classical.choice`, `Quot.sound`.

**Looped64 proof script**, also five registered actions:

`simp_all only [MathCan.TacticFibers.legalTriple]`

`all_goals simp_all`

`simp_all only [MathCan.TacticFibers.allowed]`

`all_goals simp_all`

`omega`

Full source SHA256 `ca8e4d5a4130e01c68b8f36e3daeb5fff25996994478f24585d533948300c6f7`. Same allowed axioms. The recurrent model completed in nine logical tactic evaluations, compared with global frequency's twelve; the total theorem-coverage score was identical.

The registered criterion of at least one original new-family theorem certified by a recurrent policy but missed by **all three non-neural** comparator policies was **false**. The recurrence-specific criterion was also **false** because the flat control solved all four.

The eight original uncaptured theorems have untouched reference proofs compiling in MathCan. Their failures reflect the fixed action grammar, ranking and search budgets. Several include algebraic normalization, finite decision procedures and lemma application outside this restricted 19-family grammar.

## Full proof trust boundary and recorded amendments

The original successful training-only natural-exact mining required a pre-heldout source-only queue amendment, disclosed in [PRETEST_MINING_AMENDMENT_A.md](PRETEST_MINING_AMENDMENT_A.md), with the original failed mining attempt preserved.

The initial new-source evaluation was **incomplete** after 18 policy/theorem cases because a newer original MathCan theorem used `:= term` syntax not understood by the old source renderer. [Evaluation amendment B](EVALUATION_AMENDMENT_B.md) preserves that incomplete original scored artifact and documents the target-bounded renderer correction; the next evaluation-02 job failed its regression test before scoring. The subsequently repaired renderer passed source isolation and the complete 120-case evaluation-03 ran using all original frozen model checkpoint bytes, unchanged grammar, source commit, theorem names and strict budgets.

The [complete evaluation-03 CI #43686631584](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43686631584) finished all 120 original search trajectories and archived the **entire** generated proof source and compiler evidence (5,954 files) as `ram07-complete-prospective-eval03-43686631584`, artifact ID `14150907817`, ZIP SHA256 `5e6616aa6ddec3e70119b26fb10e79023d681f2be5194d4ae0ec5b7c4d73c00d`. Original immutable report SHA256 `581b62588568e43d282afd74ce8975e631400cb32d6b13836685dfff1fbaba5b`.

The extra third audit initially misread the word `admits` inside a MathCan/TacticFibers documentation comment as an executable `admit` tactic. [Evaluation amendment C](EVALUATION_AMENDMENT_C.md) records that 28 of 38 accepted original source files contain this benign comment and all 38 selected tactic scripts were clean. The audit-only correction validates executable registered tactic strings and keeps the full source-hash and allowed-axiom gate.

The [independent audit-only CI #43687337744](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43687337744) downloaded the **unchanged original scored artifact** and independently recompiled **all 38** accepted complete original sources with original Lean4.35 and explicit original allowed-axiom inventory. It found **zero false-positive certificates**, zero original score changes and published complete source-manifest SHA256 `a3e916fd9202d2e8c95ef033d348c8a719de68eb756f450b99e611db7feda1f7`. Its audit artifact `ram07-external-third-audit-43687337744` has GitHub ID `14150552956` and ZIP SHA256 `c28f61181bd66ce3f62459ee3abf4cbba13e2b16b5b7140f11f750a9656b9cbc`.

Original source-oracle proof statements, reference proof commands and synthetic curriculum outcomes earned **zero** credit in the model/native certification score. All claims above are about proof search on preexisting original formal mathematics, not new mathematical conjectures.

Peak hosted evaluation process-tree RSS was **2,050,228,224 bytes (~1.91 GiB)**, below the registered 6 GiB cap, with no resource abort. All wall times are single-host measurements; throughput and proof counts should be reproduced on new theorem families before extrapolating performance to practical Lean reasoners.

## Registered scientific inference and next test

The practical change from RAM-06 is a genuine natural-source Lean `exact_typed` training signal made accessible through named bound-variable introductions. The observed positive came from a single old original theorem root and was heavily oversampled. It did **not** produce new-family original theorem proof coverage beyond strong symbolic baselines.

A future separately registered campaign can broaden natural positive tactic support through more **distinct** original source roots and add symbolic `decide`, ring normalization, and local previously proved lemma-retrieval controls to the shared grammar. This is a new training/evaluation distribution and requires a new genuine source-family holdout before any test claim. RAM-07's frozen negative result and all three disclosed engineering steps remain unchanged.
