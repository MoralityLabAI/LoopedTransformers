# RAM-07 preregistration: natural MathCan exact proof-state mining

**Frozen November 15, 2026, before any RAM-07 data collection, model training, or held-out Lean proof-state access.** This is an independent successor to RAM-06 on a new branch, with no changes to RAM-01 through RAM-06 results, frozen data or checkpoints.

## Research question and non-neural controls

RAM-06 trained with 12 independently compiler-verified synthetic auxiliary exact-positive examples but **zero** successful exact applications among 793 original MathCan candidate attempts. Its completed original theorem evaluation was 7/12 for each learned model and 8/12 for a training-only family-frequency baseline.

In a separate source-family-held-out experiment, can a small recurrent scorer trained on **real positive exact applications from original MathCan proof-state descendants**, discovered without original teacher prefixes, certify a complete original new-family theorem that all three non-neural policies (fixed native order, train-frequency, shape-conditioned frequency) miss under identical proof-search budgets? A recurrence-specific gain must also be missed by the nonrecurrent flat64 scorer. Report all zero and negative results without revising this criterion.

## Immutable provenance and disclosure of source revision

**Original training sources**: `patrickdugan/MathCan@c84e2025334e9a967c1297425a490f98d5b3d189` and FC-12 original train-only artifact `43547762837`, with state SHA256 `403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0` and original candidate-effect semantics SHA256 `bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec`. Original reference proof commands are explicitly prohibited as mining or model inputs. This original source still compiles with `leanprover/lean4:v4.35.0`.

**New test source revision**: `patrickdugan/MathCan@fca46bfb3430ac85ffca16f6a6a83bb0b0231e41`, toolchain `leanprover/lean4:v4.35.0`. This is a newer immutable repository commit with four new top-level MathCan modules absent from the pinned old training source and all previously selected RAM-03/04/05/06 holdouts. The new modules were selected using only module names and first-three theorem declaration names. None of their proof bodies, actual compiler goal states or candidate outcomes is used to design/train/select a model, create a tactic, or tune a baseline. This is a procedural time-ordered, public-source holdout; the newer source may have further changes to older modules, so the test import graph must be built entirely at the **new test commit**, while all training extraction uses the **old original commit**. Preserve both source SHA manifests.

The **twelve original RAM-07 prospective targets**, exact first three source-order theorem declarations in each newly introduced file, are:

- `MathCan/CoarseStateDependence.lean`: `two_parent_covariance_numerator`, `joint_exceeds_product_marginal`, `shared_critical_linear_growth`.
- `MathCan/OrderedJet.lean`: `scalar_order_invisible`, `adjacent_weighted_numerator`, `adjacent_coefficient_numerator`.
- `MathCan/TacticFibers.lean`: `exactly_two_matching_orbit`, `diagonal_is_legal`, `shifted_is_legal`.
- `MathCan/TreeReconstruction.lean`: `weights_sum_to_fifteen`, `contrast_transfer_eigenvalue`, `moment_gap_identity`.

`MathCan/OrderedJet.lean` is not imported from `MathCan.lean` at the frozen new source commit; explicitly compile the file through its own Lean module/target as part of the original source oracle. The other three new modules and all transitive imports must be built at that same new commit. Require a separate original-source proof oracle for all 12/12 target statements.

## Genuine MathCan training theorem selection

Retain the original FC-12 **27 TRAIN cohort roots** and its five unmodified development-only original theorem groups:
`HookState.endpoint_append`, `HookState.pathSum_gauge`, `TOEGraph.integral_ext`, `Intervention.obsEqOn_symm` and `TOECosmology.anchored_present`. The remaining 22 FC-12 roots remain training-only.

Enrich the new TRAIN-only root selector with exactly the **first five** source-order theorems from four old, already publicly evaluated MathCan modules at the **old** source commit: `Hodge.lean`, `FundedHooks.lean`, `TOEJordan.lean` and `TOETensorConstraints.lean`. None is a RAM-07 prospective module; no previously published first-case benchmark result is used to pick a *successful proof prefix* for the extension. The extended roots are now training material for RAM-07 only and are excluded from its new heldout suite. If an extended root duplicates one of the 27 original roots, deduplicate by exact source-file+theorem and record the exclusion. Fixed total nominal roots: 47 before deduplication (27 old + 20 added).

Explore at most **12 genuinely compiler-observed states per theorem**, depth at most **5** valid tactic transitions and no more than **8** remaining goals, using only the registered RAM-07 candidate grammar. Generate no new hypotheses, no `by_cases` introducing an assumed copy of the goal, no synthetic auxiliary tautologies, no reference proof commands and no reference-prefix actions. The entire training corpus must be composed of actual Lean goal states reached from original MathCan theorem roots using the authorized tactics with source-derived arguments. An accepted candidate is a real compiler result and every new acceptance/rejection state must be **independently replayed in a separate Lean process** before fitting. Crash and request timeout results remain *unknown*, not relabeled false.

## Hypothesis reachability repair and RAM-07 registered grammar

The original RAM-06 expansion from actual MathCan showed that ordinary `intro` often introduced inaccessible local identifiers such as `word✝` and `a✝`, so the generator could not form source-name-valid scripts to apply those hypotheses. The source-only registered repair adds **one deterministic named introduction** per state: `intro ram07_arg0`, then `intro ram07_arg1`, up to `intro ram07_arg4`, chosen by counting preceding compiler-visible `ram07_argN` locals and avoiding collisions. This is a legitimate `intro` tactic on the original goal; it does not add new assumptions beyond actual binders in the original statement.

Retain all 18 previous RAM-06 action families, bounded MathCan single-definition `simp only`/`simp_all only`/`dsimp only`, and the source-type-conditioned exact application. Add one `intro_named` family, for **19 total families**, plus an expanded exact_typed script template built exclusively from compiler-visible local names:
`exact (h x y).symm` and `exact h x y` when a visible hypothesis has a local function/`obsEqOn` type, a compiler-visible `List` argument and a visible corresponding validity/implication hypothesis. A script is only a candidate; true exact matching is always determined by Lean. The generator may build at most **24 typed-exact concrete scripts** from the visible context in a fixed deterministic order, without using the theorem identifier or original reference proof. The previous `exact (h word valid).symm` historical FC-12 success is known from an original source proof, but **that source command and its local binder names are never used as a training prefix**; RAM-07 must discover the analogous application independently using only `intro_named` plus local type inspection. Report exact natural-positive source files, distinct root theorems, full script and new source hashes. A matched historical source command is a debugging comparator only, not an independent novel original mathematical proof.

The entire 19-family candidate pool is frozen before any RAM-07 prospective theorem goal inspection. All model and baseline policies receive the same contextual grammar and Lean compiler feedback. Candidate sorting must never use the new test theorem name, original source filename or gold proof script.

## Authentic positive exact gate, training and development

Before fitting, require at least **one** independently Lean-replayed, naturally collected **positive exact_typed or exact_local action from an original MathCan TRAIN root**, with no reference commands or synthetic/scaffold-generated goal assumptions. Report the complete number and source-root diversity, even if it is only one. If none is found, stop before fitting and publish a failed natural-positive-mining diagnostic; leave the new source heldout test untouched. For an optional development score, allow zero natural positive exact examples among the original five groups but keep their negative/support counts explicit.

Train loop32 (four tied passes, width32), loop64 (four tied passes, width64), and flat64 (nonrecurrent width64), seeds17,29,43; nine fits in total. For each effective batch of 64, reserve 50% of examples for original genuine multigoal states, **10% for independently verified positive natural exact** from TRAIN only, and fill from the natural ordinary TRAIN pool; keep a finite signed class-weighted four-outcome CE plus clipped goal-delta SmoothL1 weight0.2. AdamW .002/weight decay .01, gradient clip1, microbatch16×gradient accumulation4, at most1600 updates, validation every100, patience4. Checkpoint select exclusively by the original five-source theorem development loss. No synthetic auxiliary RAM-06 examples are used in RAM-07 training. Preserve all nine best/last checkpoints, optimizer and RNG states, global and per-family train/dev outcomes, exact source and model SHAs and measured CPU/GPU/RAM limits.

## Non-neural control and full-proof gate

Fixed native family ordering, train-only global family-frequency ordering and train-only smoothed shape-conditioned family-frequency ranking (prior weight8) remain unchanged from RAM-06, except they consume the new same-grammar naturally collected TRAIN dataset. The global frequency control remains a necessary comparator: a neural gain against native ordering alone is insufficient. The flat64 policy provides the nonrecurrent control.

Primary search: 12 new original target theorems × six controllers (native, global frequency, shape-frequency, loop32, loop64, flat64) at **30 seconds total wall per theorem**, at most **48 logical compiler tactic evaluations**, at most **8 successful transitions per path**, maximum8 goal branches, per-request timeout3 seconds and at most two server restarts. Secondary search: the same 12 targets × (native, global frequency, shape-frequency, loop64) at 60s/128 logical evaluations/16 transitions. Count all real processes, time, inference and branch attempts. Original-source full proof, explicit source SHA, allowed axiom list `propext`/`Classical.choice`/`Quot.sound` only, and separate byte-identical full-source Lean recompilation must finish before the score deadline. For positive scores, a third fresh full-source Lean compilation outside scoring audits the source while preserving original score verdicts.

**Primary success criterion**: at least one original new-family theorem independently kernel-certified by loop32 or loop64 and missed by **all three** non-neural baseline rankings at the registered primary budget, with zero false positive certificates. Separately report whether the same original theorem is absent from flat64 (recurrence-specific). The names of the new twelve original statements remain fixed even when all models fail.

Original source oracles, generated synthetic tasks, scratch-closure labels, historical proof prefixes, and source reproduction of older already known original theorems are **never** counted as new certified training-to-test transfer. Report every budget failure, model-only theorem, baseline-only theorem and the original 12×6 and 12×4 case matrices, with explicit limits and no post-hoc selective retries.

## Compute and isolation

GitHub-hosted CPU, Python3.11, pinned CPU PyTorch2.6.0/psutil7.1.3 and Lean4.35.0, two configured optimizer threads. Use unchanged process-tree resident RAM limit <=6GiB, optional CUDA allocation cap 2.5GiB with at least2GiB free system RAM/commit and 512MiB dedicated VRAM headroom. Separate two independent original MathCan checkouts: training at `c84e202...` and the entire heldout test context at `fca46b...`. The heldout checkout must not appear in the train-only job. No paid compute, merges or deployments.
