# RAM-06 preregistration: typed exact supervision and stronger symbolic baselines

**November 15, 2026, registered before any RAM-06 compiler collection, model fit, or new held-out theorem proof search.** This work branches from RAM-05 PR #49. FC-12/13 and RAM-01 through RAM-05 source, data, checkpoints, score reports and post-score engineering disclosures remain immutable.

## Registered question

Does training a compact tied-loop Lean tactic policy on independently verified positive **parameterized exact** examples yield a uniquely model-certified original MathCan theorem in a *new source-family-held-out* evaluation beyond both (a) the best global empirical tactic-frequency baseline and (b) a stronger, coarse-context typed symbolic-frequency baseline, with all policies sharing exactly the same argument grammar, complete-goal state and strict proof budget?

RAM-05 certified 8/12 versus fixed native 6/12, while frequency and flat also certified 8/12. The new **registered primary comparator** is the **strongest contemporaneous non-neural controller on the same test set** among native, global training-frequency, and a frozen goal-shape conditional symbolic ranking. All three are fixed before opening the new goal states; report the three controllers separately, and include the feed-forward neural control. No test-label-based policy selection is allowed. The primary success criterion is at least **one original new-family theorem** with a complete independently certified Lean proof from loop32 or loop64 that *all three non-neural comparators* miss at the same budget, and zero false certificates. A recurrence-specific gain must additionally be absent from flat64.

## Immutable Lean source, split and theorem selector

Pinned original source is `patrickdugan/MathCan@c84e2025334e9a967c1297425a490f98d5b3d189` with `leanprover/lean4:v4.35.0`. The original source contains reference proofs; those proof commands never enter model features, ranking, training labels, or candidate grammar. The final source-reference oracle is run separately after training freeze.

The new **twelve original prospective theorems** are the *first three top-level theorem declarations in original source order* in each of the four previously unused modules:

- `MathCan/FundedHooks.lean`: `conserved_along`, `mark_to_market_conserved`, `trader_gain_exact`.
- `MathCan/TOEFLRW.lean`: `form_degeneracies`, `ricci_eigen_trace`, `bivector_eigen_trace`.
- `MathCan/TOECovariantGates.lean`: `ricci_scalar_trace`, `doubled_tracefree_temporal`, `doubled_tracefree_spatial`.
- `MathCan/TOEMixedPrincipal.lean`: `one_flavor_schur_cancel`, `one_flavor_schur_invertible`, `rank_one_square_expansion`.

None of these four modules were selected as an FC-11/12/13 or RAM-03/04/05 training source, development group or prospective theorem evaluation family. This is a procedural time-ordered holdout on public MathCan source. All twelve original reference theorems must compile under the pinned toolchain as an independent 12/12 oracle, and their original proof commands are excluded from model input.

## Original genuine MathCan training and development

Read only the original FC-12 train-only source artifact (`43547762837`), verify 51 original goal-state SHA256 `403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0` and 961 original action effect semantic SHA256 `bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec`. Preserve these historic records unchanged.

Re-run the original RAM-03/RAM-05 bounded 27-MathCan-root collection under the new registered argument grammar: at most ten observed states per theorem root, depth4, max eight goals; compiler-probe each registered generated tactic candidate. Observe the complete ordered remaining goal list, visible local telescope and visible definition names. Independently replay **every** candidate observation in fresh Lean sessions before using its label. Preserve all accepted/failed script effects, exact source prefixes and source SHA256. A crash or timeout is an *unknown* observation rather than a fabricated negative.

Use the existing RAM-03 fixed five-development-theorem groups (`HookState.endpoint_append`, `HookState.pathSum_gauge`, `TOEGraph.integral_ext`, `Intervention.obsEqOn_symm`, `TOECosmology.anchored_present`). Keep all descendants and original root theorem prefixes within their split; exclude exact visible-state overlaps and archive the exclusion receipt.

## Separate provenance: generated exact curriculum

The previous RAM-05 genuine 213-state corpus had **zero positive exact_local actions**, with 718 attempted. Add a *separately labeled* training-only exact curriculum authored in `experiments/lean_ram06/ExactCurriculum.lean`. These are **synthetic auxiliary Lean theorems**, not new MathCan source theorems and never counted in original theorem coverage.

Exactly twelve curriculum theorems enter train and six *different* curriculum theorems enter development. Exercise local exact matching, symmetric equalities, left/right conjunction projections, implication applications of one and two named local arguments, and forward/backward iff elimination. Do not duplicate a curriculum declaration or goal signature across train and development. For each curriculum declaration, obtain the root from an actual Lean4.35 persistent server, instantiate **all scripts** using the same source-only RAM-06 grammar, and probe and independently replay each positive and negative candidate in a different Lean session. The original curriculum source and the SHA256 of each before/after goal and script must be recorded.

Require at least **10 distinct compiler-verified positive exact_typed TRAIN candidates** and **4 distinct positive exact_typed DEVELOPMENT candidates** and at least four observed exact-related negatives in each split before fitting. Audit failures, supports, and source overlaps are hard stops. Never copy original reference proof commands into model features; synthetic reference bodies are only audit scaffolding, and curriculum provenance must remain tagged `generated_verified_auxiliary`. Report natural MathCan positives and synthetic curriculum positives **separately**; a synthetic success is never an original certified mathematical theorem.

Maintain the original MathCan candidate data as a byte-preserved separate sub-artifact. The merged curriculum+natural model-training corpus has a NEW SHA256 and provenance manifest. All train labels require real Lean proof-effect observations.

## Immutable 18-family source-derived tactic grammar

Preserve RAM-05's seventeen registered families, controlled single visible MathCan definition `simp only`, `simp_all only` and `dsimp only` actions, and all contextual `exact_local`, `rw_local`, `rcases_local` and list-induction actions. Add exactly one new family `exact_typed`, with **syntactically bounded** scripts generated exclusively from compiler-printed local hypothesis names and their *printed types*. Allowed shapes are:

- `exact h`, `exact h.symm`, `exact h.left`, `exact h.right`
- `exact h hp`, `exact h hp hq`
- `exact h.mp hp`, `exact h.mpr hq`

The symbols `h`, `hp`, `hq` are local variable/hypothesis names with strict identifier validation. Only generate application arguments whose printed types are syntactically compatible with the antecedents of a local implication/iff (a conservative static prefilter); only allow equality symmetry from visible equality hypotheses and conjunction projections from visible conjunction hypotheses. At most 18 typed-exact candidates per state. All candidates are independently checked by actual Lean. Plain `exact_local` family retains its historical action candidates; source-derived candidate string generation, guard, family IDs, and candidate budgets must be frozen *before* prospective evaluation.

## Models and stronger non-neural controls

Train `loop32` (width32, four shared passes), `loop64` (width64, four shared passes), and `flat64` (width64, no recurrent cell) from the new independently verified 18-family corpus with fixed seeds17,29,43. Train all nine, select one checkpoint per architecture on the balanced old five-source plus six-curriculum-theorem development-only loss. Preserve the legacy 0.5 multigoal-state sampling, and reserve 20% of the effective batch for compiler-verified positive `exact_typed` auxiliary TRAIN examples so the new class is seen. Keep the effective batch64, 16×4 accumulation, AdamW lr.002, decay.01, grad norm1.0, at most1600 updates, validation every100, patience4. Weighted four-outcome classification plus 0.2 clipped goal-count delta SmoothL1. Early stopping uses development only. Save best/last checkpoint, optimizer, sampler and global RNG state, exact checksum, RAM, CPU/GPU calibration and seed selection. The original RAM-05 model weights are preserved and cannot be substituted as RAM-06 trained models.

Three frozen symbolic baselines:
- **Native:** the original deterministic registered family/script order, no learned frequencies or model scores.
- **Global frequency:** mean progress/closure utility of each action family from the independently verified combined TRAIN subset, with no dev or test outcomes.
- **Shape-conditioned frequency:** train-only smoothed family reward per (single-vs-multi-goal state, equality/implication/inequality/conjunction/other goal-shape) cell, shrunk to the global TRAIN family mean with fixed prior weight eight. Equal empirical evidence gets stable original family/script order, no Lean feedback cache shared with another controller.

All policies see the same compiler goal, candidate grammar, search budget, and independent source proof gate. No choice of best baseline using test labels changes how that baseline ranks; report both separate and the maximum fixed non-neural count only for assessing primary model-only coverage. A local training-only frequency control that also solves any model-only theorem prevents a claim of model-only gain over strongest baselines.

## Frozen search and scoring

Primary: twelve original theorems × six controllers (native, global frequency, shape frequency, loop32, loop64, flat64); per theorem **30s total wall, 48 logical Lean tactic evaluations, 8 transitions**, max 8 simultaneous goals, request timeout3s, up to two server restarts. Secondary: all twelve exact originals × (native, global frequency, shape frequency, loop64), **60s, 128 evaluations, 16 transitions**, same compiler process/timeout accounting. Compile exact script in the original source context; every goal branch must close; `#print axioms` may show only `propext`, `Classical.choice` and `Quot.sound`, or none. No sorry, sorryAx or extra axioms. A second full source-identical replay and its SHA256 must finish before deadline to score; a separate third external Lean recompilation audits every accepted proof after score without changing its result. All candidate failures, source bytes, process exits and proof gate outcomes are preserved.

**Registered success:** at least one complete original theorem only a recurrent policy solved and all three non-neural comparators missed at the same primary budget, with zero false positives. Separately report recurrence-specific gain, meaning also absent from flat64. A new curriculum exact-positive is *training progress*; cached top1 or scratch closure is not a certified original theorem. Report original 12×6 and 12×4 case matrices, actual process/resource counts, model-only/native-only sets, and training/verification cost. No post-hoc partial retries or score replacement. An engineering defect found after opening heldout outcomes requires disclosure and a new complete all-policy comparison run with frozen weights and source names.

## Resource and software limits

Use GitHub-hosted CPU with Python3.11, CPU PyTorch2.6.0, psutil7.1.3 and pinned Lean4.35, two CPU threads. Native Windows RTX3050 with 4GB VRAM is optional for local reproduction; monitor whole process-tree RSS <=6GiB, allocated GPU <=2.5GiB and preserve RAM/VRAM headroom; choose CUDA only if identical small-model calibration is at least 1.15× faster than CPU. Preserve the RAM-05 and all earlier branches; no merge, deployment or paid compute.
