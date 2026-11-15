# RAM-08 preregistration: natural exact diversity and PEF source-family transfer

**November 15, 2026, frozen before RAM-08 training, model selection or new heldout compiler-state access.** This experiment is stacked on RAM-07; RAM-01 through RAM-07 results, SHA-pinned artifacts and negative outcomes remain unchanged. No paid compute or merge.

## Original sources and untouched prospective theorem selector

TRAIN/DEV source: `patrickdugan/MathCan@c84e2025334e9a967c1297425a490f98d5b3d189` with `leanprover/lean4:v4.35.0`. Original FC-12 source gym `MoralityLabAI/Compactification-Theory@e8a780254b5cf41d479095f4f22bcfc4aa76ed72`, the same 27 original theorem roots, original source-state SHA256 `403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0` and original 961 candidate effects SHA256 `bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec` from original run `43547762837`.

**New TEST-only original MathCan source**: unmerged PR #44 head `patrickdugan/MathCan@83f61997948818b68da94429c2fe736d93ea3093`, also Lean4.35.0 and independently green Lean CI. Exactly the **first three top-level theorem declarations**, original source order, from each of the following **four newly introduced PEF modules**:

- `MathCan/CausalInteraction.lean`: `four_corner_decomposition`, `linear_four_corner_interaction_zero`, `quadratic_interaction_witness`
- `MathCan/OrderedTangent.lean`: `upper_after_lower_general`, `lower_after_upper_general`, `shear_commutator`
- `MathCan/PairedHistory.lean`: `suffixPairCorrect_one`, `pairedSuffixAccuracyExactHalf`, `suffixOnlyCannotDecodeBoth`
- `MathCan/PrecisionEntropyRobust.lean`: `quaternary_four_step_suffix_gap`, `quaternary_four_step_gap_attained`, `quaternary_gap_robust_under_16_integer_units`

The four new modules have dependencies including `MathCan/PrecisionEntropy` and `MathCan/PrecisionEntropySchedule`; build all from this exact pinned **new source revision** in a separate test-only job. These files were absent from the old training source and from all previous FC/RAM prospective source-family test selections. Their names, imports, first-three declarations, original source hashes and PR CI statuses were used for selection. Do not open their proof commands, tactic goals or candidate outcomes before training freeze. Test source is public and unmerged, so this is a **procedural time-ordered original source-family holdout**, not a cryptographic hidden evaluation. All twelve original reference proofs must compile separately as the 12/12 oracle; their original proof commands never enter model features, training or tactical argument generation.

## Expand genuine natural-source exact training diversity

Preserve the **47 original RAM-07 train/dev roots**, including its five unchanged development-only groups: `HookState.endpoint_append`, `HookState.pathSum_gauge`, `TOEGraph.integral_ext`, `Intervention.obsEqOn_symm`, `TOECosmology.anchored_present`.

Add exactly these **18 old original MathCan source TRAIN roots** selected on theorem-statement syntax (conjunction, implication, local equality and proof hypotheses), without consulting any proof commands:

1. `TOEMixedPrincipal.independent_kinetic_and_spectral_gates`
2. `TOENonAbelian.zero_mixed_nonzero_source`
3. `TOEFormSectors.two_helicity_kernel_removed`
4. `TOECovariantGates.positive_two_square_zero`
5. `TOEYangMills.diagonal_stationary_mixed_running`
6. `TOETensorCompletion.einstein_derivative_vanishes`
7. `TOETripletJordan.longitudinal_coderivative_not_gauge`
8. `TOETripletJordan.exact_lightcone_null`
9. `Intervention.obsEqOn_action_subset`
10. `Intervention.obsEqOn_postcompose`
11. `Intervention.obsEqOn_action_stable`
12. `GraphAuthorization.service_iff_waste_within_budget`
13. `GraphAuthorization.avoidable_regret_iff_excess_slack`
14. `HookMultiTick.route_conservation`
15. `Settlement.path_superpotential_bound`
16. `FiniteHorizonGap.epsilon_sixth_equiv`
17. `DeletionMemory.correct_decoder_separates`
18. `TOEDressedMatter.flrw_temporal_tracefree_iff`

Every listed theorem is in `MathCan/<module>.lean` at the **old** pinned source. Deduplicate on exact (source_file,theorem), report all duplicates, and assert no extra root belongs to the five development groups. Nominal 65 original roots before deduplication.

For each actual original source theorem, explore at most **18 genuine Lean compiler states**, at most **six accepted tactic transitions**, maximum eight pending goals, with deterministic canonical state de-duplication and the original RAM-07 source-only named-introduction scheduling. Add a predeclared **constructor-successor priority**: after a genuine compiler-accepted `constructor` creates multiple goals, place its original goal state at the beginning of the existing multigoal queue. This changes training-state scheduling only and never fabricates a hypothesis, goal, proof or label.

Every new candidate must be probed by Lean4.35 and **independently replayed in fresh Lean sessions** with matching full before/after pending goals, source SHA and label. Fatal process exits and timeouts are `unknown`, never relabeled negative. Preserve all failed attempts. Original proof body and teacher prefix commands are excluded from data collection and model inputs.

**Training hard gate**: at least **two distinct old-source TRAIN root theorems** with positive independently checked `exact_typed` or `exact_local` actions, and at least **three positive candidate observations** across those roots, after deduplicating identical (theorem,state_hash,script) observations. Each must arise by real registered tactic transitions from the original MathCan source theorem root. Synthetic `ExactCurriculum` examples, artificial assumption injection, source-reference proof copies and previously heldout PEF actions are excluded. If this minimum fails, stop before fitting or opening PEF test proof states and publish the original failed collector receipt.

## Registered source-visible grammar and strong controls

Retain the **19 RAM-07 source-visible tactic families** including deterministic `intro ram07_argN`, typed original local hypothesis applications, exact symmetry/transitivity, parameterized `rw`, `rcases` and `induction`, bounded one-visible-definition `simp only`/`simp_all only`/`dsimp only` and arithmetic controls. Add **one** universal core Lean4.35 tactic `decide` as family 20, in a fixed position after `rfl` and before `simp_all`. It is available to all neural and symbolic controllers under identical budgets. No Mathlib-only `ring`/`norm_num` and no arbitrary new math-library lemma retrieval. Every generated source-based action is still individually checked by Lean.

Train the same three sub-250k policies, `loop32` (width32, four tied recurrent passes), `loop64` (width64, four tied passes), `flat64` (width64, zero recurrent passes), each seeds17/29/43 for **nine fits**. Use the registered AdamW LR.002, decay.01, effective batch64 (microbatch16×grad accumulation4), 50% real multigoal-state inputs, **10% independently verified natural exact-positive TRAIN** observations, other original TRAIN rows for the balance; 1600 optimizer updates maximum, val every100, patience4, grad clip1, weighted four-outcome CE plus clipped signed goal-count delta SmoothL1 weight0.2. Checkpoint/seed selection uses only the same five old original MathCan development theorems, with all model weights, optimizer/RNG state and source/data SHA256 frozen **before opening test source**.

Three non-neural controls: (1) fixed native family/script order, (2) global TRAIN-only family-frequency ranker, (3) shape-conditioned TRAIN-only family-frequency shrinkage with fixed prior8, all on the same 20-family candidate pool. No model/test theorem names, source filenames, reference scripts, future goal labels or gold information can enter a policy ranking or training example.

## Original theorem certification and success rule

Primary comparison: exactly 12 new PEF theorem declarations × six policies (native, frequency, shape_frequency, loop32, loop64, flat64), with **30 seconds total wall per theorem**, at most **48 actual logical tactic evaluations**, eight successful transitions per path, no more than eight pending goals, 3-second per-call timeout and two process restarts. Secondary comparison: same twelve original declarations × (native, frequency, shape_frequency, loop64) at **60 seconds, 128 evaluations and sixteen transitions**, with the same process and proof trust boundary. Account for every failure, compiler process and total resource use.

Each accepted proof closes every original goal, compiles a separate complete original PEF theorem with only its selected registered tactic script, **no `sorry` or `admit`**, and explicit `#print axioms` permitting only `propext`, `Classical.choice` and `Quot.sound` or no axioms. A second byte-identical original source compilation must pass **inside** the registered theorem deadline. A third post-score external Lean source replay verifies every accepted source hash, executable script whitelist and axiom inventory while leaving all original scores unchanged.

**Registered primary criterion:** at least one original PEF new-source-family theorem certified by a recurrent policy but missed by **all three non-neural** controllers under the primary same-grammar budget, with zero false-positive certificates. Separately report recurrence-specific original theorems missed by flat64. Publish full per-theorem case matrix, positive and negative results, natural exact source-root diversity, failed search reasons, wall/compiler costs and model SHA. A model-only result against the weaker native ordering alone is insufficient. The original source reference oracle never counts as a learned theorem.

## Resource and integration limits

Run on GitHub-hosted Ubuntu CPU with Python3.11, pinned PyTorch2.6.0 CPU, psutil7.1.3, Lean4.35, two CPU threads and whole-process RSS cap6GiB. Local Windows RTX3050 4GiB VRAM and ~10GiB free system RAM support CPU fallback; CUDA is selected only when the same pilot exceeds CPU by >=1.15× with allocated VRAM<=2.5GiB and >=2GiB free system RAM headroom. Old original TRAIN source and new unmerged PEF test source are **separate pinned checkouts and separate CI jobs**; never copy test files into the train job. Keep the PR unmerged.
