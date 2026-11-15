# RAM-05 preregistration: bounded visible unfolding and fresh Lean family transfer

**Frozen 2026-11-15 before RAM-05 data collection, fitting, or any RAM-05 heldout proof-state access.**
This is a new experiment stacked on RAM-04 PR #42. FC-12/13 and RAM-01/02/03/04 weights, original outcomes, partial-run failure evidence, and audit receipts are immutable.

## Motivating observation and question

The fully audited RAM-04 prospective result certified 9/12 original MathCan theorems for the recurrent, flat, frequency, and native policies. Its failures included one fatal recursive simplifier loop (FoundryProbe.two_currency_zero_at_lock) and two proof-search budget exhaustions. Inspection of the **already published RAM-04** logs showed unbounded visible-definition expansion in scripts such as `simp_all [MathCan.V4Transient.count, MathCan.V4Transient.nz]` and failures of generic `simp_all` in that same case. Those prior cases can be used for engineering diagnosis and labeled post-hoc tests only.

The **RAM-05 prospective question** is whether a small recurrent, bounded-unfolding policy trained on compiler-observed outcomes can independently certify a new original theorem in fresh source families which same-grammar contemporary native and training-frequency baselines miss at the same strict search budget.

## Immutable research source and training boundaries

Pinned source: `patrickdugan/MathCan@c84e2025334e9a967c1297425a490f98d5b3d189`, Lean4.35.0, untouched original proofs.

Original FC-12 TRAIN roots: 27 original theorems, 51 compiler-observed original states, SHA256 `403331aba9c9f127f6ac169bb8b732de8a7b56e934f6cb78b4375d1c59732bf0`; 961 original compiler-probed candidate effects with semantic SHA256 `bd713c7fd3d9d2e82f1304d7556f7a25bfa8b1a0e416dbd2e8d271dc4ad047ec`. Download the pinned train-only artifact from GitHub workflow `43547762837`. Source-gym commit `e8a780254b5cf41d479095f4f22bcfc4aa76ed72`.

Use the RAM-03 original *train-only* source selection, including its development groups:
- `MathCan/HookState.lean:endpoint_append`
- `MathCan/HookState.lean:pathSum_gauge`
- `MathCan/TOEGraph.lean:integral_ext`
- `MathCan/Intervention.lean:obsEqOn_symm`
- `MathCan/TOECosmology.lean:anchored_present`

Collect from all 27 original TRAIN roots with at most 10 genuine states per theorem, depth4 and <=8 concurrent goals. The corpus is rebuilt under a new registered safe action grammar. Every candidate result must come from Lean and be independently replayed in new Lean sessions, with full ordered before/after goals, source hashes and compiler outcomes. Preserve the 961 original legacy labels and the previous RAM-03/04 corpus as independent evidence; the new corpus has a distinct fingerprint and may have different label counts. Timeouts and unexpected terminated compiler processes are **unknown observations**, not false tactics or training negatives. If the collection fails an independent replay or has zero positive multigoal-input training/development examples, stop before fitting and publish the failure.

## The first RAM-05 twelve prospective original theorems

Freeze the **first three** original top-level theorem declarations, in source order, from exactly the following four files. These modules are disjoint from all FC-11/12/13, RAM-03, and RAM-04 selected original train, development and held-out research suites:

- `MathCan/TOETwoMultiplet.lean`: `two_multiplet_cone_matching`, `compensation_positive`, `aligned_flavor_wedge`.
- `MathCan/TOEYangMills.lean`: `opposed_electric_mixed_zero`, `opposite_electric_norm`, `diagonal_match`.
- `MathCan/TOEConeMatching.lean`: `matched_cones_have_root`, `cone_gap_at_first_root`, `mismatched_cones_lift_first_root`.
- `MathCan/TOETensorConstraints.lean`: `source_cone_match_iff_compensation`, `rank_three_needs_two_multiplets`, `one_multiplet_cannot_have_rank_three`.

The original theorem names and pinned source tree were inspected to identify these modules. No RAM-05 prospective compiler goal state or reference proof command is used for any policy, configuration, tactic risk guard, training split or seed selection. Compile all twelve untouched original reference proofs separately as the 12/12 oracle **after** checkpoint freeze. This is a time-ordered source-family holdout over public theorems, not a cryptographically hidden research benchmark.

## Bounded simplification tactic grammar, frozen before new training

Retain RAM-03 goal-state representation, contextual hypothesis parsing, remaining sibling-goal list, and already registered safe actions `omega`, `rfl`, `simp_all`, `assumption`, `intro`, `constructor`, `exact_local`, `simpa_local`, `rw_local`, `apply_local`, `rcases_local`, `cases_local`, `induction_list` and `all_goals simp_all`. Keep actual local-context argument enumeration in the existing fixed order.

Replace unbounded MathCan *definition-bulk* simp/dsimp families with **single visible constant, only-mode** actions:
- `simp only [MathCan.VisibleDefinition]`
- `simp_all only [MathCan.VisibleDefinition]`
- `dsimp only [MathCan.VisibleDefinition] at *`

A `VisibleDefinition` must be an actual compiler-observed, locally used MathCan definition constant from the original goal/hypothesis telescope, matching the fixed identifier grammar. Generate at most three constants per family in deterministic sorted order, with distinct concrete scripts. The model does not nominate arbitrary global lemmas or hallucinate references. The new grammar has 17 registered families (original 17 minus unfold_simp, arith_norm, dsimp_visible plus the three bounded families); every policy receives the **same** candidate pool. No all-definition bulk rewrite and no arbitrarily named rewrite hints. The old RAM-04 simplifier crash remains an explicitly labeled *post-hoc* safety diagnostic; do not change the new grammar after seeing RAM-05 test results.

Lean execution must treat a fatal subprocess exit as an **unknown candidate**, bounded by the same at-most-two server restarts, and must retain the raw original error logs. No recursion-depth or heartbeat increase after preregistration. The original persistent Lean server and compiler trust boundary are otherwise unchanged.

## Model and baseline freeze

Reuse RAM-03 tokenization/feature representation and four-outcome plus goal-count-delta training objective with the new same-family semantic action mapping. Architectures: `loop32` (four tied passes, width32), `loop64` (four tied passes, width64), and `flat64` (zero recurrent passes, width64). Seeds: 17,29,43.

Training: AdamW(learning-rate .002, weight-decay .01), effective batch64 with microbatch16×gradient accumulation4, 0.5 multigoal-input training sampling, class-weighted cross entropy plus 0.2 clipped goal-count delta SmoothL1, grad norm cap1.0, max1600 updates, validate every100, patience4. Checkpoint and architecture selection uses only the original five-theorem development groups, balanced single/multigoal loss. Train all nine fits. Keep optimizer and RNG state and SHA256 for every best and last checkpoint. No RAM-05 heldout-source access until weights, development seed choices, independent train-effect audit, fixed grammar and source hashes are frozen.

Native search tries the same bounded-only candidate pool in the frozen family order. The family-frequency ranking depends exclusively on genuine train-only observed labels. Model rankers receive only original visible Lean goal-state text, sibling-goal text/count and currently instantiated local actions; filenames, theorem names, teacher scripts, future states, test labels and gold reference proofs are excluded from input features.

## Search costs and proof trust gate

Primary comparison: all five policies (native, train-only frequency, loop32, loop64, flat64), twelve exact cases, **30s wall, 48 logical evaluations, eight successful transitions per path, maximum eight goals**, per-candidate timeout3s and at most two process restarts.

Secondary comparison: native and loop64 only, same twelve exact original cases, **60s wall, 128 logical evaluations, sixteen transitions**, same per-request3s and restart cap. These are fixed before evaluation. Account for each actual process, session startup and teardown, inference, failed attempts, final kernel compile and independent same-source replay. A proof finishing after the time cutoff earns zero credit. Preserve process-crash unknowns and every failed candidate. Syntactic goal-state dedup preserves goal order and sibling count.

Certificate requires every remaining goal closed, a separate complete original theorem compilation with exactly the selected script, no `sorry` or `admit`, and `#print axioms` permitting only `propext`, `Classical.choice` and `Quot.sound` (or no axioms). Then replay **byte-identical full final Lean source** in a fresh process before scoring success. After scoring, additionally recompile every accepted original complete proof in another independent Lean process; this third replay cannot alter frozen coverage.

## Registered success and inference interpretation

Primary learning success requires at least one uniquely **model-only original theorem** certified by loop32 or loop64 on the new twelve-case source-family suite, absent from the same-budget native baseline, with zero false positives. Report separately whether this gain is also absent from both flat64 and training-only frequency (recurrence-specific evidence). A training classifier improvement, scratch goal closure, or untrusted compiler trace is never counted as a completed theorem.

Cost secondary outcomes include total seconds, evaluations, process restarts and RAM per certified proof, with all failures counted in compute cost. Report all 12 cases for every policy, including no-progress, timeout, and budget exhaustion. A discrepancy or post-heldout engineering fix invalidates the affected whole comparison and requires a complete, separately labeled rerun, retaining the initial artifacts and unchanged checkpoints. Earlier RAM-04 three unsolved theorems may be evaluated only as **post-hoc** diagnostics, excluded from the new primary score.

## Resource budget

CPU-only GitHub CI for initial training/evaluation. Record Python, CPU-only PyTorch and Lean version, 2 CPU threads, peak entire process-tree RSS. Apply unchanged RAM-03 guard: <=6GiB RSS, <=2.5GiB allocated CUDA, >=2GiB free RAM/commit, >=512MiB free dedicated GPU memory. On the user's 4GiB RTX3050, calibrate identical CPU/GPU FP32 updates and choose CUDA only if >=1.15× CPU throughput. No paid resources, deployment, or merge.
