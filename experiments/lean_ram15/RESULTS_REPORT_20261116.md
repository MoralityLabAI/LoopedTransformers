# RAM-15 | Prospective structural-grammar transfer on fresh MathCan NOR families

**Date:** 16 November 2026  
**Status:** Completed first fresh-source full-search evaluation; all in-budget proof certificates independently checked; registered recurrent-only criterion **failed**.  
**Repository:** `MoralityLabAI/LoopedTransformers`, pinned original MathCan `0e899d07d077146973c20ba7777a2d7d4e9b8abd`.

## Scientific finding

RAM-14 discovered, retrospectively, that a type-directed natural-number induction macro and source-visible previous-lemma rewrites could prove two historical `GramianTransfer` theorems missed by the original 20-family tactic grammar. RAM-15 tests whether those grammar additions transfer to **14 original, source-order theorems from seven new NOR15–NOR18 MathCan modules**, none present in the previous RAM-13 source commit.

The original protocol and seven module Git blob hashes were frozen in commit `b5b7836755a457734fa612e335c48c1c64549f40` before opening any of the selected source files. The 14 theorem names and matched budgets were locked in commit `b4a7aabd68c062ec2583dbabb4a1c6c551c620b3` before compiler goal access. All six primary policies received the **same source-derived candidate set**, including the structural macros. The three old recurrent/flat models used unchanged RAM-08 checkpoint bytes and the original 20-family embedding, with transparent frozen proxy labels for newly proposed tactics. Training-only original tactic data supported the two empirical symbolic controls.

### Registered complete-source proof results

| Controller | Primary: 25s/48 calls | Secondary: 50s/96 calls | Original 20-family ablation |
|---|---:|---:|---:|
| Native source-visible order | **7/14** | **7/14** | **6/14** |
| TRAIN-only global-frequency | 7/14 | 7/14 | |
| TRAIN-only shape-frequency (prior 8) | 7/14 | 7/14 | |
| Looped32, four recurrent passes | 7/14 | | |
| Looped64, four recurrent passes | 7/14 | 7/14 | **7/14** |
| Flat64, no recurrent passes | 7/14 | | |

The complete preregistered matrix consists of **84 primary + 56 secondary + 28 original-grammar ablation = 168** policy/theorem searches. Exactly **83** produced two in-budget, full-original-theorem Lean4.35 source certificates and an identical independent compiler replay. A separately executed third full-source Lean4.35 audit passed on **all 83**, giving **249 successful source compilations** and **zero detected false certificates**. Thirteen accepted policy cases used at least one new structural tactic macro.

**Registered result:** zero original test theorems were certified by either recurrent model and missed by all three symbolic controllers. Flat64 proved exactly the same seven as Looped64. The larger secondary budget gave **zero additional theorem coverage**. The original-grammar ablation showed a one-theorem gain for native ordering with the expanded grammar and **zero gain for Looped64**. The native-only gain was `NOR15VerifierAlias.firstAccepted_congr`. It is a grammar coverage result because every controller used the same candidate set and native ordering itself gained a proof.

### Seven source families, original selected theorem roots

| New MathCan module | Original first two theorems | Certified by each expanded primary policy |
|---|---|---:|
| `NOR15MaskCapacity` | `partition_success_bounded_by_signature_capacities`; `signature_shortfalls_add_to_group_sizes` | 0/2 |
| `NOR15TrajectoryObservability` | `trajectory_pair_reconstruction_bound`; `near_endpoint_pair_forces_large_error` | 0/2 |
| `NOR15VerifierAlias` | `firstAccepted_congr`; `identical_public_masks_force_same_first_success` | 1/2 |
| `NOR16PublicOutputCertificate` | `sound_complete_public_output_certificate`; `unique_accepted_candidate_certifies_future` | 2/2 |
| `NOR16PublicQueryConsensus` | `predictive_consensus_mass_dominates`; `posterior_correct_mass_ceiling` | 2/2 |
| `NOR17BudgetedPublicCertificate` | `exact_public_sequence_agreement_implies_prefix`; `true_program_survives_public_prefix_filter` | 2/2 |
| `NOR18CachedVerifierWork` | `prefix_agreement_refines`; `prefix_rejection_persists` | 0/2 |

These failures are finite search-budget outcomes and do not constitute Lean unprovability certificates.

## Native proof trust and provenance

The private first-scored run [**43798790112**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43798790112) successfully executed all **14 independent case jobs** with the 25/48/10 and 50/96/16 registered limits. Each job verified the byte-frozen original RAM-08 training ZIP (`a2ac363547858871c4e066c96bac67253dfeb7ff4e2a0050bb5886c8ca9edcfc`), its 20,930 independently replayed original compiler effects (`f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f`) and all three original model checkpoint SHA-256 digests **before opening new prospective compiler goals**. All native compilers ran Lean 4.35.

The in-run third-audit job encountered `lake: command not found` because `GITHUB_PATH` did not apply until a later shell step. It failed before accessing the original proof receipts. [**Audit amendment A**](AUDIT_AMENDMENT_A.md) records this independently from scientific scores. The repair [**audit-only run 43799424205**](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43799424205) downloaded **exactly the original first-run case artifacts**, corrected PATH for its own Lean process, and reconstructed every accepted proof byte-for-byte without rescore. Every third compilation passed with permitted `#print axioms` inventory (none or only `propext`, `Classical.choice`, `Quot.sound`). The final audit archive is GitHub artifact **14195782928**, SHA-256 `8c76076abc938a2477f45b3561c63a7c2b365dc759a9c1ce2f804f47c177b47a`, size 5,245,132 bytes.

No private model checkpoint bytes or old RAM-08 training corpora were uploaded to public MathCan. Original RAM-08 PEF and RAM-10 evaluation scores remain independent and unchanged.

## Candidate-generation and recurrence interpretation

The extended grammar adds three source-only structural families: `nat_induction` for typed natural-number binders, `simp_prior` for source-visible definition/structure rewrites, and `rewrite_prior` for eligible preceding theorem headers. Its inference adapter maps the three new script types to existing 20-family embeddings without modifying trained parameters. Candidate order for native search is fixed by inserting one new structural action after every four original actions. All controller ranks operate on the same concrete action multiset with an upper bound of 32 added scripts per state.

Let `C_G(s)` denote the original grammar's concrete actions at state `s`, and `C_{G^+}(s)` the expanded grammar. Then `C_G(s) \subseteq C_{G^+}(s)`, with the extension's provenance and step budget recorded. A fixed tactic evaluation budget means coverage need not improve monotonically: new candidates consume opportunities, even though the underlying action vocabulary contains the old one. The native one-case improvement is a *real compiled theorem* and the recurrent-only target remains unmet; the effect of the new grammar must be measured with the same per-policy budget.

### Search cost and controls

| Phase | Policy | Genuine tactic evaluations across all 14 cases |
|---|---|---:|
| Primary | Native / global frequency / shape frequency | 430 / 365 / 361 |
| Primary | Looped32 / Looped64 / Flat64 | 365 / 383 / 361 |
| Secondary | Native / global frequency / shape frequency / Looped64 | 714 / 649 / 645 / 667 |
| Original grammar ablation | Native / Looped64 | 456 / 381 |

The expanded native controller used fewer tactic calls and proved one more theorem than its original-grammar control. The recurrent model's proof coverage was unchanged while the expansion slightly increased its cumulative tactic evaluations.

## Reproduce and interpret

- [Frozen preregistration](PREREGISTRATION.md) and [exact 14-case source manifest](LOCKED_MANIFEST.json).
- [Shared structural source grammar](ram15_grammar.py), [frozen six-model scorer](harness.py), [per-theorem runner](run_matrix.py), [source-only tests](test_contract.py).
- [All 168 original scores](RAM15_CASES_20261116.jsonl) and [machine-readable independent audit summary](RAM15_RESULT_20261116.json).
- [Original private run](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43798790112) and [successful independent replay](https://github.com/MoralityLabAI/LoopedTransformers/actions/runs/43799424205).

**Limitations.** This is a fresh *source-file family* holdout on 14 publicly visible original theorem roots, with within-module correlations and shared Lean vocabulary. The original models were trained on earlier mathematical families. The structural macros come from a posthoc result on previously exposed Gramian statements, though all RAM-15 choices were frozen before scoring its new cohort. Historical frozen-model rankings provide a valid matched control while a new specialized 23-family model would require a separate train/DEV/test preregistration. The high number of failed cases suggests stronger syntactic proof planning and controlled multi-step action generation; the present experiment establishes no recurrent-only proof advantage.

**Next research milestone:** RAM-16, train source-visible Nat-induction and prior-lemma retrieval on old source families, then evaluate on an unopened later MathCan revision with both matched strong symbolic controllers and param-matched flat policies. Freeze candidate caps and model selection before opening those theorems, retaining all RAM-15 results as previously viewed public cases.
